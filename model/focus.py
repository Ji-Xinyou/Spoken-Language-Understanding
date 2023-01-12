#
#! This is not used in this project
#! This is not used in this project
#! This is not used in this project
#
import torch
import torch.nn as nn
import torch.nn.utils.rnn as r

from torch.autograd import Variable
from utils.batch import Batch

'''
An implementation of a two-lstm model, revised from focus

1. This model first passes the sequence to a bidirectional Recurrent NN(GRU, LSTM, ...)

2. Then used the
        - first token's reverse output
        - last token's forward output
    stacked as the second unidirectional Recurrent NN's h_0(hidden)

3. The input sequence of the second Recurrent NN is the output of the first Recurrent NN

NOTE: the decode() method is exactly the same as the baseline model
'''

class FC(nn.Module):

    def __init__(self, input_size, num_tags):
        super(FC, self).__init__()
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, hiddens, labels=None):
        logits = self.output_layer(hiddens)
        # make masked value extremely small
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

class FocusModel(nn.Module):
    '''
    An implementation of Focus
    Must have in cfg:
        vocab_size, embed_size, hidden_size, num_layer, dropout, tag_pad_idx, num_tags
    '''

    def __init__(self, cfg):
        super(FocusModel, self).__init__()
        self.cfg = cfg
        # always embed the word first
        self.word_embed = nn.Embedding(cfg.vocab_size, cfg.embed_size, padding_idx=0)
        # encoder lstm, by default 2 layer bidirectional
        # in:  (N, seq_len, embed_size)
        # out: (N, seq_len, hidden_size)
        self.enc_lstm = getattr(nn, cfg.encoder_cell)(
                                input_size=cfg.embed_size,
                                hidden_size=cfg.hidden_size // cfg.num_layer,
                                num_layers=cfg.num_layer,
                                bidirectional=True,
                                batch_first=True
                                )
        self.enc_dropout = nn.Dropout(p=cfg.dropout)

        # decoder lstm, by default this is a unidirectional one
        # in:  (N, seq_len, hidden_size // num_layer), half the previous model output's dim
        # out: (N, seq_len, hidden_size)
        self.dec_lstm = getattr(nn, cfg.encoder_cell)(
                                input_size=cfg.num_tags,
                                hidden_size=cfg.hidden_size,
                                num_layers=1,
                                bidirectional=False,
                                batch_first=True
                                )
        self.dec_dropout = nn.Dropout(p=cfg.dropout)

        self.out = FC(cfg.hidden_size, cfg.num_tags)
    
    def forward(self, batch: Batch):
        '''
        Convert a batch to output and loss
        '''
        #REMINDER: use loaded word2vec before calling this
        embedding = self.word_embed(batch.input_ids)
        packed = r.pack_padded_sequence(embedding,
                                        batch.lengths,
                                        batch_first=True,
                                        enforce_sorted=True)
        # here enc is a concatenated of forward and reverse hidden states at each timestep
        enc, (hn, cn) = self.enc_lstm(packed)
        enc, _ = r.pad_packed_sequence(enc, batch_first=True)
        # enc = self.enc_dropout(enc)
        (N, L, DH) = enc.shape
        D = 2
        H = DH // D
        _enc = enc.view((N, L, D, H))
        # get the h_left, a.k.a the backward output of the first token
        # (N, L, cfg.hidden // cfg.num_layer), e.g. (32, 256(512 // 2))
        first_back = _enc[:, 0, 1, :]

        # decoder:
        #   hidden: prev hidden + encoder's hidden states (timestep corresponding)
        #   cell:   prev cell
        #   input:  prev output (a tag)

        # each is a logit of tags
        input = torch.zeros((len(batch), 1, self.cfg.num_tags), requires_grad=True).cuda()
        hidden = torch.zeros((1, len(batch), self.cfg.hidden_size), requires_grad=True).cuda()
        cell = torch.cat((first_back, first_back), dim=1).unsqueeze(0).cuda()
        dec_len = enc.shape[1]

        loss = 0
        prob = torch.zeros((len(batch), dec_len, self.cfg.num_tags), requires_grad=True).cuda()

        for t in range(dec_len):
            prev_hidden = enc[:, t, :] # (Batch, hidden_size)
            prev_hidden = prev_hidden.reshape(1, prev_hidden.shape[0], -1)
            hidden += prev_hidden
            out, (hidden, cell) = self.dec_lstm(input, (hidden, cell))
            p, l = self.out(out, batch.tag_ids[:, t])
            loss += l
            prob[:, t, :] = p.squeeze(1)

            tmp = torch.zeros((len(batch), 1, self.cfg.num_tags), requires_grad=True).cuda()
            for i in range(len(batch)):
                tmp[i, 0, p.argmax(-1).squeeze(1)[i]] = 1
            input = tmp

        return prob, loss

    def decode(self, label_vocab, batch):
        '''
        todo: what
        '''
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]

            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)

            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)

        return predictions, labels, loss.cpu().item()
