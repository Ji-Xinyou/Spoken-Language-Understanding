import torch
import torch.nn as nn
import torch.nn.utils.rnn as r

from model.slu_baseline_tagging import TaggingFNNDecoder
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

class TwoLSTM(nn.Module):
    '''
    A Focus-like Two Lstm Model for SLU.
    Must have in cfg:
        vocab_size, embed_size, hidden_size, num_layer, dropout, tag_pad_idx, num_tags
    '''

    def __init__(self, cfg):
        super(TwoLSTM, self).__init__()
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
                                input_size=cfg.hidden_size // cfg.num_layer,
                                hidden_size=cfg.hidden_size // cfg.num_layer,
                                num_layers=cfg.num_layer,
                                bidirectional=False,
                                batch_first=True
                                )
        self.dec_dropout = nn.Dropout(p=cfg.dropout)

        self.out = TaggingFNNDecoder(cfg.hidden_size // cfg.num_layer, cfg.num_tags, cfg.tag_pad_idx)
    
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
        enc, _ = self.enc_lstm(packed)
        enc, _ = r.pad_packed_sequence(enc, batch_first=True)
        enc = self.enc_dropout(enc)

        # use h_left to initialize the s0 as the paper said
        # (N, L, dir, hidden)
        (N, L, DH) = enc.shape
        D = 2
        H = DH // D
        _enc = enc.view((N, L, D, H))

        # get the h_left, a.k.a the backward output of the first token
        # (N, L, cfg.hidden // cfg.num_layer), e.g. (32, 256(512 // 2))
        first_back = _enc[:, 0, 1, :]
        # unsqueeze the D * num_layers
        # get the h_right, a.k.a the forward output of the last token
        last_forward = _enc[:, -1, 0, :]

        # stacked = torch.stack((first_back, first_back), dim=1)
        stacked = torch.stack((last_forward, first_back), dim=1)

        # use the forward output of the first LSTM
        enc = _enc[:, :, 0, :]

        # hidden and cell state initialization
        # if num_layer == 1, we do not stack the input to the second LSTM
        if self.cfg.num_layer == 1:
            h0 = first_back.unsqueeze(0)
        else:
            h0 = stacked.transpose(0, 1)
        
        if self.cfg.encoder_cell == 'GRU':
            dec, _ = self.dec_lstm(enc, h0)
        elif self.cfg.encoder_cell == 'RNN':
            dec, _ = self.dec_lstm(enc, h0)
        else:
            c0 = torch.zeros(size=h0.shape).cuda()
            dec, _ = self.dec_lstm(enc, (h0, c0))

        dec = self.dec_dropout(dec)

        tag_out = self.out(dec, batch.tag_mask, batch.tag_ids)

        return tag_out

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
