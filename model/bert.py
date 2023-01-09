import sys
import os
import torch

import torch.nn as nn
from transformers import BertModel, BertTokenizer

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
os.environ['NO_PROXY'] = 'huggingface.co'

from utils.example import Example
from utils.args import init_args
from utils.batch import Batch
from utils.initialization import set_random_seed, set_torch_device
from utils.vocab import PAD

class Bert(nn.Module):

    def __init__(self, cfg, device):
        super(Bert, self).__init__()
        self.cfg = cfg
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(cfg.berttype)
        # download the model to local dir
        self.model = BertModel.from_pretrained(f'./model/{cfg.berttype}')
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = TaggingFNNDecoder(768, cfg.num_tags, cfg.tag_pad_idx)
    
    def forward(self, batch: Batch):
        input_ids = self.tokenizer(batch.utt, padding=True, return_tensors="pt")
        input_ids = torch.tensor(input_ids['input_ids']).to(self.device)
        embed = self.model(input_ids)[0]
        embed = self.dropout(embed)

        # remove the <CLS>(101) and <SEP>(102) embedding
        # first remove 102
        # find the 102's index, aka the index of the last token
        indices = torch.where(input_ids == 102)[1]
        _embed = torch.zeros(embed.shape[0], embed.shape[1] - 2, embed.shape[2]).to(self.device)
        for i, idx in enumerate(indices):
            tmp = torch.cat((embed[i, :idx, :], embed[i, idx+1:, :]), dim=0) # remove 102
            tmp = tmp[1:, :] # remove 101
            _embed[i] = tmp
        
        # print(max(batch.lengths), input_ids.shape, _embed.shape, batch.tag_mask.shape, batch.tag_ids.shape)
        # sometimes there are invalid inputs, making the _embed.shape[1](length)
        # is more than 1 compared to the tags(max length)
        # we pad a 0 embedding to the _embed to fill this length
        if _embed.shape[1] < batch.tag_mask.shape[1]:
            tmp = torch.zeros(embed.shape[0], batch.tag_mask.shape[1], embed.shape[2]).to(self.device)
            tmp[:, :_embed.shape[1], :] = _embed
            _embed = tmp

        tag_output = self.fc(_embed, batch.tag_mask, batch.tag_ids)
        return tag_output
    
    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        option_prob_loss = self.forward(batch)
        if option_prob_loss is None:
            pass
        prob, loss = option_prob_loss
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

class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.out1 = nn.Linear(input_size, input_size)
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        hiddens = self.out1(hiddens)
        logits = self.output_layer(hiddens)
        # make masked value extremely small
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

# tests
if __name__ == "__main__":
    args = init_args(sys.argv[1:])
    set_random_seed(args.seed)
    device = set_torch_device(args.device)
    print("Initialization finished ...")
    print("Random seed is set to %d" % (args.seed))
    print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

    train_path = os.path.join(args.dataroot, args.tr_filename)
    dev_path = os.path.join(args.dataroot, 'development.json')

    Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)

    print("loading train dataset")
    train_dataset = Example.load_dataset(train_path)
    print("loading dev dataset")
    dev_dataset = Example.load_dataset(dev_path)
    print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    args.vocab_size = Example.word_vocab.vocab_size
    args.num_tags = Example.label_vocab.num_tags
    args.pad_idx = Example.word_vocab[PAD]
    args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

    model = Bert(args).to(device)
