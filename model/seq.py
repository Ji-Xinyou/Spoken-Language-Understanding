import torch
import torch.nn as nn
import torch.nn.utils.rnn as r

# a part of decoder
class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

class Encoder(nn.Module):
    
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.word_embed = nn.Embedding(cfg.vocab_size, cfg.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(
            cfg.embed_size,
            cfg.hidden_size,
            num_layers=cfg.num_layer,
            bidirectional=True,
            batch_first=True)

    def forward(self, x):
        e = self.word_embed(x)
        o, _ = self.rnn(e)
        o = o[:, :, :self.cfg.hidden_size] + o[:, :, self.cfg.hidden_size:]
        return o
        

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()

