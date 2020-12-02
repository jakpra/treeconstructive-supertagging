'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import os
import itertools

import numpy as np
import torch
import transformers as pt

from ..nn import Encoder, PAD, START, END
from ..rnn import RNNEncoder


class SpanEncoder(Encoder):
    @property
    def pad_token_id(self):
        return self.to_ix['word'][PAD]

    @property
    def bos_token_id(self):
        return self.to_ix['word'][START]

    @property
    def eos_token_id(self):
        return self.to_ix['word'][END]


class RNNSpanEncoder(SpanEncoder, RNNEncoder):
    pass


class LookupSpanEncoder(SpanEncoder):
    def __init__(self, embedding_dim, weights, token_maps, seq_to_ix):
        # TODO: make this a nn.Module?
        self.weights = weights
        self.token_maps = token_maps
        self.seq_to_ix = seq_to_ix
        self.embedding_dim = embedding_dim

    # TODO: make this just output a vector of sentence indices
    def prepare_inputs(self, seqs, padding=None, device=torch.device('cpu'), **kwargs):
        del kwargs
        _seqs = []
        padding = max(len(s) for s in seqs) if padding is None else padding
        for seq in seqs:
            weights = self.weights[self.seq_to_ix[' '.join(seq)]]
            while len(weights) < padding:
                weights.append(np.zeros(self.embedding_dim).astype(np.float))
            _seqs.append(weights)
        return torch.tensor(_seqs, dtype=torch.float, device=device)

    # TODO: this looks up the sentence indices in the weights matrix
    def forward(self, seqs, **kwargs):
        return seqs.transpose(0, 1)  # (seq_batch_size, seq_length, emb_dim).transp() -> (seq_length, cat_batch_size, emb_dim)


class BERTLikeSpanEncoder(SpanEncoder):
    def __init__(self, *args, pretrained=None, **kwargs):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.pretrained = pretrained

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def prepare_input(self, seq, train=False, lower=True, device=torch.device('cpu'), **kwargs):
        tokens = [self.tokenizer.encode(w.lower() if lower else w) for w in seq]
        enc_shapes = [len(enc) for enc in tokens]
        return torch.tensor([list(itertools.chain.from_iterable(tokens))]), enc_shapes

    def prepare_inputs(self, seqs, padding=None, train=False, lower=True, device=torch.device('cpu'), **kwargs):
        _seqs = []  # {k: [] for k in self.to_ix}
        _words = []
        _tok_idxs = []
        tok_padding = 2
        padding = max(len(s) for s in seqs) if padding is None else padding
        for seq in seqs:
            idxs = [self.tokenizer.encode(w.lower() if lower else w, add_special_tokens=False) for w in seq]
            enc_shapes = [1] + [len(enc) for enc in idxs] + [1]

            _seq = list(itertools.chain.from_iterable(idxs))
            _seq = self.tokenizer.prepare_for_model(_seq,
                                                    add_special_tokens=True,
                                                    return_token_type_ids=False,
                                                    return_attention_mask=False)['input_ids']
            _seqs.append(_seq)

            _seq_len = len(_seq)
            if _seq_len > tok_padding:
                tok_padding = _seq_len

            tok_idxs = torch.tensor(range(_seq_len)).split_with_sizes(enc_shapes)[1:-1]
            _tok_idxs.append(tok_idxs)

            seq_len = len(seq)
            words = [self.tokenizer.pad_token_id - 1] * seq_len + [self.tokenizer.pad_token_id] * (padding - seq_len)
            _words.append(words)

        for seq in _seqs:
            seq.extend([self.tokenizer.pad_token_id] * (tok_padding - len(seq)))
        _seqs = torch.tensor(_seqs, dtype=torch.long, device=device)
        _words = torch.tensor(_words, dtype=torch.long, device=device)
        return {'word': _words, 'enc': _seqs,  # 'word' is just a hack for compatibility with other SpanEncoders
                'idxs': _tok_idxs, 'padding': padding}

    def forward(self, seqs, word_mask=None, device=torch.device('cpu'), **kwargs):
        enc, idxs, padding = seqs['enc'], seqs['idxs'], seqs['padding']
        batch_size, tok_seq_len = enc.size()
        tokenized_out = self.model(enc.to(device))[0]  #  TODO: , attention_mask=None if word_mask is None else word_mask.float()
        cls = tokenized_out[:, 0]

        out = torch.zeros(batch_size, padding, tokenized_out.size(-1), dtype=torch.float32, device=device, requires_grad=False)
        for i in range(batch_size):
            _idxs = idxs[i]
            for j in range(len(_idxs)):
                __idxs = _idxs[j]
                word = tokenized_out[i, __idxs]
                word = torch.sum(word, 0) / __idxs.size(0)
                out[i, j] = word

        del tokenized_out

        return out, cls

    def to_dict(self):
        result = super().to_dict()
        result.update({'pretrained': self.pretrained})
        return result


class BERTSpanEncoder(BERTLikeSpanEncoder):
    def __init__(self, emb=False, pretrained='bert-large-uncased', finetune=True):
        super().__init__()
        if emb:
            self.model = pt.BertModel(pt.BertConfig.from_pretrained(pretrained))
        else:
            self.model = pt.BertModel.from_pretrained(pretrained)
        self.tokenizer = pt.BertTokenizer.from_pretrained(pretrained)

        self.hidden_dim = self.model.encoder.layer[-1].output.dense.out_features


class RoBERTaSpanEncoder(BERTLikeSpanEncoder):
    def __init__(self, emb=False, pretrained='roberta-base', finetune=True):
        super().__init__()
        if emb:
            self.model = pt.RobertaModel(pt.RobertaConfig.from_pretrained(pretrained))
        else:
            self.model = pt.RobertaModel.from_pretrained(pretrained)
        self.tokenizer = pt.RobertaTokenizer.from_pretrained(pretrained)

        self.hidden_dim = self.model.encoder.layer[-1].output.dense.out_features
