'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import math
import torch
import torch.nn.functional as F

from collections import defaultdict, OrderedDict, Counter

from .nn import NN, Encoder, UNK, PAD, START, END, SEP
from ..ccg.category import Category, Slashes as sl, InvalidCategoryError
from ..ccg.util import sexpr_nodeblock_to_cat, sexpr_seq_to_cat
from ..util.reader import SCategoryReader, CategoryReader
from ..util.functions import pre_order
from ..util.strings import *


class TransformerSeqDecoder(NN):
    def __init__(self, hidden_dim, context_hidden_dim, labels, max_len=512,
                 transformer_layers=2, attention_heads=3, activation='gelu', dropout=0.2, n_hidden=2,
                 is_sexpr=False, with_sep=True,
                 device=torch.device('cpu')):
        super().__init__()

        self.max_len = max_len
        self.address_dim = 1
        self.with_sep = with_sep

        self.hidden_dim = self.context_hidden_dim = context_hidden_dim

        self.activation = F.gelu
        self.norm = F.log_softmax

        # ensure PAD is last index
        if PAD in labels:
            assert labels[PAD] == len(labels) - 1
            del labels[PAD]

        if self.with_sep and SEP not in labels:
            labels[SEP] = len(labels)

        self.output_dim = len(labels)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.context_hidden_dim, elementwise_affine=False)

        self.intermediate = torch.nn.ModuleList()
        for _ in range(n_hidden):
            self.intermediate.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        cr = CategoryReader()
        self.out_to_ix = {(k if (is_sexpr or k in (PAD, UNK, SEP, START)) else cr.read(k, validate=False).s_expr()): v for k, v in
                          labels.items()}
        self.out_to_ix[PAD] = len(self.out_to_ix)
        self.out_to_ix[START] = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

        self.device = device

    def prepare_outputs(self, seqs, batch_size=None, order='pre', arg_first=False, padding=None, device=torch.device('cpu')):
        '''
        :param argmax: a list of CCG category sequences (lists)
        :return: an output tensor of shape (batch, seq, address)
        '''
        if order == 'pre':
            sort_key = lambda a_l: pre_order(a_l[0])
        else:
            raise NotImplementedError(f'order={order}')
        if arg_first:  # TODO: This might actually be what Kogkalidis+ do; should do the same for consistency's sake
            raise NotImplementedError(f'arg_first')

        padding = 0
        batch_size = batch_size or len(seqs)
        _seqs = []
        for i, seq in enumerate(seqs):
            _seq = [self.out_to_ix[START]]  # TODO: SEP?

            for cat in seq:
                nb = sorted(cat.decompose(self.out_to_ix, binary=True), key=sort_key)
                _nb = []
                for _, l in nb:
                    if l in self.out_to_ix:
                        _nb.append(self.out_to_ix[l])
                    else:
                        _nb.append(self.out_to_ix[UNK])
                _seq.extend(_nb)
                if self.with_sep:
                    _seq.append(self.out_to_ix[SEP])

            _seqs.append(torch.tensor(_seq, dtype=torch.long, device=device))

        while len(_seqs) < batch_size:
            _seqs.append(torch.Tensor().to(_seqs[0]))

        return torch.nn.utils.rnn.pad_sequence(_seqs, batch_first=True, padding_value=self.out_to_ix[PAD])

    def extract_outputs(self, argmax, output_none=False):
        '''
        :param argmax: an output tensor of shape (batch, seq, address)
        :return: a list of CCG category sequences (lists)
        '''
        _y = []
        scr = SCategoryReader()
        for seq in argmax if isinstance(argmax, list) else argmax.view(argmax.size(0), -1).tolist():
            _seq = []
            seq = list(map(self.ix_to_out.get, seq))

            while len(seq) > 0:
                cat, seq = sexpr_seq_to_cat(seq, with_sep=self.with_sep, validate=False)
                if cat is None or cat.root is None:
                    if output_none:
                        cat = Category(None, validate=False)
                        _seq.append(cat)
                else:
                    cat = scr.read(cat.s_expr(), validate=False)
                    _seq.append(cat)

            _y.append(_seq)
        return _y

def test(device: str):
    sl = 25
    nc = 1000

    t = TransformerSeqDecoder(100, 100, {'(NP)': 0, '(S)': 1, '(/)': 2, '(\\)': 3}, max_len=512,
                 # featurized=True, grow_labels=[], mapped_grow_labels=None, max_depth=6, max_mapped_depth=None,
                 transformer_layers=2, attention_heads=3, activation='gelu', dropout=0.2, n_hidden=2,
                 is_sexpr=False, device=device)
    # encoder_input = torch.rand(128, sl, 300).to(device)
    x = torch.rand(128, sl, 100).to(device)
    word_mask = torch.ones(128, sl).to(device)
    y = torch.randint(0, 5, (128, sl * 2)).to(device)
    # y_mask = make_mask((128, sl * 2, sl * 2)).to(device)

    # o = t.extract_outputs(y)
    # i = t.prepare_outputs(o)

    t.train()
    f_v = t.forward(x, y=y, word_mask=word_mask)

    t.eval()
    i_v = t.forward(x[:8], word_mask=word_mask[:8])

    # # p, s = t.vectorized_beam_search(encoder_input[0:20], encoder_output[0:20], encoder_mask[0:20], 0, 3)
    # f_v = t.forward(encoder_input, encoder_output, decoder_input, encoder_mask, decoder_mask)
    # i_v = t.infer(encoder_input[0:20], encoder_output[0:20], encoder_mask[0:20, :50], 0)
