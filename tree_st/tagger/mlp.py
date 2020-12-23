'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import random
import torch
import torch.nn.functional as F
from collections import OrderedDict

from .nn import NN, Encoder, UNK, PAD, START, END, SEP
from .oracle.oracle import find_valid_fillers
from ..ccg.category import Category
from ..ccg.util import sexpr_nodeblock_to_cat, sexpr_seq_to_cat
from ..util.reader import SCategoryReader, CategoryReader
from ..util.functions import pre_order
from ..util.address_map import AtomicAddressMap, DummyAddressMap


class MLPDecoder(NN):
    def __init__(self, hidden_dim, context_hidden_dim, labels,
                 atomic_labels=None, mapped_grow_labels=None, max_mapped_depth=6,
                 enc_attention=False, dec_attention=False, dropout=0.2, n_hidden=2, is_sexpr=False,
                 device=torch.device('cpu')):
        '''
        A simple MLP tagger.
        '''
        super().__init__()

        self.max_depth = 0
        self.max_width = 2**self.max_depth
        self.address_dim = 2**(self.max_depth + 1) - 1

        self.max_mapped_depth = self.max_depth if max_mapped_depth is None else max_mapped_depth
        self.max_mapped_width = 2 ** self.max_mapped_depth
        self.mapped_address_dim = 2 ** (self.max_mapped_depth + 1) - 1

        self.hidden_dim = self.context_hidden_dim = context_hidden_dim

        self.featurized = False
        self.activation = F.gelu  # torch.nn.ReLU()  # TODO: try cube activation
        self.norm = F.log_softmax
        self.enc_attention = enc_attention
        self.dec_attention = False
        self.attention = enc_attention or dec_attention
        self.oracle = oracle

        # ensure PAD is last index
        if PAD in labels:
            assert labels[PAD] == len(labels) - 1
            del labels[PAD]

        self.output_dim = len(labels)

        self.intermediate = torch.nn.ModuleList()
        for _ in range(n_hidden):
            self.intermediate.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.context_hidden_dim, elementwise_affine=False)

        cr = CategoryReader()
        self.out_to_ix = {(k if (is_sexpr or k in (PAD, UNK)) else cr.read(k, validate=False).s_expr()): v for k, v in labels.items()}
        self.out_to_ix[PAD] = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

        self.address_map = AtomicAddressMap(self.address_dim, self.mapped_address_dim, self.out_to_ix,
                                            tgt_out_to_ix=atomic_labels)

        self.grow_labels = []
        self.mapped_grow_labels = torch.tensor(grow_labels if mapped_grow_labels is None else mapped_grow_labels,
                                               dtype=torch.long, device=device)

        self.device = device

    def prepare_outputs(self, seqs, batch_size=None, padding=None, device=torch.device('cpu')):
        '''
        :param argmax: a list of CCG category sequences (lists)
        :return: an output tensor of shape (batch, seq, address)
        '''
        _seqs = []
        padding = padding or max(len(s) for s in seqs)
        batch_size = batch_size or len(seqs)
        _seqs = torch.tensor([[[self.out_to_ix[PAD]]]], dtype=torch.long, device=device).repeat(batch_size, padding, self.address_dim)
        for i, seq in enumerate(seqs):
            _seq = []
            for cat in seq:
                nb = dict(cat.decompose(self.out_to_ix, binary=False))
                _nb = []
                for a in range(1, self.address_dim + 1):
                    try:
                        out = nb.get(a, PAD)
                        if out in self.out_to_ix:
                            _nb.append(self.out_to_ix[out])
                        else:
                            _nb.append(self.out_to_ix[UNK])
                    except KeyError:
                        print(f'a={a}, nb={nb}, self.out_to_ix={self.out_to_ix}', file=sys.stderr)
                        raise
                _seq.append(_nb)

            _seqs[i, :len(_seq)] = _seq = torch.tensor(_seq, dtype=torch.long, device=device)

        return _seqs

    def prepare_deps(self, seqs, padding=None, device=torch.device('cpu')):
        '''
        :param argmax: a list of CCG category sequences (lists)
        :return: an output tensor of shape (batch, seq, address)
        '''
        _seqs = []
        padding = max(len(s) for s in seqs) if padding is None else padding
        for deps in seqs:
            _deps = []
            for i in range(padding):
                deps_i = deps[i]
                _deps_i = []
                for a in range(1, self.address_dim+1):
                    if a in deps_i:
                        # When generating a_k at address 10^{n-k}1 (2^{n-k+1} + 1) of head,
                        # pay attn to (address 10^{m-nargs(a_k)} (2^{m-nargs(a_k)}) of) dep(head, k).
                        dep = deps_i[a]
                        _deps_i.append((a-1, dep.dep, dep.get_head_addr()-1))
                _deps.append(_deps_i)

            _seqs.append(_deps)
        return _seqs

    def extract_outputs(self, argmax, use_address_map=False, output_none=False):
        '''
        :param argmax: an output tensor of shape (batch, seq, address)
        :return: a list of CCG category sequences (lists)
        '''
        _y = []
        scr = SCategoryReader()
        for seq in argmax.tolist():
            _seq = []
            for sparse_nb in seq:
                dense_nb = OrderedDict()
                last_depth = -1
                no_grow_at_last_depth = True
                for a, argmax_ix in enumerate(sparse_nb, start=1):
                    address = f'{a:b}'
                    depth = len(address) - 1
                    if depth != last_depth:
                        cat = sexpr_nodeblock_to_cat(dense_nb.items(), binary=False, validate=False)
                        if cat is None or cat.validate():
                            pass
                        else:
                            break
                        last_depth = depth
                        no_grow_at_last_depth = True

                    label = self.address_map.ix_to_out[argmax_ix] if use_address_map else self.ix_to_out[argmax_ix]
                    if label == PAD:
                        continue
                    dense_nb[a] = label
                cat = sexpr_nodeblock_to_cat(dense_nb.items(), binary=False, validate=False)
                if cat is None:
                    if output_none:
                        cat = Category(None, validate=False)
                        _seq.append(cat)
                else:
                    cat = scr.read(cat.s_expr(), validate=False)
                    _seq.append(cat)
            _y.append(_seq)
        return _y

    def y_grow_condition(self, y):
        '''
        mask that only includes batch items and addresses that have a grow_label
        '''
        return torch.any(y.index_select(2, self.mapped_grow_labels).bool(), 2)

    def forward(self, x, y=None, word_mask=None, **kwargs):
        batch_size, seq_len, hidden = x.size()
        assert hidden == self.context_hidden_dim, (f'hidden: {hidden}, context_hidden_dim: {self.context_hidden_dim}')
        x = x.view(-1, self.context_hidden_dim).to(self.device)
        total_batch_size = x.size(0)
        categories_gold = None

        states = {}
        if y is not None:
            if self.oracle == 'global':
                y = y.to(self.device)
            else:
                categories_gold = self.extract_outputs(y.view(batch_size, seq_len, -1))
                y = torch.ones(batch_size, seq_len, self.address_dim, dtype=torch.long, device=self.device,
                                         requires_grad=False) * self.out_to_ix[PAD]
                states['y'] = y

        h = torch.zeros(total_batch_size, self.address_dim, self.hidden_dim, device=self.device)
        h[:, 0, :] = x
        states['h'] = h

        y_hat = torch.zeros(total_batch_size, self.address_dim, self.output_dim, device=self.device)
        states['y_hat'] = y_hat
        # TODO: make a y_hat_softmax (dim -1 is self.output_dim) and a separate y_hat_argmax (dim -1 is 1?)
        # TODO: y_hat_softmax will have the prediction probabilities for whatever granularity is used and skip descendent addresses of larger chunks
        # TODO: y_had_argmax will have decomposed gold/predicted categories' argmaxes at each address and will be used for masks

        mask = torch.zeros(total_batch_size, self.address_dim, dtype=torch.bool, device=self.device, requires_grad=False)
        mask[:, 0] = 1 if word_mask is None else word_mask.view(-1)
        states['mask'] = mask

        atom_mask = torch.zeros(total_batch_size, self.mapped_address_dim, dtype=torch.bool, device=self.device,
                                requires_grad=False)
        atom_mask[:, 0] = 1 if word_mask is None else word_mask.view(-1)
        states['atom_mask'] = atom_mask

        x = x.unsqueeze(1)

        total_batch_size = x.size(0)

        mask = states['mask']
        y_hat = states['y_hat']

        # predict node label

        h = states['h']
        current_h = h[:, 0, :]

        if y is not None and self.oracle != 'global':
            y = states['y']

        if self.attention:
            attn_query = current_h.view(batch_size, seq_len, self.hidden_dim)

            if self.enc_attention:
                enc_state = x.view(batch_size, seq_len, self.context_hidden_dim)
                enc_attention_weights = torch.matmul(attn_query, enc_state.transpose(1, 2))
                # states['enc_attn'][:, 0, :] = enc_attention_weights.view(total_batch_size, 1, seq_len)
                enc_attention_weights = F.softmax(enc_attention_weights, dim=2)
                enc_attention_values = torch.matmul(enc_attention_weights, enc_state).view(total_batch_size, 1,
                                                                                           self.context_hidden_dim)
                current_h = (current_h + enc_attention_values)

            current_h = self.layer_norm(current_h)

        for intermediate in self.intermediate:
            current_h = self.dropout(self.activation(intermediate(current_h.clone())))

        current_o = self.O(current_h)

        y_hat[:, 0, :] = current_y
        states['y_hat'] = y_hat

        return states

