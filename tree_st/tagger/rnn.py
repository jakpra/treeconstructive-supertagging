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


class RNNEncoder(Encoder):
    def __init__(self, embedding_dim, feat_embedding_dim=0, vocab_sizes={}, hidden_dims=[300], vocab={},
                 rnn=torch.nn.GRU, bidirectional=True, dropout=[0.0],
                 embed=None, feat_chars=0, feat_embed=None, emb_dropout=0.0):
        super().__init__(embedding_dim, hidden_dims=hidden_dims)
        self.embedding_dim = embedding_dim
        self.feat_embedding_dim = feat_embedding_dim
        self.feat_chars = feat_chars
        self.to_ix = dict(vocab)
        for key in vocab:
            if UNK not in vocab[key]:
                self.to_ix[key][UNK] = len(self.to_ix[key])
            if PAD not in vocab[key]:
                self.to_ix[key][PAD] = len(self.to_ix[key])
        if START not in self.to_ix['word']:
            self.to_ix['word'][START] = len(self.to_ix['word'])
        if END not in self.to_ix['word'].keys():
            self.to_ix['word'][END] = len(self.to_ix['word'])

        if vocab_sizes:
            self.vocab_sizes = vocab_sizes
        else:
            self.vocab_sizes = {k: len(v) for k, v in vocab.items()}

        self.bidirectional = bidirectional

        if embed is None:
            self.embeddings = torch.nn.Embedding(self.vocab_sizes['word'], embedding_dim)
        else:
            self.embeddings = embed

        if feat_embed is None:
            self.feat_embeddings = torch.nn.ModuleDict()
            for _n in range(1, feat_chars+1):
                pre_key = f'pre_{_n}'
                suf_key = f'suf_{_n}'
                self.feat_embeddings[pre_key] = torch.nn.Embedding(self.vocab_sizes[pre_key], feat_embedding_dim)
                self.feat_embeddings[suf_key] = torch.nn.Embedding(self.vocab_sizes[suf_key], feat_embedding_dim)
        else:
            self.feat_embeddings = feat_embed

        self.emb_dropout = torch.nn.Dropout(emb_dropout)

        # The LSTM takes embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = rnn(embedding_dim + 2 * feat_chars * feat_embedding_dim, hidden_dims[0],
                       num_layers=len(hidden_dims), bidirectional=bidirectional, dropout=dropout[0], batch_first=True)

        self.dropout = torch.nn.ModuleList([torch.nn.Dropout(d) for d in dropout])

    # for backward compatibility
    def prepare_input(self, seq, *args, train=False, lower=True, **kwargs):
        idxs = []
        for w in seq:
            if lower:
                w = w.lower()
            if train and w not in self.word_to_ix and len(self.word_to_ix) <= self.vocab_size:
                self.word_to_ix[w] = len(self.word_to_ix)
            idxs.append(self.word_to_ix.get(w, self.word_to_ix[UNK]))
        return torch.tensor([idxs], dtype=torch.long)

    def forward(self, seqs, *args, device=torch.device('cpu'), **kwargs):
        # TODO: add CVT

        word_mask = ((seqs['idx'] != self.bos_token_id) & (seqs['idx'] != self.eos_token_id)).to(device)

        embeds = [self.embeddings(seqs['idx'].to(device))]
        for _n in range(1, self.feat_chars + 1):
            pre_key = f'pre_{_n}'
            suf_key = f'suf_{_n}'
            try:
                embeds.append(self.feat_embeddings[pre_key](seqs[pre_key].to(device)))
            except RuntimeError:
                print(self.feat_embeddings[pre_key], seqs[pre_key], file=sys.stderr)
                raise
            embeds.append(self.feat_embeddings[suf_key](seqs[suf_key].to(device)))
        embeds = torch.cat(embeds, dim=-1)
        embeds = self.emb_dropout(embeds)

        rnn_out, cls = self.rnn(embeds)
        rnn_out = self.dropout[-1](rnn_out)

        batch_size, seq_len, hidden = rnn_out.size()
        rnn_out = rnn_out.view(batch_size, seq_len, 1 + self.bidirectional, hidden // (1 + self.bidirectional))
        rnn_out = torch.sum(rnn_out, dim=2) / (1+self.bidirectional)

        rnn_out = rnn_out[word_mask, :].view(batch_size, -1, self.hidden_dim)

        cls = cls.view(batch_size, -1, 1 + self.bidirectional, hidden // (1 + self.bidirectional))[:, -1]
        cls = torch.sum(cls, dim=1) / (1 + self.bidirectional)

        del embeds, word_mask

        return rnn_out, cls

    # for backward compatibility
    def get_span(self, seq, i, j, train=False, device=torch.device('cpu')):
        if hash(' '.join(seq)) == self.cache_ix:
            seq_enc = self.cache_enc
        else:
            x = self.prepare_input(seq, train=train, device=device)
            seq_enc = self(x)
            self.cache_enc = seq_enc
            self.cache_ix = hash(' '.join(seq))
        return seq_enc[i:j, :]

    def to_dict(self):
        result = super().to_dict()
        result.update({'embedding_dim': self.embedding_dim,
                       'feat_embedding_dim': self.feat_embedding_dim,
                       'hidden_dim': self.hidden_dim,
                       'vocab_sizes': self.vocab_sizes,
                       'to_ix': self.to_ix,
                       'bidirectional': self.bidirectional})
        return result


class SeqRNNDecoder(NN):
    def __init__(self, hidden_dim, context_hidden_dim, labels, atomic_labels=None, reuse_embedding=True, max_len=512,
                 grow_labels=[], mapped_grow_labels=None,  mapped_max_len=None, max_depth=6,
                 with_sep=True,
                 enc_attention=False, dec_attention=False, dropout=0.2, n_hidden=2, is_sexpr=False,
                 device=torch.device('cpu')):

        super().__init__()

        self.max_len = max_len
        self.address_dim = 2**(max_depth + 1) - 1

        self.mapped_max_len = max_len if mapped_max_len is None else mapped_max_len
        self.mapped_address_dim = self.address_dim

        self.hidden_dim = self.context_hidden_dim = context_hidden_dim

        self.activation = F.gelu  # torch.nn.ReLU()  # TODO: try cube activation
        self.norm = F.log_softmax
        self.enc_attention = enc_attention
        self.dec_attention = dec_attention
        self.reuse_embedding = reuse_embedding
        self.attention = enc_attention or dec_attention

        # ensure PAD is last index
        if PAD in labels:
            assert labels[PAD] == len(labels) - 1
            del labels[PAD]

        self.with_sep = with_sep
        if self.with_sep and SEP not in labels:
            labels[SEP] = len(labels)

        self.output_dim = len(labels)

        if self.max_len > 1:

            if self.hidden_dim != self.context_hidden_dim:
                self.hidden_dim = self.context_hidden_dim

            self.rnn_cell = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.intermediate = torch.nn.ModuleList()
        for _ in range(n_hidden):
            self.intermediate.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        if (not self.transf or self.dec_attention) and self.max_len > 0:
            self.embedding_matrix = torch.nn.Parameter(
                torch.rand(self.output_dim + 1, self.hidden_dim, device=device) * 0.02)
            self.output_embedder = lambda x: F.embedding(x, self.embedding_matrix, padding_idx=self.output_dim,
                                                         scale_grad_by_freq=True)
            if self.reuse_embedding:
                self.O = lambda x: x @ (self.embedding_matrix.transpose(1, 0) + 1e-10)
            else:
                self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        else:
            self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.context_hidden_dim, elementwise_affine=False)

        cr = CategoryReader()
        self.out_to_ix = {(k if (is_sexpr or k in (PAD, UNK, SEP)) else cr.read(k, validate=False).s_expr()): v for k, v in
                          labels.items()}
        self.out_to_ix[PAD] = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}
        self.sep = torch.tensor([self.out_to_ix[SEP]], dtype=torch.long, requires_grad=False, device=device)

        self.address_map = DummyAddressMap(self.address_dim, self.address_dim, self.out_to_ix,
                                           tgt_out_to_ix=atomic_labels)

        self.grow_labels = grow_labels

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

        padding = padding or max(len(s) for s in seqs)
        batch_size = batch_size or len(seqs)
        _seqs = torch.ones(batch_size, padding, self.address_dim, dtype=torch.long, device=device) * self.out_to_ix[PAD]

        for i, seq in enumerate(seqs):
            for j, cat in enumerate(seq):
                nb = sorted(cat.decompose(self.out_to_ix, binary=True), key=sort_key)
                _nb = []
                for _, l in nb:
                    if l in self.out_to_ix:
                        _nb.append(self.out_to_ix[l])
                    else:
                        _nb.append(self.out_to_ix[UNK])

                _nb.append(self.out_to_ix[SEP])

                _seqs[i, j, :len(_nb)] = torch.tensor(_nb, dtype=torch.long, device=device)

        return _seqs

    def extract_outputs(self, argmax):
        '''
        :param argmax: an output tensor of shape (batch, seq, address)
        :return: a list of CCG category sequences (lists)
        '''
        _y = []

        for seq in argmax.view(argmax.size(0), -1, self.address_dim).tolist():
            _seq = []
            for nb in seq:
                nb = list(map(self.ix_to_out.get, nb))

                if len(seq) > 0:
                    cat, rest = sexpr_seq_to_cat(nb, with_sep=self.with_sep, validate=False)
                    if cat is None or cat.root == PAD:
                        break
                    _seq.append(cat)

                if rest:
                    pass

            _y.append(_seq)
        return _y

    def y_grow_condition(self, ys, seq_lens):
        one_hot = F.one_hot(ys, num_classes=self.output_dim+1)
        return torch.sum(one_hot.index_select(2, self.sep)) < seq_lens

    def forward(self, x, y=None, word_mask=None, enc_cls=None, **kwargs):
        batch_size, seq_len, hidden = x.size()
        assert hidden == self.context_hidden_dim, f'hidden: {hidden}, context_hidden_dim: {self.context_hidden_dim}'
        x = x.to(self.device)
        if y is None:
            y_unaligned = None
        else:
            y_unaligned = torch.ones(batch_size, self.max_len, dtype=torch.long, device=self.device) * self.out_to_ix[PAD]
            for i, s in enumerate(y):
                for j, c in enumerate(s):
                    k = 0
                    for a in c:
                        if a == self.out_to_ix[PAD]:
                            break
                        y_unaligned[i, j] = a
                        k += 1

        if word_mask is not None:
            word_mask = word_mask.to(self.device)

        seq_lens = torch.sum(word_mask.long(), dim=1)

        states = {}
        h = torch.zeros(batch_size, self.max_len+1, self.hidden_dim, device=self.device)

        if self.transf:
            ...
        else:
            h[:, 0] = enc_cls

        states['h'] = h

        y_hat = torch.zeros(batch_size, self.max_len, self.output_dim, device=self.device)
        states['y_hat'] = y_hat

        mask = torch.zeros(batch_size, self.max_len, dtype=torch.bool, device=self.device,
                           requires_grad=False)
        mask[:, 0] = 1
        states['mask'] = mask

        atom_mask = torch.zeros(batch_size, self.mapped_max_len, dtype=torch.bool, device=self.device,
                                requires_grad=False)
        atom_mask[:, 0] = 1
        states['atom_mask'] = atom_mask

        for t in range(self.max_len):
            try:
                self._rnn(t, states, x, batch_size, seq_len, seq_lens, y=y_unaligned)
            except StopIteration:
                break

        # TODO: consider doing the alignment here so we can reuse the same code in nn.compute_acc_and_loss

        y_hat = states['y_hat']
        argmaxes = torch.argmax(y_hat, dim=2)
        address_mask = states['atom_mask'].view(batch_size, -1)
        argmaxes[~address_mask] = self.out_to_ix[PAD]

        y_hat_aligned = torch.zeros(batch_size, seq_len, self.address_dim, self.output_dim, device=self.device)
        mask_aligned = torch.zeros(batch_size, seq_len, self.address_dim, dtype=torch.bool, device=self.device)

        for i, s in enumerate(argmaxes):
            _j = 0
            k = 0
            for j, c in enumerate(s):
                if c == self.out_to_ix[PAD] or _j == seq_len:
                    break
                mask_aligned[i, _j] = True
                y_hat_aligned[i, _j, k] = y_hat[i, j]
                k += 1
                if c == self.out_to_ix[SEP] or k == self.address_dim:  # TODO: the latter is messy, ideally we'd deal with this differently
                    _j += 1
                    k = 0
                    continue

        states['y_hat'] = y_hat_aligned
        states['mask'] = mask_aligned

        return states

    def _rnn(self, t, states, x, batch_size, seq_len, seq_lens, y=None):
        '''

        :param parent_h: the hidden output representation of the parent node
        :param parent_c: the hidden context representation of the parent node
        :param y: oracle parent label if available
        :return: hidden and context representations of the left and right children
        '''

        mask = states['mask']
        y_hat = states['y_hat']

        h = states['h']
        current_h = h[:, t, :].clone()

        with torch.no_grad():
            cumulative_argmaxes = (torch.argmax(y_hat, dim=2) if y is None else y)[:, :t]
            cumulative_argmaxes[~mask[:, :t]] = self.out_to_ix[PAD]

        if self.attention:
            attn_query = current_h.view(batch_size, 1, self.hidden_dim)

            if self.enc_attention:
                enc_state = x.view(batch_size, seq_len, self.context_hidden_dim)
                enc_attention_weights = torch.matmul(attn_query, enc_state.transpose(1, 2))
                enc_attention_weights = F.softmax(enc_attention_weights, dim=2)
                enc_attention_values = torch.matmul(enc_attention_weights, enc_state).view(batch_size,
                                                                                           self.context_hidden_dim)
                current_h = (current_h + enc_attention_values)

            if self.dec_attention and t >= 1:
                dec_state = self.output_embedder(cumulative_argmaxes.clone())

                dec_state = dec_state.view(batch_size, t, self.context_hidden_dim)
                dec_attention_weights = torch.matmul(attn_query, dec_state.transpose(1, 2))
                dec_attention_weights = F.softmax(dec_attention_weights, dim=2)
                dec_attention_values = torch.matmul(dec_attention_weights, dec_state).view(batch_size,
                                                                                           self.context_hidden_dim)
                current_h = (current_h + dec_attention_values)  # / 2

            current_h = self.layer_norm(current_h)

        for intermediate in self.intermediate:
            current_h = self.dropout(self.activation(intermediate(current_h.clone())))

        current_o = self.O(current_h)

        current_y = current_o[:, :-1]

        y_hat[:, t, :] = current_y
        states['y_hat'] = y_hat

        if t == self.max_len - 1 or (y is not None and t == y.size(1) - 1):
            raise StopIteration('max len reached')

        # grow seq

        current_y_argmaxes = torch.argmax(y_hat[:, t], dim=1) if y is None else y[:, t]
        cumulative_argmaxes = torch.cat([cumulative_argmaxes, current_y_argmaxes.view(-1, 1)], dim=1)

        mapped_cumulative_y = cumulative_argmaxes
        mapped_current_y = mapped_cumulative_y[:, t]

        # TODO eventually
        shortcut_children = torch.zeros(mapped_current_y.size(), dtype=torch.bool, device=self.device)

        atom_mask = states['atom_mask']

        grow_mask = self.y_grow_condition(mapped_cumulative_y[:, :t], seq_lens)  # TODO: find workaround for interpolated tagset

        current_y_argmaxes = current_y_argmaxes[grow_mask]
        grow_bool = torch.any(grow_mask)
        if not grow_bool:
            raise StopIteration('all seqs have failed the grow condition')

        skip_grow_mask = grow_mask & ~shortcut_children  # TODO

        atom_mask[:, t + 1] = grow_mask
        states['atom_mask'] = atom_mask

        mask[:, t + 1] = skip_grow_mask
        states['mask'] = mask

        current_h = current_h[grow_mask, :]
        current_y = self.output_embedder(current_y_argmaxes)
        next_h = self.rnn_cell(current_y, current_h)

        h[grow_mask, t + 1] = next_h

        states['h'] = h


class PointwiseSeqDecoder(NN):
    def __init__(self, hidden_dim, context_hidden_dim, labels, atomic_labels=None, reuse_embedding=True,
                 transformer=False, grow_labels=[], mapped_grow_labels=None, max_size=6, mapped_max_size=None,
                 enc_attention=False, dec_attention=False, dropout=0.2, n_hidden=2, is_sexpr=False,
                 device=torch.device('cpu')):
        '''
        Generating a separate sequence for each word in the input.
        '''
        super().__init__()

        self.max_size = max_size
        self.address_dim = max_size

        self.mapped_max_size = max_size if mapped_max_size is None else mapped_max_size

        self.hidden_dim = self.context_hidden_dim = context_hidden_dim

        self.transf = transformer
        self.activation = F.gelu  # torch.nn.ReLU()  # TODO: try cube activation
        self.norm = F.log_softmax
        self.enc_attention = enc_attention
        self.dec_attention = dec_attention
        self.reuse_embedding = reuse_embedding
        self.attention = enc_attention or dec_attention

        # ensure PAD is last index
        if PAD in labels:
            assert labels[PAD] == len(labels) - 1
            del labels[PAD]

        self.with_sep = True
        labels[SEP] = len(labels)

        self.output_dim = len(labels)

        if self.max_size > 1:

            if self.hidden_dim != self.context_hidden_dim:
                self.hidden_dim = self.context_hidden_dim

            if self.transf:
                self.transformer = ...
            else:
                self.rnn_cell = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.intermediate = torch.nn.ModuleList()
        for _ in range(n_hidden):
            self.intermediate.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        if (not self.transf or self.dec_attention) and self.max_size > 0:
            self.embedding_matrix = torch.nn.Parameter(torch.rand(self.output_dim+1, self.hidden_dim, device=device) * 0.02)
            self.output_embedder = lambda x: F.embedding(x, self.embedding_matrix, padding_idx=self.output_dim,
                                                         scale_grad_by_freq=True)
            if self.reuse_embedding:
                self.O = lambda x: x @ (self.embedding_matrix.transpose(1, 0) + 1e-10)
            else:
                self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        else:
            self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.context_hidden_dim, elementwise_affine=False)

        cr = CategoryReader()
        self.out_to_ix = {(k if (is_sexpr or k in (PAD, UNK, SEP)) else cr.read(k, validate=False).s_expr()): v for k, v in labels.items()}
        self.out_to_ix[PAD] = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

        self.address_map = DummyAddressMap(self.address_dim, self.address_dim, self.out_to_ix,
                                           tgt_out_to_ix=atomic_labels)

        self.grow_labels = grow_labels
        self.mapped_grow_labels = torch.tensor(grow_labels if mapped_grow_labels is None else mapped_grow_labels,
                                               dtype=torch.long, device=device)

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

        padding = padding or max(len(s) for s in seqs)
        batch_size = batch_size or len(seqs)
        _seqs = torch.ones(batch_size, padding, self.max_size, dtype=torch.long, device=device) * self.out_to_ix[PAD]

        for i, seq in enumerate(seqs):
            for j, cat in enumerate(seq):
                nb = sorted(cat.decompose(self.out_to_ix, binary=True), key=sort_key)
                _nb = []
                for _, l in nb:
                    if l in self.out_to_ix:
                        _nb.append(self.out_to_ix[l])
                    else:
                        _nb.append(self.out_to_ix[UNK])

                _nb.append(self.out_to_ix[SEP])

                _seqs[i, j, :len(_nb)] = torch.tensor(_nb, dtype=torch.long, device=device)

        return _seqs

    def extract_outputs(self, argmax):
        '''
        :param argmax: an output tensor of shape (batch, seq, address)
        :return: a list of CCG category sequences (lists)
        '''
        _y = []

        for seq in argmax.view(argmax.size(0), -1, self.max_size).tolist():
            _seq = []
            for nb in seq:

                nb = list(map(self.ix_to_out.get, nb))

                if len(seq) > 0:
                    cat, rest = sexpr_seq_to_cat(nb, with_sep=self.with_sep, validate=False)
                    if cat is None or cat.root == PAD:
                        break
                    _seq.append(cat)

                if rest:
                    pass

            _y.append(_seq)
        return _y

    # dummy
    # def y_grow_condition(self, y):
    #     return torch.ones(y.size())

    def y_grow_condition(self, y):
        return ~torch.eq(y, self.out_to_ix[SEP])

    # TODO: y_grow_condition that (roughly) ensures valid output tree

    def forward(self, x, y=None, word_mask=None, **kwargs):
        batch_size, seq_len, hidden = x.size()
        assert hidden == self.context_hidden_dim, (f'hidden: {hidden}, context_hidden_dim: {self.context_hidden_dim}')
        x = x.view(-1, self.context_hidden_dim).to(self.device)
        total_batch_size = x.size(0)
        if y is not None:
            y = y.view(total_batch_size, -1).to(self.device)
            assert y.size() == (total_batch_size, self.max_size), y.size()

        states = {}
        h = torch.zeros(total_batch_size, self.max_size, self.hidden_dim, device=self.device)

        if self.transf:
            ...
        else:
            h[:, 0, :] = x

        states['h'] = h

        y_hat = torch.zeros(total_batch_size, self.max_size, self.output_dim, device=self.device)
        states['y_hat'] = y_hat

        mask = torch.zeros(total_batch_size, self.max_size, dtype=torch.bool, device=self.device, requires_grad=False)
        mask[:, 0] = 1 if word_mask is None else word_mask.view(-1)
        states['mask'] = mask

        atom_mask = torch.zeros(total_batch_size, self.mapped_max_size, dtype=torch.bool, device=self.device,
                                requires_grad=False)
        atom_mask[:, 0] = 1 if word_mask is None else word_mask.view(-1)
        states['atom_mask'] = atom_mask

        x = x.unsqueeze(1)
        for t in range(self.max_size):
            try:
                self._rnn(t, states, x, batch_size, seq_len, y=y)
            except StopIteration:
                break

        return states

    def _rnn(self, t, states, x, batch_size, seq_len, y=None):

        total_batch_size = x.size(0)

        mask = states['mask']
        y_hat = states['y_hat']

        # predict node label

        h = states['h']
        current_h = h[:, t, :].clone()

        if self.transf:
            ...

        with torch.no_grad():
            cumulative_argmaxes = (torch.argmax(y_hat, dim=2) if y is None else y)[:, :t]
            cumulative_argmaxes[~mask[:, :t]] = self.out_to_ix[PAD]

        if self.attention:
            attn_query = current_h.view(batch_size, seq_len, self.hidden_dim)

            if self.enc_attention:
                enc_state = x.view(batch_size, seq_len, self.context_hidden_dim)
                enc_attention_weights = torch.matmul(attn_query, enc_state.transpose(1, 2))
                # states['enc_attn'][:, address_mask, :] = enc_attention_weights.view(total_batch_size, n_addresses, seq_len)
                enc_attention_weights = F.softmax(enc_attention_weights, dim=2)
                enc_attention_values = torch.matmul(enc_attention_weights, enc_state).view(total_batch_size, self.context_hidden_dim)
                current_h = (current_h + enc_attention_values)

            if self.dec_attention and t >= 1:
                dec_state = self.output_embedder(cumulative_argmaxes.clone())

                dec_state = dec_state.view(batch_size, seq_len * t, self.context_hidden_dim)
                dec_attention_weights = torch.matmul(attn_query, dec_state.transpose(1, 2))
                # states['dec_attn'][:, address_mask, :, :2 ** d - 1] = dec_attention_weights.view(total_batch_size, n_addresses, seq_len, (2 ** d - 1))
                dec_attention_weights = F.softmax(dec_attention_weights, dim=2)
                dec_attention_values = torch.matmul(dec_attention_weights, dec_state).view(total_batch_size, self.context_hidden_dim)
                current_h = (current_h + dec_attention_values)  # / 2

            current_h = self.layer_norm(current_h)

        for intermediate in self.intermediate:
            current_h = self.dropout(self.activation(intermediate(current_h.clone())))

        current_o = self.O(current_h)

        current_y = current_o[:, :-1]

        y_hat[:, t, :] = current_y
        states['y_hat'] = y_hat

        if t == self.max_size - 1:
            raise StopIteration('max size reached')

        # grow tree

        current_y_argmaxes = torch.argmax(y_hat[:, t], dim=1) if y is None else y[:, t]
        cumulative_argmaxes = torch.cat([cumulative_argmaxes, current_y_argmaxes.view(-1, 1)], dim=1)

        mapped_cumulative_y = cumulative_argmaxes  # [:, :-1]
        mapped_current_y = mapped_cumulative_y[:, t]

        # TODO eventually
        shortcut_children = torch.zeros(mapped_current_y.size(), dtype=torch.bool, device=self.device)

        atom_mask = states['atom_mask']

        grow_mask = self.y_grow_condition(mapped_current_y) & atom_mask[:, t]  # TODO: find workaround for interpolated tagset

        current_y_argmaxes = current_y_argmaxes[grow_mask]
        grow_bool = torch.any(grow_mask)
        if not grow_bool:
            raise StopIteration('all trees have failed the grow condition')

        skip_grow_mask = grow_mask & ~shortcut_children

        atom_mask[:, t+1] = grow_mask
        states['atom_mask'] = atom_mask

        mask[:, t+1] = skip_grow_mask
        states['mask'] = mask

        if self.transf:
            ...
        else:
            current_h = current_h[grow_mask, :]
            current_y = self.output_embedder(current_y_argmaxes)
            next_h = self.rnn_cell(current_y, current_h)

            h[grow_mask, t + 1] = next_h

            states['h'] = h


class TreeRNNDecoder(NN):
    def __init__(self, hidden_dim, context_hidden_dim, labels, atomic_labels=None, reuse_embedding=True,
                 oracle='global',
                 grow_labels=[], mapped_grow_labels=None, max_depth=6, max_mapped_depth=None,
                 enc_attention=False, dec_attention=False, dropout=0.2, n_hidden=2, is_sexpr=False,
                 device=torch.device('cpu')):
        '''
        A top-down binary Tree-recurrent generator.
        '''
        super().__init__()

        self.max_depth = max_depth
        self.max_width = 2**max_depth
        self.address_dim = 2**(max_depth + 1) - 1

        self.max_mapped_depth = max_depth if max_mapped_depth is None else max_mapped_depth
        self.max_mapped_width = 2 ** self.max_mapped_depth
        self.mapped_address_dim = 2 ** (self.max_mapped_depth + 1) - 1

        self.hidden_dim = self.context_hidden_dim = context_hidden_dim

        self.featurized = False
        self.activation = F.gelu  # torch.nn.ReLU()  # TODO: try cube activation
        self.norm = F.log_softmax
        self.enc_attention = enc_attention
        self.dec_attention = dec_attention
        self.reuse_embedding = reuse_embedding
        self.attention = enc_attention or dec_attention
        self.oracle = oracle

        # ensure PAD is last index
        if PAD in labels:
            assert labels[PAD] == len(labels) - 1
            del labels[PAD]

        self.output_dim = len(labels)

        if self.max_depth > 0:
            if self.hidden_dim != self.context_hidden_dim:
                self.hidden_dim = self.context_hidden_dim

            self.l_rnn_cell = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)
            self.r_rnn_cell = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.intermediate = torch.nn.ModuleList()
        for _ in range(n_hidden):
            self.intermediate.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        if (self.dec_attention or not self.featurized) and self.max_depth > 0:
            self.embedding_matrix = torch.nn.Parameter(torch.rand(self.output_dim+1, self.hidden_dim, device=device) * 0.02)
            self.output_embedder = lambda x: F.embedding(x, self.embedding_matrix, padding_idx=self.output_dim,
                                                         scale_grad_by_freq=True)
            if self.reuse_embedding:
                self.O = lambda x: x @ (self.embedding_matrix.transpose(1, 0) + 1e-10)
            else:
                self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        else:
            self.O = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.context_hidden_dim, elementwise_affine=False)

        cr = CategoryReader()
        self.out_to_ix = {(k if (is_sexpr or k in (PAD, UNK)) else cr.read(k, validate=False).s_expr()): v for k, v in labels.items()}
        self.out_to_ix[PAD] = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

        self.address_map = AtomicAddressMap(self.address_dim, self.mapped_address_dim, self.out_to_ix,
                                            tgt_out_to_ix=atomic_labels)

        self.grow_labels = grow_labels
        self.mapped_grow_labels = torch.tensor(grow_labels if mapped_grow_labels is None else mapped_grow_labels,
                                               dtype=torch.long, device=device)
        self._address_mask = torch.tensor(range(self.address_dim), dtype=torch.float32, device=device) + 1

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


        #TODO: remove
        states['h_update_tracker'] = set()

        x = x.unsqueeze(1)
        for d in range(self.max_depth + 1):
            try:
                self._rnn(d, states, x, batch_size, seq_len, y=y, categories_gold=categories_gold)
            except StopIteration:
                break

        return states

    def _rnn(self, d, states, x, batch_size, seq_len, y=None, categories_gold=None):

        address_mask = torch.floor(torch.log2(self._address_mask)) == d
        n_addresses = torch.sum(address_mask).item()
        n_children = 2 * n_addresses
        total_batch_size = x.size(0)

        mask = states['mask']
        y_hat = states['y_hat']

        # predict node label

        h = states['h']
        current_h = h[:, address_mask, :]

        if y is not None and self.oracle != 'global':
            y = states['y']

        with torch.no_grad():
            cumulative_argmaxes = (torch.argmax(y_hat, dim=2) if y is None else y.view(total_batch_size, -1))[:, :2 ** d - 1]
            cumulative_argmaxes[~mask[:, :2 ** d - 1]] = self.out_to_ix[PAD]

        if self.attention:
            attn_query = current_h.view(batch_size, seq_len * n_addresses, self.hidden_dim)

            if self.enc_attention:
                enc_state = x.view(batch_size, seq_len, self.context_hidden_dim)
                enc_attention_weights = torch.matmul(attn_query, enc_state.transpose(1, 2))
                # states['enc_attn'][:, address_mask, :] = enc_attention_weights.view(total_batch_size, n_addresses, seq_len)
                enc_attention_weights = F.softmax(enc_attention_weights, dim=2)
                enc_attention_values = torch.matmul(enc_attention_weights, enc_state).view(total_batch_size, n_addresses,
                                                                                           self.context_hidden_dim)
                current_h = (current_h + enc_attention_values)

            if self.dec_attention and d >= 1:
                dec_state = self.output_embedder(cumulative_argmaxes.clone())

                dec_state = dec_state.view(batch_size, seq_len * (2 ** d - 1), self.context_hidden_dim)
                dec_attention_weights = torch.matmul(attn_query, dec_state.transpose(1, 2))
                # states['dec_attn'][:, address_mask, :, :2 ** d - 1] = dec_attention_weights.view(total_batch_size, n_addresses, seq_len, (2 ** d - 1))
                dec_attention_weights = F.softmax(dec_attention_weights, dim=2)
                dec_attention_values = torch.matmul(dec_attention_weights, dec_state).view(total_batch_size, n_addresses,
                                                    self.context_hidden_dim)
                current_h = (current_h + dec_attention_values)

            current_h = self.layer_norm(current_h)

        for intermediate in self.intermediate:
            current_h = self.dropout(self.activation(intermediate(current_h.clone())))

        current_o = self.O(current_h)

        if (self.dec_attention or not self.featurized) and self.reuse_embedding and self.max_depth > 0:
            current_y = current_o[:, :, :-1]
        else:
            current_y = current_o

        y_hat[:, address_mask, :] = current_y
        states['y_hat'] = y_hat

        if d == self.max_depth:
            raise StopIteration('max depth reached')

        # grow tree

        with torch.no_grad():
            current_y_argmaxes = torch.argmax(y_hat[:, 2 ** d - 1:2 ** (d + 1) - 1], dim=2)
            if y is not None:
                if self.oracle != 'global':
                    categories_hat = self.extract_outputs(cumulative_argmaxes.view(batch_size, seq_len, -1), output_none=True)
                    for s, (sent_current, sent_tags_hat, sent_tags) in enumerate(zip(current_y_argmaxes.view(batch_size, seq_len, -1), categories_hat, categories_gold)):
                        for t, (tag_current, tag_hat, tag) in enumerate(zip(sent_current, sent_tags_hat, sent_tags)):
                            if tag is not None and tag.root is not None:
                                for a, current in enumerate(tag_current, start=2**d - 1):
                                    fillers = find_valid_fillers(tag_hat, a + 1, tag, self.out_to_ix, k=8, binary=False)
                                    if fillers is None:
                                        # TODO: have to handle this?
                                        continue
                                    filler_ix = sorted(self.out_to_ix[l] for l in fillers)
                                    if current in filler_ix:
                                        y[s, t, a] = current
                                    else:
                                        if self.oracle == 'dynamic_best':  # highest-ranked by model (current) vs highest-ranked by external metric
                                            y[s, t, a] = filler_ix[torch.argmax(y_hat.view(batch_size, seq_len, self.address_dim, -1)[s, t, a, filler_ix]).item()]
                                        elif self.oracle == 'dynamic_random':
                                            y[s, t, a] = random.choice(filler_ix)
                                        else:
                                            raise NotImplementedError
                states['y'] = y
                current_y_argmaxes = y.view(total_batch_size, -1)[:, 2 ** d - 1:2 ** (d + 1) - 1]

            cumulative_argmaxes = torch.cat([cumulative_argmaxes, current_y_argmaxes], dim=1)

            mapped_cumulative_y = self.address_map(cumulative_argmaxes, indices=True)
            mapped_cumulative_y = mapped_cumulative_y[:, :, :-1]
            mapped_current_y = mapped_cumulative_y[:, 2 ** d - 1:2 ** (d + 1) - 1]
            shortcut_children = torch.any(mapped_cumulative_y[:, 2 ** (d + 1) - 1:2 ** (d + 2) - 1].bool(), 2)

            atom_mask = states['atom_mask']

            grow_mask = self.y_grow_condition(mapped_current_y) & atom_mask[:, address_mask]

            current_y_argmaxes = current_y_argmaxes[grow_mask]
            grow_bool = torch.any(grow_mask, 1)
            if not torch.any(grow_bool, 0):
                raise StopIteration('all trees have failed the grow condition')

            double_grow_mask = grow_mask.view(-1, 1).repeat(1, 2).view(total_batch_size, -1)
            child_mask = (torch.floor(torch.log2(self._address_mask)) == d + 1)
            _address_mask = self._address_mask[child_mask].long() - 1

            skip_double_grow_mask = double_grow_mask & ~shortcut_children

            atom_mask[:, _address_mask] = double_grow_mask
            states['atom_mask'] = atom_mask

            mask[:, _address_mask] = skip_double_grow_mask
            states['mask'] = mask

            grow_child_mask = torch.zeros(total_batch_size, self.address_dim, dtype=torch.bool, device=self.device)
            grow_child_mask[:, _address_mask] = skip_double_grow_mask

            l_mask = (self._address_mask % 2 == 0)
            r_mask = (self._address_mask % 2 == 1)

            l_mask = grow_child_mask & l_mask
            r_mask = grow_child_mask & r_mask
            _l_mask = skip_double_grow_mask[:, :-1:2]
            _r_mask = skip_double_grow_mask[:, 1::2]

        current_h = current_h[grow_mask, :]

        current_y = self.output_embedder(current_y_argmaxes)
        l_h = self.l_rnn_cell(current_y, current_h)
        r_h = self.r_rnn_cell(current_y, current_h)

        h = h.masked_scatter(
            l_mask.unsqueeze(-1).expand(l_mask.shape[0], l_mask.shape[1], h.shape[2]),
            l_h[_l_mask[grow_mask]])

        h = h.masked_scatter(
            r_mask.unsqueeze(-1).expand(r_mask.shape[0], r_mask.shape[1], h.shape[2]),
            r_h[_r_mask[grow_mask]])

        states['h'] = h
