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


class AddrMLPDecoder(NN):
    def __init__(self, hidden_dim, context_hidden_dim, labels, atomic_labels=None, reuse_embedding=True,
                 oracle='global',
                 grow_labels=[], mapped_grow_labels=None, max_depth=6, max_mapped_depth=None,
                 enc_attention=False, dec_attention=False, dropout=0.2, n_hidden=2, is_sexpr=False,
                 device=torch.device('cpu')):
        '''
        A top-down binary Tree-structured non-recurrent generator.
        '''
        super().__init__()

        self.max_depth = max_depth
        self.max_width = 2**max_depth
        self.address_dim = 2**(max_depth + 1) - 1

        self.max_mapped_depth = max_depth if max_mapped_depth is None else max_mapped_depth
        self.max_mapped_width = 2 ** self.max_mapped_depth
        self.mapped_address_dim = 2 ** (self.max_mapped_depth + 1) - 1

        self.hidden_dim = self.context_hidden_dim = context_hidden_dim

        self.featurized = True
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

        self.child_enc = torch.zeros(self.address_dim, 2 * self.max_depth, dtype=torch.float32, requires_grad=False,
                                     device=device)
        for a in range(2, self.address_dim + 1):
            string = f'{a:b}'[1:]
            binary = torch.tensor([1 if c == '0' else -1 for c in string], dtype=torch.float32, device=device)
            depth = len(binary)
            self.child_enc[a - 1, :depth] = binary
            self.child_enc[a - 1, -depth:] = binary

        self.feat_linear = torch.nn.Linear(4 * self.max_depth, self.hidden_dim)

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
                        # attend to (address 10^{m-nargs(a_k)} (2^{m-nargs(a_k)}) of) dep(head, k).
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
        states['h'] = h

        slash_enc = torch.zeros(total_batch_size, self.address_dim, self.max_depth, dtype=torch.float32,
                                device=self.device, requires_grad=False)
        states['slash_enc'] = slash_enc

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
        '''
        :param parent_h: the hidden output representation of the parent node
        :param parent_c: the hidden context representation of the parent node
        :param y: oracle parent label if available
        :return: hidden and context representations of the left and right children
        '''

        address_mask = torch.floor(torch.log2(self._address_mask)) == d
        n_addresses = torch.sum(address_mask).item()
        n_children = 2 * n_addresses
        total_batch_size = x.size(0)

        mask = states['mask']
        y_hat = states['y_hat']

        # predict node label

        h = states['h']
        current_h = h[:, address_mask, :]

        current_h = x.expand(-1, n_addresses, -1) + current_h
        current_h = self.layer_norm(current_h)

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

        slash_enc = torch.zeros(total_batch_size, n_children, 2 * self.max_depth,
                                requires_grad=False, dtype=torch.float32, device=self.device)
        slashes = states['slash_enc'][:, address_mask, self.max_depth - d:]

        parent_argmaxes = torch.argmax(mapped_current_y, dim=2, keepdim=True)

        f_slash_mask = parent_argmaxes == self.mapped_grow_labels[0]
        b_slash_mask = parent_argmaxes == self.mapped_grow_labels[1]

        slashes = torch.cat((slashes, (b_slash_mask.float() * -1. + f_slash_mask.float())), dim=2)
        slashes = slashes.view(-1, d+1).repeat(1, 2).view(total_batch_size, n_children, -1)

        slash_enc[:, :, :d+1] = slashes
        slash_enc[:, :, -d-1:] = slashes
        states['slash_enc'][:, child_mask, -d-1:] = slashes

        child_enc = self.child_enc[child_mask]
        child_enc = child_enc.unsqueeze(0).expand(total_batch_size, -1, -1)

        feats = torch.cat([child_enc, slash_enc], dim=-1)

        h_updates = set(grow_child_mask.view(-1).nonzero(as_tuple=False).view(-1).tolist())
        repeated_updates = h_updates.intersection(states['h_update_tracker'])
        if repeated_updates:
            print('\nWARNING: attempted to make repeated updates to the following indices:', tuple(sorted(repeated_updates)), file=sys.stderr)
        else:
            pass

        states['h_update_tracker'] |= h_updates

        h = h.masked_scatter(grow_child_mask.unsqueeze(-1).expand(grow_child_mask.shape[0], grow_child_mask.shape[1], h.shape[2]), self.dropout(self.activation(self.feat_linear(feats)))[skip_double_grow_mask, :])
        states['h'] = h
