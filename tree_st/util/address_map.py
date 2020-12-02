'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import torch
import torch.nn.functional as F

from ..tagger.nn import PAD, UNK
from .reader import SCategoryReader
from .functions import binary_to_decimal

ATOMIC = {
  "(S)": 0,
  "(S[dcl])": 1,
  "(S[wq])": 2,
  "(S[q])": 3,
  "(S[qem])": 4,
  "(S[em])": 5,
  "(S[bem])": 6,
  "(S[b])": 7,
  "(S[frg])": 8,
  "(S[for])": 9,
  "(S[intj])": 10,
  "(S[inv])": 11,
  "(S[to])": 12,
  "(S[pss])": 13,
  "(S[pt])": 14,
  "(S[ng])": 15,
  "(S[as])": 16,
  "(S[asup])": 17,
  "(S[poss])": 18,
  "(S[adj])": 19,
  "(NP)": 20,
  "(NP[nb])": 21,
  "(NP[expl])": 22,
  "(NP[thr])": 23,
  "(N)": 24,
  "(N[num])": 25,
  "(PP)": 26,
  "(,)": 27,
  "(.)": 28,
  "(conj)": 29,
  "(:)": 30,
  "(;)": 31,
  "(RRB)": 32,
  "(LRB)": 33,
  "(/)": 34,
  "(\\)": 35
}


class AddressMap(torch.nn.Module):
    def __init__(self, src_address_dim, tgt_address_dim, src_out_to_ix, *args, tgt_out_to_ix=ATOMIC):
        super(AddressMap, self).__init__()
        self.src_address_dim = src_address_dim

        self.src_out_to_ix = src_out_to_ix
        if PAD not in self.src_out_to_ix:
            self.src_out_to_ix[PAD] = len(self.src_out_to_ix)
        self.src_output_dim = len(self.src_out_to_ix)
        self.src_ix_to_out = {v: k for k, v in self.src_out_to_ix.items()}

        self.address_dim = tgt_address_dim

        self.out_to_ix = dict(tgt_out_to_ix or ATOMIC)
        if PAD not in self.out_to_ix:
            self.out_to_ix[PAD] = len(self.out_to_ix)
        self.output_dim = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

    def forward(self, x, indices=False, norm=False, argmax=False):
        assert not (norm and argmax)
        batch_size, n_addresses = x.size(0), x.size(1)
        if indices:
            x = F.one_hot(x, num_classes=self.src_output_dim).float()
        else:
            n_outputs = x.size(2)
            if n_outputs < self.src_output_dim:
                x = torch.cat([x, torch.zeros(batch_size, n_addresses, self.src_output_dim-n_outputs).to(x)], dim=2)

        x = torch.matmul(x.reshape(batch_size, n_addresses * self.src_output_dim),
                         self.map[:n_addresses].view(n_addresses * self.src_output_dim, -1)
                         ).view(-1, self.address_dim, self.output_dim)

        if argmax:
            x = torch.argmax(x, dim=2)
        elif norm:
            x = self.norm(x)

        return x


class DummyAddressMap(AddressMap):
    def __init__(self, src_address_dim, tgt_address_dim, src_out_to_ix, *args, tgt_out_to_ix=ATOMIC):
        super(AddressMap, self).__init__()
        self.src_address_dim = src_address_dim

        self.src_out_to_ix = src_out_to_ix
        if PAD not in self.src_out_to_ix:
            self.src_out_to_ix[PAD] = len(self.src_out_to_ix)
        self.src_output_dim = len(self.src_out_to_ix)

        self.address_dim = src_address_dim

        self.out_to_ix = src_out_to_ix
        self.output_dim = len(self.out_to_ix)
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

    def forward(self, x, indices=False, norm=False, argmax=False):
        assert not (norm and argmax)
        batch_size, n_addresses = x.size(0), x.size(1)
        if indices:
            x = F.one_hot(x, num_classes=self.src_output_dim).float()
        else:
            n_outputs = x.size(2)
            if n_outputs < self.src_output_dim:
                x = torch.cat([x, torch.zeros(batch_size, n_addresses, self.src_output_dim - n_outputs).to(x)], dim=2)

        if argmax:
            x = torch.argmax(x, dim=2)

        return x


class AtomicAddressMap(AddressMap):
    def __init__(self, src_address_dim, tgt_address_dim, src_out_to_ix, tgt_out_to_ix=ATOMIC):
        super(AtomicAddressMap, self).__init__(src_address_dim, tgt_address_dim, src_out_to_ix, tgt_out_to_ix=tgt_out_to_ix)
        _map = torch.zeros(self.src_address_dim, self.src_output_dim, self.address_dim, self.output_dim,
                           requires_grad=False)
        csr = SCategoryReader()
        for a in range(1, self.src_address_dim + 1):
            for i in range(self.src_output_dim):
                cs = self.src_ix_to_out[i]
                if cs == PAD:
                    continue
                if cs == UNK:
                    continue
                c = csr.read(cs, validate=False)
                nb = dict(c.decompose(self.out_to_ix, binary=True))
                _a = f'{int(a):b}'
                for __a, l in sorted(nb.items()):
                    try:
                        _map[a - 1, i, binary_to_decimal(int(_a + str(__a)[1:])) - 1, self.out_to_ix[l]] = 1.
                    except IndexError:
                        break
                    except KeyError:
                        print(cs, nb)
                        raise
        self.register_buffer('map', _map)
        self.norm = torch.nn.LayerNorm(self.output_dim, elementwise_affine=False)
        # self.norm = lambda t: F.normalize(t, p=2, dim=-1)
        # self.norm = torch.nn.InstanceNorm1d(self.mapped_output_dim)
        # self.norm = lambda t: t / (self.address_dim * self.output_dim)
        # self.norm = lambda t: t


class FullAddressMap(AddressMap):
    def __init__(self, src_address_dim, src_out_to_ix, tgt_out_to_ix=None):
        super(FullAddressMap, self).__init__(src_address_dim, 1, src_out_to_ix, tgt_out_to_ix=tgt_out_to_ix)
        _map = torch.zeros(self.src_address_dim, self.src_output_dim, 1, self.output_dim,
                           requires_grad=False)
        csr = SCategoryReader()
        for i in range(self.output_dim):
            cs = self.ix_to_out[i]
            if cs == PAD:
                continue
            if cs == UNK:
                continue
            c = csr.read(cs, validate=False)
            nb = dict(c.decompose(self.src_out_to_ix, binary=True))
            for __a, l in sorted(nb.items()):
                try:
                    _map[binary_to_decimal(__a) - 1, self.src_out_to_ix[l], 0, i] = 1.
                except IndexError:
                    break
                except KeyError:
                    print(cs, nb)
                    raise
        self.register_buffer('map', _map)
        self.norm = torch.nn.LayerNorm(self.output_dim, elementwise_affine=False)
        # self.norm = lambda t: F.normalize(t, p=2, dim=-1)
        # self.norm = torch.nn.InstanceNorm1d(self.mapped_output_dim)
        # self.norm = lambda t: t / (self.address_dim * self.output_dim)
        # self.norm = lambda t: t


if __name__ == '__main__':
    address_dim = 7
    output_dim = 5
    src = {'(NP)': 0, '(/)': 1, '(\\)': 2, '(\\(~R(NP))(~A(NP)))': 3, '(/(~R(NP))(~A(NP)))': 4}
    atomic = {'(NP)': 0, '(/)': 1, '(\\)': 2}
    full = {'(NP)': 0, '(\\(~R(NP))(~A(NP)))': 1, '(/(~R(NP))(~A(NP)))': 2, '(/(~R(\\(~R(NP))(~A(NP))))(~A(NP)))': 3}

    y1 = torch.tensor([[[.05, .05, .5, .2, .2],
                        [.1, .7, 0, .1, .1],
                        [.95, 0, 0, .05, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]],
                       [[.05, .15, .45, .35, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]],
                       [[0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]
                       ])
    print('y1', y1, (3, 7, 5), y1.size())

    y3 = torch.tensor([[2, 3, 0, 5, 5, 5, 5],
                       [4, 5, 5, 5, 5, 5, 5]])
    print('y3', y3, (2, 7, 1), y3.size())

    am = AtomicAddressMap(address_dim, address_dim, src, tgt_out_to_ix=atomic)

    y2 = am(y1)
    print('y2', torch.argmax(y2, 2), (3, 7, 4), y2.size())

    y4 = am(y3, indices=True, argmax=False, norm=False)
    print('y4', torch.argmax(y4, 2), (2, 7, 4), y4.size())

    am2 = FullAddressMap(address_dim, src, tgt_out_to_ix=full)

    y5 = am2(y1, argmax=True, norm=False)
    print('y5', y5)

    y6 = am2(y3, indices=True, argmax=True, norm=False)
    print('y6', y6)

    am3 = FullAddressMap(address_dim, atomic, tgt_out_to_ix=full)

    y7 = am3(y2[:, :, :-1], argmax=True, norm=False)
    print('y7', y7)

    # y8 = am3(y4[:, :, :-1], indices=False, argmax=True, norm=False)
    # print('y8', y8)
