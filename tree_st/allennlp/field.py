'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from typing import Dict, List, Iterable, Optional

import numpy

from allennlp.common import JsonDict
from allennlp.data.fields.text_field import TextField
from allennlp.data.fields.sequence_label_field import SequenceLabelField

from ..ccg.category import Category
from ..util.functions import binary_to_decimal


def cat_as_json(cat: Category, bin_addr_prefix='1') -> JsonDict:
    json = {'root': cat.root, 'attr': list(cat.attr), 'category': str(cat),
            'addr_bin': bin_addr_prefix, 'addr': binary_to_decimal(int(bin_addr_prefix))}
    if cat.result:
        json['result'] = cat_as_json(cat.result, bin_addr_prefix=f'{bin_addr_prefix}0')
    if cat.arg:
        json['arg'] = cat_as_json(cat.arg, bin_addr_prefix=f'{bin_addr_prefix}1')
    return json


def json_to_cat(json_dict: JsonDict) -> Category:
    return Category(json_dict['root'],
                    result=json_to_cat(json_dict['result']) if 'result' in json_dict else None,
                    arg=json_to_cat(json_dict['arg']) if 'arg' in json_dict else None)


class ConstructiveSupertagField(SequenceLabelField):
    def __init__(self, cat: Iterable[Category], sequence_field: TextField, indices_in_sequence: Optional[List[int]] = None, **kwargs) -> None:
        self.cat = cat
        self.indices_in_sequence = indices_in_sequence

        labels = self.as_tag()
        if indices_in_sequence is not None:
            assert len(indices_in_sequence) == len(labels), f'{indices_in_sequence}, {labels} ({len(indices_in_sequence)} != {len(labels)})'
            _labels = ['O'] * len(sequence_field)
            for label, idx in zip(labels, indices_in_sequence):
                _labels[idx] = label
            labels = _labels
        super(ConstructiveSupertagField, self).__init__(labels, sequence_field)

    def as_cat(self, **kwargs) -> Iterable[Category]:
        return self.cat

    def as_json(self, **kwargs) -> Iterable[JsonDict]:
        return [cat_as_json(cat) for cat in self.as_cat()]

    def as_tag(self, sexpr: bool = False) -> Iterable[str]:
        if sexpr:
            return [t.s_expr() for t in self.as_cat()]
        return [str(t) for t in self.as_cat()]

    def __str__(self):
        string = str(self.as_cat())
        if self.indices_in_sequence is not None:
            return f'{self.indices_in_sequence} {[self.sequence_field[i] for i in self.indices_in_sequence]}: {string}'
        return string

    def __eq__(self, other):
        if isinstance(other, ConstructiveSupertagField):
            return self.as_cat() == other.as_cat()
        if all(map((lambda x: isinstance(x, Category)), other)):
            return self.as_cat() == other
        elif all(map((lambda x: isinstance(x, str)), other)):
            return self.as_tag() == other or self.as_tag(sexpr=True) == other
        else:
            raise NotImplementedError(type(other))

