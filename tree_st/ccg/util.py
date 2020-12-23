'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from .category import Category, InvalidCategoryError
from ..util.reader import SCategoryReader


def sexpr_nodeblock_to_cat(nodeblock: set, binary=True, validate=True):
    cr = SCategoryReader()
    result = None
    for item in sorted(nodeblock, key=lambda x: int(x[0])):
        a, l = item
        a = int(a)
        if a == 1:
            result = cr.read(l, validate=False)
        else:
            try:
                result.set_category(a, cr.read(l, validate=False), binary=binary, validate=False)
            except (ValueError, AttributeError) as e:
                if validate:
                    raise InvalidCategoryError(e)
    if validate:
        if result is None:
            raise InvalidCategoryError('Empty category')
        msg = result.validate()
        if msg != 0:
            raise InvalidCategoryError(msg)
    return result


def sexpr_seq_to_cat(nodeblock: list, with_sep=True, validate=True):
    from ccg.parser.scoring.nn import SEP, PAD

    if len(nodeblock) == 0:
        if validate:
            raise InvalidCategoryError('Empty category')
        return None
    cr = SCategoryReader()
    result = Category(None, validate=False)
    i = 0
    for l in nodeblock:
        a = result.next_open_address()
        if with_sep:
            if l == PAD:
                return None, []
            if l == SEP:
                break
            if a is None:
                if l != SEP:  # TODO: this line should not be necessary
                    if validate:
                        assert i == len(nodeblock), f'No open addresses in {result.s_expr()}. ' \
                            f'Expected sequence {nodeblock} to end at position {i}.'
                    _l = l
                    while _l != SEP and i < len(nodeblock) - 1:
                        i += 1
                        _l = nodeblock[i]
                return result, nodeblock[i+1:]
        elif a is None:
            return result, nodeblock[i:]
        result.set_category(a, cr.read(l, validate=False), binary=True, validate=False)
        i += 1
    if validate:
        if result is None or result.root is None:
            raise InvalidCategoryError('Empty category')
        msg = result.validate()
        if msg != 0:
            raise InvalidCategoryError(msg)
    return result, nodeblock[i+1:]
