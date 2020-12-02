'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''




LETTERS = '_YZWVUTS'


def index_category(category, nargs, result_index=0, arg_index=1, self_index=0):
    attr_str = ''.join((f'[{a}]' for a in category.attr - {'conj'})) + ('[conj]' if 'conj' in category.attr else '')
    if category.result is None:
        result_str = ''
    else:
        if category.result.equals(category.arg, ignore_attr=True):
            if len(category.arg.attr) == 0 or (category.arg.attr == category.result.attr):
                result_index = arg_index


        result_str, result_index, arg_index, self_index = index_category(category.result, nargs-1, result_index, arg_index, result_index)
        # if category.result.has_children():
        #     result_str = f'({result_str})'
    if category.arg is None:
        arg_str = ''
    else:
        arg_str, result_index, arg_index, self_index = index_category(category.arg, -1, arg_index, arg_index+1, arg_index)
        # if category.arg.has_children():
        #     arg_str = f'({arg_str})'
    if not category.has_children():
        cat_str = f'{category.root}{attr_str}{{{LETTERS[self_index]}}}'
    elif nargs < 0:
        cat_str = f'({result_str}{category.root}{arg_str}){attr_str}{{{LETTERS[self_index]}}}'
    else:
        cat_str = f'({result_str}{category.root}{arg_str}<{nargs}>){attr_str}{{{LETTERS[self_index]}}}'
    return cat_str, result_index, arg_index, self_index


def make_rule(category):
    result = str(category)
    nargs = category.nargs()
    indexed, _, _, _ = index_category(category, nargs)
    result += f'\n  {nargs} {indexed}'
    cat = category
    for n in range(nargs):
        result += f'\n  {n+1} ignore'
        cat = cat.result
    return result
