'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from ..ccg.derivation import DerivationLeafNode, DerivationTreeNode


def _write_auto(node):
    if isinstance(node, DerivationTreeNode):
        return f'(<T {node.category1} {node.head} {len(node.args)}> {"".join(map(_write_auto, node.args))}) '
    elif isinstance(node, DerivationLeafNode):
        return f'(<L {node.category1} {node.pos1} {node.pos2} {node.word} {node.category2}>) '
    else:
        raise NotImplementedError


def write_auto(derivation):
    return _write_auto(derivation.root)


def _write_pretty(node):
    if isinstance(node, DerivationTreeNode):
        args_str = super().paste_list(list(map(lambda a: _write_pretty, node.args)))
        cat_str = str(node.category1)
        comb_str = str(node.combinator) if node.combinator else ''
        m = len(args_str.split('\n')[0])
        n = max(m, len(cat_str), len(comb_str))
        return '\n'.join([f'{args_str}{" " * (n - m)}', f'{"_" * (n - len(comb_str))}{comb_str}',
                          f'{cat_str}{" " * (n - len(cat_str))}'])
    elif isinstance(node, DerivationLeafNode):
        cat_str = str(node.category1)
        n = max(len(node.word), len(cat_str))
        return '\n'.join([f'{node.word}{" " * (n - len(node.word))}', '_' * n, f'{cat_str}{" " * (n - len(cat_str))}'])
    else:
        raise NotImplementedError


def write_pretty(derivation):
    return f'{_write_pretty(derivation.root)}\n'


def escape_category(category):
    attr_str = ''.join((f'[{a}]' for a in category.attr - {'conj'})) + ('[conj]' if 'conj' in category.attr else '')
    if category.result is None:
        result_str = ''
    else:
        result_str = escape_category(category.result)
        if category.result.has_children():
            result_str = f'_LRB_{result_str}_RRB_'
    if category.arg is None:
        arg_str = ''
    else:
        arg_str = escape_category(category.arg)
        if category.arg.has_children():
            arg_str = f'_LRB_{arg_str}_RRB_'
    cat_str = f'{result_str}{category.root}{arg_str}'
    return f'{cat_str}{attr_str}'


def _write_ptb(node, prefix='', combinator=None, head=False):
    if isinstance(node, DerivationTreeNode):
        prefix += f'({escape_category(node.category1)}'
        if combinator is not None:
            prefix += f'-{combinator}'
        indent = len(prefix)
        result = f'{_write_ptb(node.args[0], prefix=prefix + " ", combinator=node.combinator, head=node.head == 0)}'
        for i, arg in enumerate(node.args[1:], start=1):
            result += f'\n{_write_ptb(arg, prefix=" " * indent + " ", combinator=node.combinator, head=node.head==i)}'
        result += ')'
        return result
    elif isinstance(node, DerivationLeafNode):
        result = prefix + f'({escape_category(node.category1)}'
        if combinator is not None:
            result += f'-{combinator}'
        result += f' {node.word})'
        return result
    else:
        raise NotImplementedError


def write_ptb(derivation):
    return _write_ptb(derivation.root)


if __name__ == '__main__':
    from ccg.util.reader import AUTODerivationsReader
    import glob
    from tqdm import tqdm

    with open('data/ccgbank/ccgbank.ptb', 'w', newline='\n') as out:
        for filename in tqdm(glob.glob('data/ccgbank/AUTO/*/*.auto')):
            dr = AUTODerivationsReader(filename, validate=False)  # this is an iterator-type class
            for deriv_and_meta in dr:  # this is a dictionary containing the Derivation object and some metadata
                deriv = deriv_and_meta['DERIVATION']

                print(write_ptb(deriv), file=out)
                print(file=out)
