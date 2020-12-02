'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import math
from collections import Counter, OrderedDict, defaultdict

from ..util.functions import binary_to_decimal
from ..util.strings import one_bit, zero_bit, slash


class InvalidCategoryError(Exception):
    pass


class Category:
    def __init__(self, root, result=None, arg=None, attr={}, validate=True):
        self.root = root.root if isinstance(root, Category) else root
        self.result: Category = result if (result is None or isinstance(result, Category)) else Category(result, validate=False)
        self.arg: Category = arg if (arg is None or isinstance(arg, Category)) else Category(arg, validate=False)
        self.attr = set(attr)
        if validate:
            msg = self.validate()
            if msg != 0:
                raise InvalidCategoryError(msg)

    def validate(self):
        if not self.root:
            return 'Missing root', f'({self.root} {self.result} {self.arg})'
        if slash(self.root):
            if not self.has_children():
                return 'Slash root does not have children', f'({self.root} {self.result} {self.arg})'
            if (self.result is None) != (self.arg is None):
                return 'Unbalanced subtree', f'({self.root} {self.result} {self.arg})'
            return self.result.validate() or self.arg.validate()
        else:
            if self.has_children():
                return 'Non-slash root has children', f'({self.root} {self.result} {self.arg})'
        return 0

    def get_category(self, address, binary=True, default=None):
        address = str(address) if binary else f'{int(address):b}'
        current = self
        for bit in address[1:]:
            # print(current, bit, self, address[1:])
            if current is None or not current.has_children():
                return default
            if zero_bit(bit):
                current = current.result
            elif one_bit(bit):
                current = current.arg
            else:
                raise ValueError(f'address should be binary or decimal int, got {address}.')
        return current

    def get_node(self, address, binary=True, default=None):
        c = self.get_category(address, binary, default)
        return c.root if c is not None else None

    def set_category(self, address, c, binary=True, validate=True):
        address = str(address) if binary else f'{int(address):b}'
        if not address:
            raise ValueError('empty address')
        elif address == '1':
            if c is None:
                self.root = None
                self.arg = None
                self.result = None
                self.attr = set()
            else:
                self.root = c.root
                self.arg = c.arg
                self.result = c.result
                self.attr = c.attr
            if validate:
                msg = self.validate()
                if msg != 0:
                    raise InvalidCategoryError(msg)
            return
        cat = self.get_category(address[:-1])
        if cat is None:
            raise ValueError(f'address {address[:-1]} not available in category {self}')
        bit = address[-1]
        if zero_bit(bit):
            cat.result = c
        elif one_bit(bit):
            cat.arg = c
        else:
            raise ValueError(f'address should be binary or decimal int, got {address}.')
        if validate:
            msg = self.validate()
            if msg != 0:
                raise InvalidCategoryError(msg)

    def set_node(self, address, n, attr=None, binary=True, validate=True):
        address = str(address) if binary else f'{int(address):b}'
        cat = self.get_category(address[:-1])
        if cat is None:
            raise ValueError(f'address {address} not available in category {self}')
        bit = address[-1]
        if zero_bit(bit):
            if cat.result:
                cat.result.root = n
            else:
                cat.result = Category(n, validate=validate)
            if attr is not None:
                for _attr in attr:
                    cat.result.add_attr(_attr)
        elif one_bit(bit):
            if cat.arg:
                cat.arg.root = n
            else:
                cat.arg = Category(n, validate=validate)
            if attr is not None:
                for _attr in attr:
                    cat.arg.add_attr(_attr)
        else:
            raise ValueError(f'address should be binary or decimal int, got {address}.')
        if validate:
            msg = self.validate()
            if msg != 0:
                raise InvalidCategoryError(msg)

    def copy(self):
        return Category(self.root, None if self.result is None else self.result.copy(),
                                   None if self.arg is None else self.arg.copy(), self.attr.copy(), validate=False)

    def replace(self, address, c, binary=True):
        new = self.copy()
        new.set_category(address, c, binary)
        return new

    def has_attr(self, attr):
        return attr in self.attr

    def add_attr(self, attr):
        self.attr.add(attr)

    def remove_attr(self, attr):
        if attr in self.attr:
            self.attr.remove(attr)

    def with_attr(self, address, attr, binary=True):
        new = self.copy()
        new.get_category(address, binary).add_attr(attr)
        return new

    def without_attr(self, address=None, attr=None, binary=True):
        new = self.copy()
        if attr is None:
            if address is not None:
                new.get_category(address, binary).attr = set()
            else:
                new.attr = set()
                if new.has_children():
                    new.arg = new.arg.without_attr()
                    new.result = new.result.without_attr()
        elif attr in new.get_category(address, binary).attr:
            if address is not None:
                new.get_category(address, binary).remove_attr(attr)
            else:
                new.remove_attr(attr)
                if new.has_children():
                    new.arg = new.arg.without_attr(attr=attr)
                    new.result = new.result.without_attr(attr=attr)
        return new

    def is_node(self, address, n, binary=True):
        return self.get_node(address, binary) == n

    def is_category(self, address, c, binary=True):
        return self.get_category(address, binary) == c

    def has_children(self):
        return self.result is not None or self.arg is not None

    def next_open_address(self, order='pre', arg_first=False, start='1'):
        if order != 'pre':
            raise NotImplementedError(f'order={order}')
        if arg_first:
            raise NotImplementedError(f'arg_first')

        if self.root is None:
            return start
        elif slash(self.root):
            if self.result is None:
                return f'{start}0'
            else:
                result = self.result.next_open_address(start=f'{start}0', order=order, arg_first=arg_first)
                if result is None:
                    if self.arg is None:
                        return f'{start}1'
                    else:
                        return self.arg.next_open_address(start=f'{start}1', order=order, arg_first=arg_first)
                else:
                    return result
        else:
            return None

    def nargs(self):
        if not self.has_children():
            return 0
        return (self.result.nargs() if self.result is not None else 0) + 1

    def depth(self):
        if not self.has_children():
            return 0
        return max(self.result.depth() if self.result is not None else 0,
                   self.arg.depth() if self.arg is not None else 0) + 1

    def size(self):
        if not self.has_children():
            return 1
        return (self.result.size() if self.result is not None else 0) \
               + (self.arg.size() if self.arg is not None else 0) + 1

    def leftmost_result(self):
        if not self.has_children():
            return self.root
        return self.result.leftmost_result()

    def count_atomic_categories(self, concat_attr=False):
        result = Counter()
        if not self.has_children():
            cat_str = self.root
            if concat_attr and self.attr:
                attr = self.attr - {'conj'}
                assert len(attr) <= 1, attr
                attr_str = f"[{''.join(attr)}]"
                cat_str = f"{cat_str}{attr_str}"
            result[cat_str] += 1
        else:
            result.update(self.result.count_atomic_categories(concat_attr=concat_attr))
            result.update(self.arg.count_atomic_categories(concat_attr=concat_attr))
        return result

    def count_addresses(self, labels=None):
        result = Counter()
        for i in range(1, 2**(self.depth()+1)):
            i_bin = f'{i:b}'
            cat = self.get_category(i_bin)
            if cat is not None and (not labels or cat.root in labels):
                result[i_bin] += 1
        return result

    def count_slashes(self):
        result = Counter()
        if self.has_children():
            result[self.root] += 1
        else:
            return result
        result.update(self.result.count_slashes())
        result.update(self.arg.count_slashes())
        return result

    def get_shape(self, keep_slashes=False, keep_atomcats=False):
        if not self.has_children():
            return Category(self.root if keep_atomcats else 'X', validate=False)
        return Category(self.root if keep_slashes else '|', self.result.get_shape() if self.result is not None else None, self.arg.get_shape() if self.arg is not None else None, validate=False)

    def to_sparse_nodeblock(self, max_address_depth=None, labels=None, concat_attr=False, binary=True):
        result = {}
        if max_address_depth is None:
            max_address_depth = self.depth()
        if labels is None:
            labels = tuple(self.count_atomic_categories().keys()) + tuple(self.count_slashes().keys())
        for i in range(1, 2 ** (max_address_depth + 1) - 1):
            _i = int(f'{i:b}') if binary else i
            cat = self.get_category(_i, binary=binary)
            if cat is not None:
                cat_str = cat.root
                if concat_attr and cat.attr:
                    attr = cat.attr - {'conj'}
                    assert len(attr) <= 1, attr
                    attr_str = f"[{''.join(attr)}]"
                    cat_str = f"{cat_str}{attr_str}"
            else:
                cat_str = None
            for lab in labels:
                result[_i, lab] = 1 if cat_str == lab else 0
        return result

    def traverse(self, order='pre', result_first=True, prefix='1'):
        if order == 'pre':
            yield (prefix, self.root)
            if result_first:
                if self.result is not None:  # TODO: need some hack for interpolated tagsets
                    for node in self.result.traverse(order=order, result_first=result_first, prefix=f'{prefix}0'):
                        yield node
                if self.arg is not None:
                    for node in self.arg.traverse(order=order, result_first=result_first, prefix=f'{prefix}1'):
                        yield node
            else:
                raise NotImplementedError(f'result_first={result_first}')
        else:
            raise NotImplementedError(f'order={order}')

    def _decompose(self, tagset, unk='<UNKNOWN>', prefix='1'):
        if self.s_expr() in tagset:
            return {prefix: self}

        nb = {}

        top = Category(self.root, validate=False)
        if top.s_expr() in tagset:
            nb[prefix] = top
        else:
            nb[prefix] = Category(unk, validate=False)

        result_pre = f'{prefix}0'
        arg_pre = f'{prefix}1'
        result_dec = self.result._decompose(tagset, prefix=result_pre) if self.result is not None else {}
        arg_dec = self.arg._decompose(tagset, prefix=arg_pre) if self.arg is not None else {}
        children = sorted([('arg', arg_dec, arg_pre), ('result', result_dec, result_pre)], key=lambda x: sum(chunk.count('(') for chunk in x[1]) / max(1, len(x[1])), reverse=True)
        # TODO
        #  Either set reverse=False or measurement that also takes into account chunk size via Category.size().
        #  ... like sum(chunk.size() for chunk in chunks) / len(chunks), reverse=True

        # TODO: instead of this, maybe add a top-down search for largest connected component including root
        #  Then compare that with size of largest connected components of children
        for role, child_dec, child_pre in children:
            if child_dec:
                top_child = Category(self.root, **{role: child_dec[child_pre]}, validate=False)
                if top_child.s_expr() in tagset:
                    nb[prefix] = top_child
                    del child_dec[child_pre]
                    break

        nb.update(result_dec)
        nb.update(arg_dec)
        return nb

    def decompose(self, tagset, binary=True):
        result = {}
        for k, v in self._decompose(tagset).items():
            _k = int(k)
            result[_k if binary else binary_to_decimal(_k)] = v.s_expr()
        return sorted(result.items())

    def to_dense_nodeblock(self, ignore_attr=True, concat_attr=False, s_expr=False, binary=True):
        result = []
        for i in range(1, 2 ** (self.depth() + 1)):
            _i = int(f'{i:b}') if binary else i
            cat = self.get_category(_i, binary=binary)
            if cat is not None and cat.root is not None:
                cat_str = cat.root
                if concat_attr and cat.attr:
                    attr = cat.attr - {'conj'}
                    assert len(attr) <= 1, attr
                    attr_str = f"[{''.join(sorted(attr))}]"
                    cat_str = f"{cat_str}{attr_str}"
                if s_expr:
                    cat_str = f'({cat_str})'
                result.append((_i, cat_str) if ignore_attr else (_i, cat_str, frozenset(cat.attr)))
        return result

    @staticmethod
    def from_dense_nodeblock(nodeblock: set, binary=True, validate=True):
        result = None
        for item in sorted(nodeblock, key=lambda x: int(x[0])):
            if len(item) == 2:
                a, l = item
                attr = None
            elif len(item) == 3:
                a, l, attr = item
            a = int(a)
            if a == 1:
                result = Category(l, validate=False)
                if attr is not None:
                    for _attr in attr:
                        result.add_attr(_attr)
            else:
                try:
                    result.set_node(a, l, attr=attr, binary=binary, validate=False)
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

    def equals(self, other, ignore_attr=False):
        if other is None:
            return False
        if isinstance(other, str):
            cat_str = str(self.without_attr() if ignore_attr else self)
            return cat_str == other
        result_eq = other.result is None if self.result is None else self.result.equals(other.result, ignore_attr)
        arg_eq = other.arg is None if self.arg is None else self.arg.equals(other.arg, ignore_attr)
        return self.root == other.root and result_eq  and arg_eq and (ignore_attr or self.attr == other.attr)

    def s_expr(self):
        attr_str = ''.join((f'[{a}]' for a in self.attr-{'conj'})) + ('[conj]' if 'conj' in self.attr else '')
        if self.result is None:
            result_str = ''
        else:
            result_str = f'(~R{self.result.s_expr()})'
        if self.arg is None:
            arg_str = ''
        else:
            arg_str = f'(~A{self.arg.s_expr()})'
        cat_str = f'({self.root}{attr_str}{result_str}{arg_str})'
        return cat_str

    def __eq__(self, other):
        if other is None:
            return False
        return self.root == other.root and self.result == other.result and self.arg == other.arg \
               and self.attr == other.attr

    def __sub__(self, other):
        result = {}
        nb1 = dict(self.to_dense_nodeblock(concat_attr=True, binary=False))
        nb2 = dict(other.to_dense_nodeblock(concat_attr=True, binary=False))
        for a in nb1.keys() | nb2.keys():
            c1 = nb1.get(a)
            c2 = nb2.get(a)
            if c1 == c2:
                continue
            if c1 is not None:
                result[a] = c1
        return result

    def __str__(self):
        attr_str = ''.join((f'[{a}]' for a in self.attr-{'conj'})) + ('[conj]' if 'conj' in self.attr else '')
        if self.result is None:
            result_str = ''
        else:
            result_str = str(self.result)
            if self.result.has_children():
                result_str = f'({result_str})'
        if self.arg is None:
            arg_str = ''
        else:
            arg_str = str(self.arg)
            if self.arg.has_children():
                arg_str = f'({arg_str})'
        cat_str = f'{result_str}{self.root}{arg_str}'
        return f'{cat_str}{attr_str}'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.s_expr())


class AtomicCategories:
    __init__ = None
    S = 'S'
    NP = 'NP'
    N = 'N'
    PP = 'PP'
    CONJ = 'conj'
    PERIOD = '.'
    COMMA = ','
    COLON = ':'
    SEMICOLON = ';'
    LRB = 'LRB'
    RRB = 'RRB'


class Slashes:
    __init__ = None
    F = '/'
    B = '\\'


ATOMIC = {AtomicCategories.S,
          AtomicCategories.NP,
          AtomicCategories.N,
          AtomicCategories.PP}
          # AtomicCategories.CONJ,
          # AtomicCategories.PERIOD,
          # AtomicCategories.COMMA,
          # AtomicCategories.COLON,
          # AtomicCategories.SEMICOLON,
          # AtomicCategories.LRB,
          # AtomicCategories.RRB}

ATOMIC_WITH_ATTR = ATOMIC.union({'S[to]', 'NP[expl]', 'S[poss]', 'NP[nb]', 'S[asup]', 'S[bem]', 'S[inv]', 'S[dcl]',
                                 'S[frg]', 'S[pss]', 'S[qem]', 'S[q]', 'S[b]', 'S[ng]', 'S[pt]', 'S[wq]', 'S[em]',
                                 'S[intj]', 'S[as]', 'S[for]', 'S[adj]',
                                 'NP[thr]', 'N[num]'})


PUNCTUATION = {AtomicCategories.PERIOD,
               AtomicCategories.COMMA,
               AtomicCategories.COLON,
               AtomicCategories.SEMICOLON,
               AtomicCategories.LRB,
               AtomicCategories.RRB}

COORDINATION = {AtomicCategories.CONJ,
                AtomicCategories.COMMA,
                AtomicCategories.SEMICOLON}

SLASH = {Slashes.F, Slashes.B}

if __name__ == '__main__':

    from ..util.reader import CategoryReader, SCategoryReader

    cr = SCategoryReader()

    # ts = {'(N)': 0, '(NP)': 1, '(PP)': 2, '(S)': 3, '(S[dcl])': 4, '(S[b])': 5, '(,)': 6, '(.)': 7, '(S[adj])': 8, '(S[to])': 9, '(conj)': 10, '(S[ng])': 11, '(S[pss])': 12, '(N[num])': 13, '(S[pt])': 14, '(S[em])': 15, '(LQU)': 16, '(RQU)': 17, '(PR)': 18, '(:)': 19,
    #       '(S[qem])': 20, '(NP[thr])': 21, '(RRB)': 22, '(NP[expl])': 23, '(;)': 24, '(S[asup])': 25, '(LRB)': 26, '(S[q])': 27, '(S[for])': 28, '(S[wq])': 29, '(S[bem])': 30, '(S[inv])': 31, '(S[poss])': 32, '(S[frg])': 33, '(S[intj])': 34, '(NP[nb])': 35, '(S[as])': 36,
    #       '(/)': 37, '(\\)': 38}
    #
    # # c = cr.read('(LQU)', validate=False)
    # c = Category(None, validate=False)
    # print(c)
    #
    # print(c.validate())
    #
    # no = c.next_open_address()
    # print(no, '1')
    #
    # nb = c.decompose(ts, binary=True)
    #
    # print(nb)
    #
    # # a = '111'
    #
    # # print(c.next_open_address(), a)
    #
    # exit(0)

    cr = CategoryReader()

    # cat = cr.read('(((S\\NP)\\NP)\\NP)\\NP').s_expr()
    # cat = cr.read('(((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP)))/(((S\\NP)\(S\\NP))\((S\\NP)\(S\\NP)))')
    # print(cat)

    ac = AtomicCategories
    sl = Slashes

    c1 = [Category(ac.NP),
          Category(sl.F, Category(sl.B, Category('S'), Category(ac.NP)), Category(ac.NP)),
        Category(sl.F, Category(sl.B, Category(sl.B, Category(sl.B, Category(sl.B, Category('S'), Category(ac.NP)), Category(ac.NP)), Category(ac.NP)), Category(ac.NP)), Category(ac.NP)),
        Category(sl.F, Category(sl.F, Category(sl.B, Category(ac.S), Category(ac.NP)), Category(ac.NP)), Category(sl.F, Category(sl.B, Category(ac.S), Category(ac.NP)), Category(ac.NP))),
          cr.read('(((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP)))/(((S\\NP)\(S\\NP))\((S\\NP)\(S\\NP)))'),
          # cr.read('(S[dcl]/NP)', concat_attr=True)
    ]
    # print(c1[2])


    # ts = {'S\\', 'NP', '/', '\\', 'S', '(S\\)/', 'S\\NP'}
    ts = '''(NP)
            (N)
            (~A(NP))
            (/)
            (\\)
            (\\(~A(NP)))
            (~R(\\))
            (~A(N))
            (/(~A(N)))
            (~R(\\(~A(NP))))
            (/(~R(\\)))
            (/(~R(\\(~A(NP)))))
            (~R(N))
            (/(~R(N)))
            (/(~R(N))(~A(N)))
            (S)
            (~A(\\))
            (/(~A(NP)))
            (~R(S))
            (~A(\\(~A(NP))))
            (\\(~R(S)))
            (\\(~R(S))(~A(NP)))
            (/(~R(\\))(~A(NP)))
            '''.split()

    # ts = '''(/)
    #      (\\)
    #      (C)
    #      (D)
    #      (E)
    #      (\\(~R(D))(~A(E)))
    #      (/(~R(B))(~A(C)))
    #      (/(~R(B)))
    #      (/(~A(C)))'''.split()
    # ts = ts[:5] + ts[6:]

    # c1 = [Category('/', Category('\\', Category('D'), Category('E')), Category('C'))]

    # ts = ts[:6] + ts[15:16] + ts[-2:-1] + [cr.read('(((S\\NP)\\NP)\\NP)\\NP').s_expr()] + ['(/(~R(\\))(~A(NP)))'] #  + ts[20:21]
    # ts += [cr.read('(((S\\NP)\\NP)\\NP)\\NP').s_expr()] + ['(/(~R(\\))(~A(NP)))']
    # ts = ['(NP)', '(S)', '(/)', '(\\)']

    print('tagset', ts)
    print()

    # dec = cat.decompose(ts)
    # print(dec)
    # print(cat.size(), '->', len(dec))
    #
    from .util import sexpr_nodeblock_to_cat
    # _cat = sexpr_nodeblock_to_cat(dec)
    # print(_cat)
    # assert _cat == cat, (_cat, cat)
    # print()

    for c in c1:
        print(c)
        print([x for x in c.traverse()])
        dec = c.decompose(ts)
        print(dec)
        print(c.size(), '->', len(dec))
        _c = sexpr_nodeblock_to_cat(dec)
        print(_c)
        assert _c == c, (_c, c)
        print()

    # c2 = list(map(lambda x:x.get_node(11), c1))
    # print(c2)
    # c3 = list(map(lambda x: x.depth(), c1))
    # print(c3)

    # nb1 = c1[2].to_dense_nodeblock()
    #
    #
    # nb2 = {(1, '/'), (100, '\\'), (1000, 'S'), (1001, 'NP'), (10, '/'), (111, 'NP'), (1101, 'NP'), (1100, 'S'), (110, '\\'), (101, 'NP')}
    # nb3 = {(1, '/')}
    # nb4 = {(1, ac.NP)}
    # nb5 = {(1, 'NP')}
    # nb6 = {(4, 'NP'), (2, '\\'), (1, '/'), (3, 'NP'), (5, 'NP')}
    # nb7 = {(100, 'NP'), (10, '\\'), (1, '/'), (11, 'NP'), (101, 'NP')}
    #
    # print(nb1)
    # print(Category.from_dense_nodeblock(nb1))
    # print(Category.from_dense_nodeblock(nb4))
    # print(Category.from_dense_nodeblock(nb4) == Category.from_dense_nodeblock(nb5))
    # print(Category.from_dense_nodeblock(nb7))
    # print(Category.from_dense_nodeblock(nb6, binary=False))
    # try:
    #     print(Category.from_dense_nodeblock(nb2))
    #     print(Category.from_dense_nodeblock(nb3))
    # except:
    #     pass
    # else:
    #     assert False

    print('OK')
