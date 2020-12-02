'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from .category import Category, AtomicCategories as ac, Slashes as sl, PUNCTUATION as PUNCT, COORDINATION as COORD


class Combinator:
    def __init__(self, name, short_name, can_combine, combine, backward=False, ignore_attr=True, sortkey=0):
        self.name = name
        self.short_name = short_name
        self.can_combine = can_combine
        self.combine = combine
        self.backward = backward
        self.crossing = False
        self.order = 1
        self.ignore_attr = ignore_attr
        self.sortkey = sortkey  # customizable restrictiveness: more restrictive combinators will be preferred to less restrictive ones

    def execute(self):
        pass

    def __str__(self):
        return f'{"<" if self.backward else ">"}{self.short_name}'

    def __repr__(self):
        return f'Combinator<{self.short_name}, {self.backward}>'

    def __eq__(self, other):
        return self.name == other.name and self.short_name == other.short_name and self.backward == self.backward

    def __lt__(self, other):
        return self.sortkey > other.sortkey

    def __gt__(self, other):
        return self.sortkey < other.sortkey

    def __hash__(self):
        return hash(self.name) + hash(self.short_name) + hash(self.backward)


class UnaryCombinator(Combinator):
    def __init__(self, name, short_name, can_combine, combine, aux, backward=False, ignore_attr=True, sortkey=0):
        super().__init__(name, short_name, can_combine, combine, backward, ignore_attr, sortkey)
        self.aux = aux

    def execute(self, category):
        assert self.can_combine(category)
        return self.combine(category)

    def __str__(self):
        return super().__str__() + f'[{self.aux}]'

    def __repr__(self):
        return f'Combinator<{self.short_name}, {self.backward}, {self.aux}>'


class BinaryCombinator(Combinator):
    def __init__(self, name, short_name, can_combine, combine, backward=False, crossing=False, order=1,
                 ignore_attr=True, sortkey=0):
        super().__init__(name, short_name, can_combine, combine, backward, ignore_attr, sortkey)
        self.crossing = crossing
        self.order = order

    def execute(self, left, right):
        primary, auxiliary = (right, left) if self.backward else (left, right)
        assert self.can_combine(primary, auxiliary), f'{left} ({"auxiliary" if self.backward else "primary"}) ' \
            f'and {right} ({"primary" if self.backward else "auxiliary"}) cannot be combined with {self}'
        return self.combine(primary, auxiliary)

    def find_missing(self, left=None, right=None, parent=None):
        if parent is None:
            return self.execute(left, right)
        else:
            primary, auxiliary = (right, left) if self.backward else (left, right)
            pass

    def __eq__(self, other):
        return super().__eq__(other) and self.crossing == other.crossing and self.order == other.order

    def __hash__(self):
        return super().__hash__() + hash(self.crossing) + hash(self.order)

    def __str__(self):
        return super().__str__() + f'{"x" if self.crossing else ""}{self.order if self.order >= 2 else ""}'

    def __repr__(self):
        return f'Combinator<{self.short_name}, {self.backward}, {self.crossing}, {self.order}>'


class Combinators:
    __init__ = None
    T = lambda b, a: UnaryCombinator('Type-raising', 'T',
                                  (lambda p: p.equals(Category(ac.NP))),
                                  (lambda p: Category(sl.B if b else sl.F, result=a, arg=Category(sl.F if b else sl.B, result=a, arg=p))),
                                  a, b)
    Noun = lambda *_: UnaryCombinator('Noun insertion', 'N',
                                  (lambda p: p.equals(Category(ac.N))),
                                  (lambda p: Category(ac.NP)),
                                  ac.NP, False)
    A = lambda b, *_: BinaryCombinator('Application', 'A',
                                  (lambda p, a: p.is_node(1, sl.B if b else sl.F) and p.get_category(11).equals(a)),
                                  (lambda p, a: p.get_category(10)),
                                  b, False, 1)
    B = lambda b, x, o: BinaryCombinator('Composition', 'B',
                                  (lambda p, a: p.is_node(1, sl.B if b else sl.F)
                                                and a.is_node(10**(o-1), sl.F if b==x else sl.B)
                                                and p.get_category(11).equals(a.get_category(10**o))),
                                  (lambda p, a: a.replace(10**o, p.get_category(10))),
                                  b, x, o)
    S = lambda b, x, o: BinaryCombinator('Substitution', 'S',
                                  (lambda p, a: p.is_node(10, sl.B if b else sl.F)
                                                and a.is_node(10 ** (o - 1), sl.F if b == x else sl.B)
                                                and p.get_category(101).equals(a.get_category(10 ** o))
                                                and p.get_node(1) == a.get_node(1)
                                                and p.get_category(11).equals(a.get_category(11))),
                                  (lambda p, a: a.replace(10 ** o, p.get_category(100))),
                                  b, x, o)
    Conj = lambda b, *_: BinaryCombinator('Conjunction', 'Cj',
                                  (lambda p, a: p.root in COORD),  # and not a.get_category(int('1'*o)).has_attr('conj')),
                                  (lambda p, a: a.with_attr(1, 'conj')),
                                  b, False, 1, ignore_attr=False, sortkey=1)
    Coord = lambda b, *_: BinaryCombinator('Coordination', 'Cd',
                                  (lambda p, a: p.has_attr('conj')
                                                and a.equals(p.without_attr(1, 'conj'), ignore_attr=False)),
                                  (lambda p, a: a.copy()),
                                  b, False, 1, ignore_attr=False, sortkey=1)
    Punct = lambda b, *_: BinaryCombinator('Punctuation', 'P',
                                  (lambda p, a: a.root in PUNCT),
                                  (lambda p, a: p.copy()),
                                  b, False, 1)


SOFT_COMBINATORS = {
    'A': Combinators.A,
    'B': Combinators.B,
    'S': Combinators.S,
    'P': Combinators.Punct,
    'Cj': Combinators.Conj,
    'Cd': Combinators.Coord
}

UNARY_COMBINATORS = {
    # 'fTS': Combinators.T(False, Category(ac.S)),
    # 'bTS': Combinators.T(True, Category(ac.S)),
    'N': Combinators.Noun(False, None)
}

HARD_COMBINATORS = {
    # 'fTS': Combinators.T(False, ac.S),
    # 'bTS': Combinators.T(True, ac.S),
    # 'N': Combinators.Noun(False, None),
    'fA': Combinators.A(False, False, 1),
    'bA': Combinators.A(True, False, 1),
    'fB': Combinators.B(False, False, 1),
    'bB': Combinators.B(True, False, 1),
    'fBx': Combinators.B(False, True, 1),
    'bBx': Combinators.B(True, True, 1),
    'fB2': Combinators.B(False, False, 2),
    'bB2': Combinators.B(True, False, 2),
    'fBx2': Combinators.B(False, True, 2),
    'bBx2': Combinators.B(True, True, 2),
    'fS': Combinators.S(False, False, 1),
    'bS': Combinators.S(True, False, 1),
    'fSx': Combinators.S(False, True, 1),
    'bSx': Combinators.S(True, True, 1),
    'fS2': Combinators.S(False, False, 2),
    'bS2': Combinators.S(True, False, 2),
    'fSx2': Combinators.S(False, True, 2),
    'bSx2': Combinators.S(True, True, 2),

    'fP': Combinators.Punct(False, False, 1),
    'bP': Combinators.Punct(True, False, 1),
    'fCj': Combinators.Conj(False, False, 1),
    # 'bCj': Combinators.Conj(True, False, 1),
    # 'fCj2': Combinators.Conj(False, False, 2),
    # 'bCj2': Combinators.Conj(True, False, 2),
    # 'fCd': Combinators.Coord(False, False, 1),
    'bCd': Combinators.Coord(True, False, 1),
    # 'fCd2': Combinators.Coord(False, False, 2),
    # 'bCd2': Combinators.Coord(True, False, 2)
}


if __name__ == '__main__':
    c1 = Category(sl.B, ac.S, ac.NP)
    c2 = Category(sl.F, c1, ac.NP)
    c3 = ac.NP
    c4 = Category(sl.F, c3, c3)
    c5 = c1
    c6 = ac.CONJ

    fTS = HARD_COMBINATORS['fTS']
    bTS = HARD_COMBINATORS['bTS']
    fA = HARD_COMBINATORS['fA']
    bA = HARD_COMBINATORS['bA']
    fB = HARD_COMBINATORS['fB']
    fBx2 = HARD_COMBINATORS['fBx2']
    fS = HARD_COMBINATORS['fS']
    bSx = HARD_COMBINATORS['bSx']
    fCj2 = HARD_COMBINATORS['fCj2']


    c7 = fCj2.execute(c6, c5)

    print(c3, fTS, fTS.execute(c3))
    print(c3, bTS, bTS.execute(c3))
    print(fTS.execute(c3)==bTS.execute(c3))
    print(c2, c3, fA, fA.execute(c2, c3))
    print(c3, c1, bA, bA.execute(c3, c1))
    print(c4, c2, bSx, bSx.execute(c4, c2))
    print(c6, c5, fCj2, c7)
    print(c5 == c7.without_attr(int('1'*2), 'conj'))

    print(list(map(str, sorted(COMBINATORS.values()))))
