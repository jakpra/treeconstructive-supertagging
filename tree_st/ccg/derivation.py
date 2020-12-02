'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
from collections import Counter, defaultdict

from .category import Category, InvalidCategoryError
from .combinator import HARD_COMBINATORS, Combinators


class DerivationNode:
    def __init__(self, derivation, category1: Category, category2: Category, head, args=[], attr={}):
        self.derivation = derivation
        self.category1 = category1
        self.category2 = category2
        self.head = head
        self.args = args
        self.attr = attr

    def start_position(self):
        pass

    def end_position(self):
        pass

    @classmethod
    def paste(cls, a, b):
        a = a.split('\n')
        b = b.split('\n')
        h = max(len(a), len(b))
        w = len(a[0])
        v = len(b[0])
        a.extend([' ' * w] * (h - len(a)))
        b.extend([' ' * v] * (h - len(b)))
        return '\n'.join(map(' '.join, zip(a, b)))

    @classmethod
    def paste_list(cls, lst):
        if len(lst) == 0:
            return ''
        if len(lst) == 1:
            return lst[0]
        return cls.paste(lst[0], cls.paste_list(lst[1:]))

    def equals(self, other, ignore_attr=False):
        return (self.start_position(), self.end_position()) == (other.start_position(), other.end_position()) \
               and self.category1.equals(other.category1, ignore_attr=ignore_attr)

    def __eq__(self, other):
        return self.equals(other, ignore_attr=True)


class DerivationLeafNode(DerivationNode):
    def __init__(self, derivation, category1, pos1, pos2, word, category2):
        super().__init__(derivation, category1, category2, self, [], {})
        self.position = len(derivation.sentence)
        self.word = word
        self.pos1 = pos1
        self.pos2 = pos2
        self.combinator = None
        derivation.sentence.append(word)

    def start_position(self):
        return self.position

    def end_position(self):
        return self.position + 1

    def get_lexical(self, ignore_attr=False):
        return [self]

    def pretty_print(self):
        cat_str = str(self.category1)
        n = max(len(self.word), len(cat_str))
        return '\n'.join([f'{self.word}{" "*(n-len(self.word))}', '_'*n, f'{cat_str}{" "*(n-len(cat_str))}'])

    def __str__(self):
        return f'(<L {self.category1} {self.pos1} {self.pos2} {self.word} {self.category2}>) '


class DerivationTreeNode(DerivationNode):
    def __init__(self, derivation, category, head_index, args):
        super().__init__(derivation, category, None, head_index, args, {})
        combinators = []
        for f in HARD_COMBINATORS.values():
            if self.check_combinator(f):
                combinators.append(f)
        if len(self.args) == 1 and self.category1 is not None:  # special case for arbitrary type raising
            for b in (0, 1):
                t = Combinators.T(b, self.category1.get_category(10))
                if self.check_combinator(t):
                    combinators.append(t)
            n = Combinators.Noun()
            if self.check_combinator(n):
                combinators.append(n)
        if len(combinators) > 1:
            combinators.sort()
            # print(f'warning: >= 2 combinators applicable for {" ".join(map(lambda x:str(x.category1), self.args))} => {self.category1}: {combinators}', file=sys.stderr)
        self.combinator = combinators[0] if combinators else None

    def start_position(self):
        return self.args[0].start_position()

    def end_position(self):
        return self.args[-1].end_position()

    def check_combinator(self, c):
        return Derivation.check_combinator(c, self.category1, [arg.category1 for arg in self.args])

    def get_lexical(self, ignore_attr=False):
        result = []
        for arg in self.args:
            result.extend(arg.get_lexical(ignore_attr=ignore_attr))
        return result

    def pretty_print(self):
        args_str = super().paste_list(list(map(lambda a: a.pretty_print(), self.args)))
        cat_str = str(self.category1)
        comb_str = str(self.combinator) if self.combinator else ''
        m = len(args_str.split('\n')[0])
        n = max(m, len(cat_str), len(comb_str))
        return '\n'.join([f'{args_str}{" "*(n-m)}', f'{"_"*(n-len(comb_str))}{comb_str}', f'{cat_str}{" "*(n-len(cat_str))}'])

    def equals(self, other, ignore_attr=False):
        return (self.start_position(), self.end_position()) == (other.start_position(), other.end_position()) \
               and self.category1.equals(other.category1, ignore_attr=ignore_attr) \
               and tuple(self.args) == tuple(other.args)

    def __str__(self):
        return f'(<T {self.category1} {self.head} {len(self.args)}> {"".join(map(str, self.args))}) '

    def __eq__(self, other):
        return self.equals(other, ignore_attr=True)


class Derivation:
    def __init__(self):
        self.sentence = []
        self.root: DerivationTreeNode = None
        self.dependencies = defaultdict(dict)

    @staticmethod
    def check_combinator(comb, parent_cat, children_cats):
        try:
            return parent_cat.equals(comb.execute(*children_cats), ignore_attr=comb.ignore_attr)
        except (TypeError, AssertionError, AttributeError, InvalidCategoryError):
            return False

    def set_dependencies(self, dependencies):
        for dep in dependencies:
            dep.set_derivation(self)

    def _count_combinators(self, node: DerivationNode):
        result = Counter()
        if node.category1 is None or not isinstance(node, DerivationTreeNode):
            return result
        if node.combinator:
            result[str(node.combinator)] += 1
        else:
            result[f'{" ".join([str(arg.category1) for arg in node.args])} => {node.category1}'] += 1
        for arg in node.args:
            result.update(self._count_combinators(arg))
        return result

    def count_combinators(self):
        '''Which combinators are used?'''
        return self._count_combinators(self.root)

    def _count_words(self, node: DerivationNode, lower=False):
        if isinstance(node, DerivationLeafNode):
            result = Counter({(node.word.lower() if lower else node.word): 1})
        else:
            result = Counter()
        for arg in node.args:
            result.update(self._count_words(arg, lower))
        return result

    def count_words(self, lower=False):
        return self._count_words(self.root, lower)

    def _count_usages(self, node: DerivationNode, lower=False):
        if isinstance(node, DerivationLeafNode):
            result = Counter({((node.word.lower() if lower else node.word), node.category1): 1})
        else:
            result = Counter()
        for arg in node.args:
            result.update(self._count_usages(arg, lower))
        return result

    def count_usages(self, lower=False):
        return self._count_usages(self.root, lower)

    def _get_node(self, node, i, j, u=0, unaries=[]) -> DerivationNode:
        if node.start_position() == i and node.end_position() == j:
            unaries = [node] + unaries
            if len(node.args) == 1:
                return self._get_node(node.args[0], i, j, u=u, unaries=unaries)
            else:
                return unaries
        else:
            args = list(filter(lambda n: n.start_position() <= i and n.end_position() >= j, node.args))
            assert len(args) == 1
            return self._get_node(args[0], i, j, u=u, unaries=unaries)

    def get_node(self, i, j, u=0) -> DerivationNode:
        return self._get_node(self.root, i, j)[u]

    def _categories(self, node):
        i, j = node.start_position(), node.end_position()
        result = []
        first, *unaries = self._get_node(node, i, j)
        arg_cats = []
        for arg in first.args:
            arg_cats.append(arg.category1)
            result.extend(self._categories(arg))
        result.append((((i, j), 0), (arg_cats or [None], first.category1), first.combinator))
        last = first.category1
        for u, unary in enumerate(unaries, start=1):
            result.append((((i, j), u), ([last], unary.category1), unary.combinator))
            last = unary.category1
        return result

    def categories(self, concat_attr=False):
        return self._categories(self.root)

    def _get_unary(self, node, ignore_attr=False):
        i, j = node.start_position(), node.end_position()
        result = []
        unaries = self._get_node(node, i, j)
        last = None
        for u, unary in enumerate(unaries):
            cat = unary.category1
            if ignore_attr:
                cat = cat.without_attr()
            result.append((((i, j), u), (last, cat)))
            last = cat
        while len(node.args) == 1:
            node = node.args[0]
        for arg in node.args:
            result.extend(self._get_unary(arg, ignore_attr=ignore_attr))
        return result

    def get_unary(self):
        return self._get_unary(self.root)

    def _get_categories(self, node: DerivationNode, u=0, ignore_attr=False):
        unaries = self._get_unary(node, ignore_attr=ignore_attr)
        result = {}
        for (span, _u), (_, cat) in unaries:
            if u == -1 or _u <= u:
                if u == 0:
                    result[span] = cat
                else:
                    result[span, _u] = cat
        return result

    def get_categories(self, u=0, ignore_attr=False):
        '''{span: category} dict'''
        return self._get_categories(self.root, u=u, ignore_attr=ignore_attr)

    def _count_categories(self, node: DerivationNode, u=0, ignore_attr=False):
        return Counter([cat for _, cat in self._get_categories(node, u=u, ignore_attr=ignore_attr).items()
                        if cat is not None])

    def count_categories(self, u=0, ignore_attr=False):
        '''Which categories are used?'''
        return self._count_categories(self.root, u=u, ignore_attr=ignore_attr)

    def get_unique_cats_by_pos(self, pos=2, ignore_attr=False):
        result = defaultdict(Counter)
        for dln in self.get_lexical(ignore_attr=ignore_attr):
            result[dln.pos1 if pos==1 else dln.pos2][dln.category1] += 1
        return result

    def _count_atomic_categories(self, node: DerivationNode, u=0, concat_attr=False):
        return sum([cat.count_atomic_categories(concat_attr) for _, cat in self._get_categories(node, u=u).items()
                    if cat is not None],
                   Counter())

    def count_atomic_categories(self, u=0, concat_attr=False):
        '''Which atomic categories are used?'''
        return self._count_atomic_categories(self.root, u=u, concat_attr=concat_attr)

    def _count_category_shapes(self, node: DerivationNode, u=0):
        return Counter([cat.get_shape() for _, cat in self._get_categories(node, u=u).items() if cat is not None])

    def count_category_shapes(self, u=0):
        '''Which category shapes occur? (The shape of (S\\NP)/NP is (X\\X)/X.)'''
        return self._count_category_shapes(self.root, u=u)

    def _count_depths(self, node: DerivationNode, u=0):
        return Counter([cat.depth() for _, cat in self._get_categories(node, u=u).items() if cat is not None])

    def count_depths(self, u=0):
        '''How deep are categories?'''
        return self._count_depths(self.root, u=u)

    def _count_sizes(self, node: DerivationNode, u=0):
        return Counter([cat.size() for _, cat in self._get_categories(node, u=u).items() if cat is not None])

    def count_sizes(self, u=0):
        '''How deep are categories?'''
        return self._count_sizes(self.root, u=u)

    def _count_args(self, node: DerivationNode, u=0):
        return Counter([cat.nargs() for _, cat in self._get_categories(node, u=u).items() if cat is not None])

    def count_args(self, u=0):
        '''How deep are categories?'''
        return self._count_args(self.root, u=u)

    def _count_slashes(self, node: DerivationNode, u=0):
        return sum([cat.count_slashes() for _, cat in self._get_categories(node, u=u).items() if cat is not None],
                   Counter())

    def count_slashes(self, u=0):
        '''Which slashes are used within categories?'''
        return self._count_slashes(self.root, u=u)

    def _count_tl_slashes(self, node: DerivationNode, u=0):
        return Counter([cat.root for _, cat in self._get_categories(node, u=u).items()
                        if cat is not None and cat.has_children()])

    def count_tl_slashes(self, u=0):
        '''Which slashes are used at the top level of categories?'''
        return self._count_tl_slashes(self.root, u=u)

    def _count_addresses(self, node: DerivationNode, labels=None, u=0):
        return sum([cat.count_addresses(labels) for _, cat in self._get_categories(node, u=u).items()
                    if cat is not None],
                   Counter())

    def count_addresses(self, labels=None, u=0):
        '''Which addresses of categories are used?'''
        return self._count_addresses(self.root, labels=labels, u=u)

    def count_sentence_unary(self):
        return Counter([len([u for (_, u), _ in self.get_unary() if u > 0])])

    def count_multiple_unary(self):
        '''How many unary combinators in a row?'''
        multiple = Counter([u for (_, u), _ in self.get_unary() if u > 0])
        return Counter(multiple.values())

    def _count_unary_levels(self, node):
        result = Counter()
        if len(node.args) == 1:
            if isinstance(node.args[0], DerivationTreeNode):
                result['non-lex'] += 1
            else:
                result['lex'] += 1
        for arg in node.args:
            result.update(self._count_unary_levels(arg))
        return result

    def count_unary_levels(self):
        return self._count_unary_levels(self.root)

    @staticmethod
    def from_lexical(tags, gold_lex):
        d = Derivation()
        lexical = []
        for tag, gold_node in zip(tags, gold_lex):
            lexical.append(DerivationLeafNode(d, tag, gold_node.pos1, gold_node.pos2, gold_node.word, tag))
        d.root = DerivationTreeNode(d, None, 0, lexical)
        return d

    def get_lexical(self, ignore_attr=False):
        return self.root.get_lexical(ignore_attr=ignore_attr)

    def lexical_deriv(self):
        lex = self.get_lexical()
        tags = [l.category1 for l in lex]
        return Derivation.from_lexical(tags, lex)

    def pretty_print(self):
        return f'{self.root.pretty_print()}\n'

    def print_stagged(self):
        return ' '.join(['|'.join([l.word, str(l.category1)]) for l in self.get_lexical()])

    def __str__(self):
        return str(self.root)

    def __eq__(self, other):
        if self.root is None:
            return other.root is None
        if other.root is None:
            return self.root is None
        return self.root == other.root


def rec_traverse(node: DerivationNode):
    if isinstance(node, DerivationLeafNode):
        print(f'Word at position {node.position}: {node.word} | category: {node.category1}')
        # do something with word-level nodes
        ...
    else:
        print(f'Phrase spanning positions [{node.start_position()}, {node.end_position()}): '
              f'{" ".join([lex.word for lex in node.get_lexical()])} | category: {node.category1} | '
              f'combinator: {node.combinator}')
        # do something with phrasal nodes
        ...
        for arg in node.args:
            # these are the words/phrases that combine to form the current node
            rec_traverse(arg)

