'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys

import re
import sexpdata
import ast
import json

from ..ccg.category import Category, InvalidCategoryError
from ..ccg.combinator import HARD_COMBINATORS
from ..ccg.derivation import Derivation, DerivationLeafNode, DerivationTreeNode
from ..ccg.dependency import Dependency, Dependencies

from .strings import slash, open_par, close_par, open_angle, close_angle, tree_node, leaf_node, open_bracket, close_bracket


class Reader:
    def __init__(self):
        pass


class TreeReader(Reader):
    def __init__(self):
        pass

    def read(self, s: str):
        pass


class CategoryReader(TreeReader):
    def __init__(self, print_err_msgs=False):
        super().__init__()
        self.print_err_msgs = print_err_msgs

    def make_category(self, lst, validate=True):
        if len(lst) == 3:
            return Category(lst[1], result=lst[0], arg=lst[2], validate=validate)
        elif len(lst) == 1:
            return Category(lst[0], validate=validate)
        else:
            raise ValueError

    def delimiter(self, char):
        return char in '()/\\[ '

    def read_attr(self, s):
        stack = []
        atomic = ''
        s += ' '
        i = 0
        while i < len(s):
            char = s[i]
            if close_bracket(char) and atomic:
                stack.append(atomic)
                atomic = ''
            elif open_bracket(char):
                pass
            elif self.delimiter(char):
                return stack, i+1
            else:
                atomic += char
            i += 1

    def rec_read(self, s, level=0, concat_attr=False, validate=True):
        stack = []
        atomic = ''
        i = 0
        while i < len(s):
            char = s[i]

            if self.delimiter(char) and atomic:
                stack.append(Category(atomic, validate=validate))
                atomic = ''
                continue

            if open_bracket(char):
                attr, j = self.read_attr(s[i + 1:])
                if concat_attr:
                    stack[-1] = f'{stack[-1]}[{", ".join(sorted(attr))}]'
                else:
                    stack[-1].attr = set(attr)
                i += j
                continue

            if len(stack) == 3:
                stack = [self.make_category(stack, validate=validate)]
                continue

            if slash(char):
                stack.append(char)
            elif open_par(char):
                inner, j = self.rec_read(s[i+1:], level=level+1, concat_attr=concat_attr, validate=validate)
                stack.append(inner)
                i += j+1
                continue
            elif close_par(char):
                if len(stack) > 1:
                    if validate:
                        raise InvalidCategoryError(f'Malformed category {s}')
                    else:
                        print(f'warning: malformed category: {s}; truncating...', file=sys.stderr)
                return stack[0], i+1
            elif char.isspace():
                pass
            else:
                atomic += char
            i += 1
        if len(stack) > 1:
            if validate:
                raise InvalidCategoryError(f'Malformed category {s}')
            else:
                print(f'warning: malformed category: {s}; truncating...', file=sys.stderr)
        return stack[0], i

    def read(self, s: str, concat_attr=False, validate=True) -> Category:
        try:
            result = self.rec_read(f'({s})', concat_attr=concat_attr, validate=validate)[0]
            if not isinstance(result, Category):
                result = Category(result, validate=validate)
        except InvalidCategoryError as e:
            if self.print_err_msgs:
                return Category(f'-ERR-{s}-', validate=False)
            else:
                raise
        cat = result
        while cat.has_children():
            cat = cat.arg
            if cat.has_attr('conj'):
                cat.remove_attr('conj')
                result.add_attr('conj')
                break
        return result


class SCategoryReader(TreeReader):
    def __init__(self):
        super().__init__()
        self.child_type_map = {'~R': 'result', '~A': 'arg'}

    def rec_read(self, s_expr, validate=True):
        children = {}
        attr = set()
        if isinstance(s_expr, list):
            root = s_expr[0].tosexp().replace('\\\\', '\\').replace('\\.', '.').replace('\\,', ',').replace('\\;', ';')
            if len(s_expr) >= 2:
                if isinstance(s_expr[1], sexpdata.Bracket):
                    attr = s_expr[1].value()
                    attr = set([at.tosexp() for at in attr])
                else:
                    for _child in s_expr[1:]:
                        child_type = _child[0].tosexp()
                        child = self.rec_read(_child[1], validate=validate)
                        children[self.child_type_map[child_type]] = child
        else:
            root = s_expr.tosexp().replace('\\\\', '\\').replace('\\.', '.').replace('\\,', ',').replace('\\;', ';')
        return Category(root, attr=attr, validate=validate, **children)

    def read(self, s: str, validate=True) -> Category:
        result = sexpdata.loads(s.replace('\\', '\\\\'), nil=None, true=None, false=None, line_comment='-LINE_COMMENT-')
        result = self.rec_read(result, validate=validate)
        return result


class CombinatorReader(TreeReader):
    pass


class DerivationReader(TreeReader):
    def __init__(self, print_err_msgs=False):
        super().__init__()
        self.derivation = Derivation()
        self.cr = CategoryReader(print_err_msgs)


class AUTODerivationReader(DerivationReader):
    def read_node(self, s, validate=True):
        i = 0
        j = s.index('>')
        while i < len(s):
            char = s[i]
            if open_angle(char) or char.isspace():
                i += 1
            elif tree_node(char):
                cat, hd, n_children = s[i+1:j].split()
                return (lambda children: DerivationTreeNode(self.derivation, self.cr.read(cat, validate=validate), int(hd), children)), \
                       int(n_children), j
            elif leaf_node(char):
                try:
                    cat1, pos1, pos2, word, cat2 = s[i+1:j].split()
                except ValueError as e:
                    raise ValueError(e.args, i, j, s[i+i:j])
                return (lambda children: DerivationLeafNode(self.derivation, self.cr.read(cat1, validate=validate),
                                                    pos1, pos2, word, None)), 0, j

    def rec_read(self, s: str, validate=True):
        i = 0
        while i < len(s):
            char = s[i]
            if open_par(char) or close_par(char) or char.isspace():
                i += 1
            elif open_angle(char):
                _node, n_children, j = self.read_node(s[i:], validate=validate)
                children = []
                i += j+1
                for _ in range(n_children):
                    child, k = self.rec_read(s[i:], validate=validate)
                    i += k
                    children.append(child)
                return _node(children), i
            else:
                raise ValueError(s[i:])
        return None

    def read(self, s: str, validate=True) -> Derivation:
        if s is None or not s.strip():
            return None
        result, _ = self.rec_read(s.replace('<UNKNOWN>', '-UNKNOWN-').replace('<PADDING>', '-PADDING-'), validate=validate)
        self.derivation.root = result
        return self.derivation


class ASTDerivationReader(DerivationReader):
    def __init__(self, print_err_msgs=False):
        super(ASTDerivationReader, self).__init__()
        self.kv_list_pattern = re.compile("\\[((\\w+):(\'([^\']|\\\\\')+\'|\\[[^\\]]+]|\\d+)(, )?)+]")
        self.kv_pattern = re.compile("(\\w+):(\'([^\']|\\\\\')+\'|\\[[^\\]]+]|\\d+)(, *|])")
        self.cat_pattern = re.compile("([( ])([a-zA-Z/\\\\:()]+|[,.;])(,)")
        self.attr_pattern = re.compile(":([a-zA-Z]+)")

    def escape(self, s):
        lines = []
        for line in s:
            line = line.strip()
            cats = self.cat_pattern.findall(line)
            for pre, cat, suf in cats:
                esc_cat = cat.replace('\\', '\\\\')
                line = line.replace(f'{pre}{cat}{suf}', f'{pre}\'{esc_cat}\'{suf}')
            mo = self.kv_list_pattern.search(line)
            if mo:
                kv_list_str = mo.group()
                kv_list = self.kv_pattern.findall(kv_list_str)
                d = {}
                for k, v, _, _ in kv_list:
                    d[k] = ast.literal_eval(v)
                bom, eom = mo.span()
                line = f'{line[:bom]}{d}{line[eom:]}'
            lines.append(line)
        return ' '.join(lines)

    def unescape_attr_(self, cat: Category):
        if self.attr_pattern.search(cat.root):
            cat.root, attr = cat.root.split(':')
            if attr != 'X':
                cat.add_attr(attr)
        if cat.root != 'conj':
            cat.root = cat.root.upper()
        if cat.result is not None:
            self.unescape_attr_(cat.result)
        if cat.arg is not None:
            self.unescape_attr_(cat.arg)
        return cat

    def rec_read(self, node, validate=True):
        '''

        :param ast:
        :param validate:
        :return:

        '''
        try:
            if isinstance(node, ast.Call):  # combinator (or top-level)
                name = node.func.id
                if name == 'ccg':  # top-level
                    return self.rec_read(node.args[1], validate=validate)
                else:
                    arg0, *args = [self.rec_read(arg, validate=validate) for arg in node.args]
                    cat = self.cr.read(arg0, validate=validate)
                    self.unescape_attr_(cat)
                    if name == 't':
                        word, attrs = args
                        word = word
                        pos = attrs.get('pos')
                        return DerivationLeafNode(self.derivation, cat, pos, pos, word, cat)
                    elif name == 'lx':  # type-raising
                        arg1, *args = args
                        cat2 = self.cr.read(arg1, validate=validate)
                        self.unescape_attr_(cat2)
                        assert cat2.equals(args[0].category1)
                        dtn = DerivationTreeNode(self.derivation, cat, 0, args)
                        dtn.xcombinator = name
                        return dtn
                    else:  # combinator
                        dtn = DerivationTreeNode(self.derivation, cat, 0, args)
                        dtn.xcombinator = name
                        return dtn
            if isinstance(node, ast.Str):  # word or annotation
                return ast.literal_eval(node).encode('utf-8', errors='replace').decode('utf-8', errors='surrogateescape')
            if isinstance(node, ast.Dict):  # attribute-value map
                return ast.literal_eval(node)
            if isinstance(node, ast.Name):
                raise TypeError(node, node.id)
            raise TypeError(node)

        except ValueError as e:
            import pdb
            pdb.set_trace()
            raise ArithmeticError(e, node)

    def read(self, lines: list, validate=True, line_comment=':-'):
        '''

        :param s:
        :param validate:
        :param line_comment: Default is ':-' for PMB. Has to be set to '%' for TUTBank.
        :return:
        '''
        string = self.escape(lines)
        try:
            result = self.rec_read(ast.parse(string, mode='eval').body, validate=validate)
        except ValueError as e:
            import pdb
            pdb.set_trace()
            raise ArithmeticError(e, string)
        self.derivation.root = result
        return self.derivation


class StaggedDerivationReader(DerivationReader):
    def read(self, s: str, validate=True) -> Derivation:
        if s is None or not s.strip():
            return None
        lexical = []
        for token in s.strip().split():
            word, pos, cat = token.split('|')
            tag = self.cr.read(cat, validate=validate)
            lexical.append(DerivationLeafNode(self.derivation, tag, pos, pos, word, tag))
        self.derivation.root = DerivationTreeNode(self.derivation, None, 0, lexical)
        return self.derivation


class PlainTextDerivationReader(DerivationReader):
    def read(self, s: str, **kwargs) -> Derivation:
        del kwargs
        if s is None or not s.strip():
            return None
        lexical = []
        for word in s.strip().split():
            dummy_pos = 'X'
            dummy_cat = Category('X')
            lexical.append(DerivationLeafNode(self.derivation, dummy_cat, dummy_pos, dummy_pos, word, dummy_cat))
        self.derivation.root = DerivationTreeNode(self.derivation, None, 0, lexical)
        return self.derivation


class DependencyReader(TreeReader):
    def __init__(self):
        self.cr = CategoryReader()

    def read(self, s: str, concat_attr=False, validate=True) -> Dependency:
        dep, head, head_cat, arg, *_ = s.strip().split()
        return Dependency(int(dep),
                          int(head),
                          self.cr.read(head_cat, concat_attr=concat_attr, validate=validate),
                          int(arg))


class OutputDependencyReader(TreeReader):
    def __init__(self):
        self.cr = CategoryReader()
        self.remove_pattern = re.compile('[<{].[}>]|\[X\]')

    def read(self, s: str, concat_attr=False, validate=True) -> Dependency:
        head, head_cat, arg, dep, *_ = s.strip().split()
        head = int(head.split('_')[1]) - 1
        dep = int(dep.split('_')[1]) - 1
        return Dependency(dep,
                          head,
                          self.cr.read(re.sub(self.remove_pattern, '', head_cat), concat_attr=concat_attr, validate=validate),
                          int(arg))


class FileReader(Reader):
    def __init__(self, filename):
        self.filename = filename
        self.f = open(self.filename, errors='ignore', encoding='utf-8')
        self.line = -1

    def readline(self, strip=True):
        self.line += 1
        return self.f.readline()

    def close(self):
        self.f.close()


class DerivationsReader(FileReader):
    def __init__(self, filename, validate=True, print_err_msgs=False):
        super().__init__(filename)
        self.dr = DerivationReader
        self.i = -1
        self.print_err_msgs = print_err_msgs
        self.validate = validate

    def next(self):
        self.i += 1
        line = self.readline()
        if not line:
            raise StopIteration
        return {'ID': f'{self.filename}.{self.i}', 'DERIVATION': self.dr(print_err_msgs=self.print_err_msgs
                                                                         ).read(line, validate=self.validate)}

    def read_all(self):
        result = []
        while True:
            try:
                result.append(self.next())
            except StopIteration:
                break
        return result

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class AUTODerivationsReader(DerivationsReader):
    def __init__(self, filename, validate=True, print_err_msgs=False):
        super().__init__(filename, validate=validate, print_err_msgs=print_err_msgs)
        self.dr = AUTODerivationReader

    def next(self):
        line = self.readline()
        if not line:
            raise StopIteration
        kvs = dict(kv.split('=') for kv in line.split())
        result = super(AUTODerivationsReader, self).next()
        result.update(kvs)
        return result


class StaggedDerivationsReader(DerivationsReader):
    def __init__(self, filename, validate=True, print_err_msgs=False):
        super().__init__(filename, validate=validate, print_err_msgs=print_err_msgs)
        self.dr = StaggedDerivationReader
        start = 0
        self.f = open(self.filename)
        line = self.readline()
        while line.startswith('#') or not line:
            start = self.f.tell() + 1
            line = self.readline()
        self.close()
        self.f = open(self.filename)
        self.f.seek(start)


class PlainTextDerivationsReader(DerivationsReader):
    def __init__(self, filename, **kwargs):
        del kwargs
        super().__init__(filename)
        self.dr = PlainTextDerivationReader
        start = 0
        self.f = open(self.filename)
        line = self.readline()
        while line.startswith('#') or not line:
            start = self.f.tell() + 1
            line = self.readline()
        self.close()
        self.f = open(self.filename)
        self.f.seek(start)


class ASTDerivationsReader(DerivationsReader):
    def __init__(self, filename, validate=True, print_err_msgs=False):
        super().__init__(filename, validate=validate, print_err_msgs=print_err_msgs)
        self.dr = ASTDerivationReader
        start = 0
        self.f = open(self.filename, errors='ignore', encoding='utf-8')
        line = self.readline()
        while line.startswith(':-') or not line:
            start = self.f.tell() + 1
            line = self.readline()
        self.close()
        self.f = open(self.filename, errors='ignore', encoding='utf-8')
        self.f.seek(start)

    def next(self, validate=True):
        self.i += 1
        try:
            line = self.readline().strip()
            if not line:
                raise StopIteration
            lines = [line]
            while not line.endswith('.'):
                line = self.readline().strip()
                lines.append(line.strip('.'))

        except UnicodeDecodeError as e:
            raise Exception(e, self.i, self.f.tell(), line, lines)
        return {'ID': f'{self.filename}.{self.i}', 'DERIVATION': self.dr(self.print_err_msgs).read(lines, validate=self.validate)}


class DependenciesReader(FileReader):
    def __init__(self, filename):
        super().__init__(filename)
        self.start_pattern = re.compile('<s id="([^"]+)"> (\\d+)')
        self.enc_pattern = re.compile('<\\\\s>')
        self.f = open(self.filename)

    def next(self, validate=True):
        line = self.f.readline()
        if not line:
            raise StopIteration
        mo = self.start_pattern.match(line)
        if mo:
            result = {'ID': mo.group(1)}
        else:
            raise StopIteration
        dependencies = []
        line = self.f.readline()
        while not self.enc_pattern.match(line):
            dependencies.append(DependencyReader().read(line, validate=validate))
            line = self.f.readline()
            if not line:
                raise StopIteration
        result['DEPENDENCIES'] = dependencies
        return result

    def read_all(self):
        result = []
        while True:
            try:
                result.append(self.next())
            except StopIteration:
                break
        return result

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class OutputDependenciesReader(FileReader):
    def __init__(self, filename):
        super().__init__(filename)
        self.dr = OutputDependencyReader()
        self.end_pattern = re.compile('<c> .*')
        self.f = open(self.filename)

    def next(self, validate=True):
        line = self.f.readline()
        while line.startswith('# ') or not line.strip():
            line = self.f.readline()
        result = {}
        dependencies = []
        while not self.end_pattern.match(line):
            dependencies.append(self.dr.read(line, validate=validate))
            line = self.f.readline()
            if not line.strip():
                raise StopIteration
        result['DEPENDENCIES'] = dependencies
        return result

    def read_all(self):
        result = []
        while True:
            try:
                result.append(self.next())
            except StopIteration:
                break
        return result

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self



if __name__ == "__main__":
    from ccg.util import argparse
    args = argparse.main()

    for fn in args.training_files:
        print(fn)
        ds = ASTDerivationsReader(fn)
        for d in ds:
            # print(ds[sentence - 1]['ID'])
            print(d['ID'], d['DERIVATION'])
            # print(ds[sentence - 1]['DERIVATION'].pretty_print())

    print('\nok.')

    exit(0)

    ds = ASTDerivationReader()

    s = '''
ccg(1,
 ba(s:dcl,
  t(np, 'I', [from:0, to:1, pos:'PRP', lemma:'speaker', sem:'PRO', wordnet:'O']),
  fa(s:dcl\\np,
   t((s:dcl\\np)/(s:pss\\np), '\\'m', [from:1, to:3, pos:'VBP', lemma:'be', sem:'NOW', wordnet:'O']),
   fa(s:pss\\np,
    fa((s:pss\\np)/pp,
     t(((s:pss\\np)/pp)/pr, 'fed', [from:4, to:7, pos:'VBN', lemma:'fed~up', sem:'IST', wordnet:'fed_up.a.01', verbnet:['Experiencer']]),
     t(pr, 'up', [from:8, to:10, pos:'RP', lemma:'up', sem:'REL', wordnet:'O'])),
    rp(pp,
     fa(pp,
      t(pp/np, 'with', [from:11, to:15, pos:'IN', lemma:'with', sem:'REL', wordnet:'O', verbnet:['Stimulus']]),
      t(np, 'him', [from:16, to:19, pos:'PRP', lemma:'male', sem:'PRO', wordnet:'male.n.02'])),
     t(., '.', [from:19, to:20, pos:'.', lemma:'.', sem:'NIL', wordnet:'O']))))))
    '''

    _s = ds.escape(s)
    ast.parse(_s, mode='eval')

    d = ds.read(s, validate=False)

    print(d)
    print(d.pretty_print())

    exit(0)

    ds = StaggedDerivationsReader(filename).read_all()

    print(ds[sentence - 1]['ID'])
    print(ds[sentence - 1]['DERIVATION'])

    exit(0)

    from ..util import argparse
    args = argparse.main()

    for filepath in args.training_files:
        deps_path = str(filepath).replace('AUTO', 'PARG').replace('auto', 'parg')
        for deriv, deps in zip(DerivationsReader(filepath), DependenciesReader(deps_path)):
            assert deriv['ID'] == deps['ID'], (deriv["ID"], deps["ID"])
            deriv = deriv['DERIVATION']
            deriv.set_dependencies(deps['DEPENDENCIES'])
            print(deriv)
            for _, dep in sorted(deriv.dependencies.items()):
                print(dep)
            print('----------')
            break
        break

    exit(0)

    # l1 = [
    #     'NP3',
    #     'N2',
    #     'NP3/N2',
    #     'NP5\\NP3',
    #     '(S7\\NP8)/NP9',
    # #     '(((S\\NP)\\NP)/((S/NP)\\NP))/(NP/NP)',
    # #     '((S\\NP1)/(NP2/NP3))\\NP4',
    # #     'S\\NP1/(NP2/NP3)\\NP4',
    # #     's/s/s/(s/s/s/s)/s/s/s'
    #     'S10/(S7\\NP8)'
    # ]
    #
    # r1 = CategoryReader()
    #
    # print(l1)
    # l2 = list(map(r1.read, l1))
    # print(l2)

    # np, n, np_o_n, np_u_np, s_u_np_o_np, s_o_s_u_np = l2
    #
    # fTS = COMBINATORS['fTS']
    # fA = COMBINATORS['fA']
    # bA = COMBINATORS['bA']
    # fB = COMBINATORS['fB']
    # bBx = COMBINATORS['bBx']
    # fBx2 = COMBINATORS['fBx2']

    # print(np, fTS, fTS.execute(np))
    # print(np_o_n, n, fA, fA.execute(np_o_n, n))
    # print(np, np_u_np, bA, bA.execute(np, np_u_np))
    # print(s_o_s_u_np, s_u_np_o_np, fB, fB.execute(s_o_s_u_np, s_u_np_o_np))
    # print(np_o_n, np_u_np, bBx, bBx.execute(np_o_n, np_u_np))
    # print(fBx2)

    # d1 = open('C:/Users/Jakob/PycharmProjects/CCG/data/AUTO/00/wsj_0001.auto').readlines()[1]
    # d1 = open('C:/Users/Jakob/PycharmProjects/CCG/data/AUTO/02/wsj_0201.auto').readlines()[5]
    # l = open('C:/Users/Jakob/PycharmProjects/CCG/data/test.txt').readlines()
    # print(l[1])
    # print(l[3])

    cr = SCategoryReader()

    s = '(/(~R(\\))(~A(NP[nb])))'

    print(s)
    cat = cr.read(s, validate=False)
    print(cat)
    print(cat.s_expr())

    exit(0)

    # filename = sys.argv[1]
    # sentence = int(sys.argv[2])
    #
    # ds = DerivationsReader(filename).read_all()
    #
    # print(ds[sentence-1]['ID'])
    # print(ds[sentence-1]['DERIVATION'])

    # ds1 = DerivationsReader('C:/Users/Jakob/PycharmProjects/CCG/data/test.txt').read_all()
    # ds2 = DerivationsReader('C:/Users/Jakob/PycharmProjects/CCG/data/test.txt').read_all()
    #
    # d1 = ds1[0]
    # d2 = ds2[0]
    # d3 = ds1[2]
    #
    # print(d1['DERIVATION'])
    # print(d2)
    # print(d3)
    # print(d1 == d2)
    # print(d1 == d3)
    # print(d3['DERIVATION'].get_node(0, 1))
    # print(d3['DERIVATION'].get_node(1, 2))
    # print(d3['DERIVATION'].get_node(2, 3))
    # print(d3['DERIVATION'].get_node(0, 2))
    # print(d3['DERIVATION'].get_node(0, 2, 1))
