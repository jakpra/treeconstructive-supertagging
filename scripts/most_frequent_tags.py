'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys, argparse
from collections import Counter
import json

from tree_st.util import argparse
from tree_st.util.reader import ASTDerivationsReader, AUTODerivationsReader, StaggedDerivationsReader
from tree_st.util.statistics import print_counter_stats
from tree_st.ccg.category import Slashes as sl


def main(args):
    out = open(args.out, 'w', newline='\n') if args.out else sys.stdout

    if args.training_format == 'ast':
        dr = ASTDerivationsReader
    elif args.training_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    # combinators = Counter()
    atomic_categories = Counter()
    categories = Counter()
    # category_shapes = Counter()
    # depths = Counter()
    # slashes = Counter()
    # tl_slashes = Counter()
    # addresses = Counter()
    # unary = Counter()
    # unary_levels = Counter()
    # sentence_unary = Counter()
    for filepath in args.training_files:
        ds = dr(filepath, validate=False)
        for d in ds:
            deriv = d['DERIVATION']
            print(d['ID'], file=sys.stderr)
            if args.derivation:
                print(d['ID'], file=out)
                print(deriv, file=out)
            lex = deriv.get_lexical(ignore_attr=False)
            # combinators.update(deriv.count_combinators())
            for dln in lex:
                atomic_categories.update(dln.category1.count_atomic_categories(concat_attr=True))
                categories[dln.category1] += 1

        ds.close()

    # print('Category depths', '---------------', sep='\n', file=out)
    # print_counter_stats(depths, 'depth', None, file=out)
    #
    # print(file=out)
    # print('Categories', '-----------------', sep='\n', file=out)
    # print_counter_stats(categories, 'category', None, file=out)
    #
    # print(file=out)
    # print('Category shapes', '-----------------', sep='\n', file=out)
    # print_counter_stats(category_shapes, 'shape', None, file=out)
    #
    # print(file=out)
    # print('Top-level slashes', '-----------', sep='\n', file=out)
    # print_counter_stats(tl_slashes, 'slash', None, file=out)
    #
    # print(file=out)
    # print('Slash addresses', '-----------', sep='\n', file=out)
    # print_counter_stats(addresses, 'address', None, file=out)
    #
    # print(file=out)
    # print('Slashes', '-----------', sep='\n', file=out)
    # print_counter_stats(slashes, 'slash', None, file=out)
    #
    # print(file=out)
    # print('Atomic categories', '-----------------', sep='\n', file=out)
    # print_counter_stats(atomic_categories, 'category', None, file=out)
    #
    # print(file=out)
    # print('Combinators', '-----------', sep='\n', file=out)
    # print_counter_stats(combinators, 'combinator', None, file=out)
    #
    # print(file=out)
    # print('# unary combinators in a row', '-----------', sep='\n', file=out)
    # print_counter_stats(unary, 'unaries in a row', None, file=out)
    #
    # print(file=out)
    # print('Level of unary combinators', '-----------', sep='\n', file=out)
    # print_counter_stats(unary_levels, 'level', None, file=out)
    #
    # print(file=out)
    # print('Unary combinators per sentence', '-----------', sep='\n', file=out)
    # print_counter_stats(sentence_unary, 'unaries per sentence', None, file=out)

    # print freqs
    #
    # for i, (k, v) in enumerate(categories.most_common(len(categories))):
    #     print(v, file=out)

    most_frequent = {}

    # unstructured tagset
    #
    for i, (k, v) in enumerate(categories.most_common(len(categories))):
        print(i, k, v, sep='\t')
        if v >= args.freq_threshold:
            most_frequent[f'({k})'] = i

    # atomic tagset
    #
    # for i, (k, v) in enumerate(atomic_categories.most_common(len(atomic_categories))):
    #     print(i, k, v, sep='\t')
    #     if v >= args.freq_threshold:
    #         most_frequent[f'({k})'] = i

    json.dump(most_frequent, out, indent=2)

    out.close()


if __name__ == '__main__':
    args = argparse.main()
    main(args)
