'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys, argparse
from collections import Counter, defaultdict

from tree_st.util import argparse
from tree_st.util.reader import ASTDerivationsReader, AUTODerivationsReader, StaggedDerivationsReader
from tree_st.util.statistics import print_counter_stats
from tree_st.ccg.category import Slashes as sl


def main(args):
    out = open(args.out, 'w', newline='\n', encoding='utf-8', errors='ignore') if args.out else sys.stdout

    for_easyccg = False

    for_parser = True or for_easyccg

    # if args.training_format == 'ast':
    #     dr = ASTDerivationsReader
    # elif args.training_format == 'stagged':
    #     dr = StaggedDerivationsReader
    # else:
    #     dr = AUTODerivationsReader
    #
    # # categories = Counter()
    # # words = Counter()
    # # usages = Counter()
    # cats_by_wordpos = defaultdict(Counter)
    # cats_by_word = defaultdict(Counter)
    # cats_by_pos = defaultdict(Counter)
    # for filepath in args.training_files:
    #     ds = dr(filepath)
    #     for d in ds:
    #         deriv = d['DERIVATION']
    #         print(d['ID'], file=sys.stderr)
    #         if args.derivation:
    #             print(d['ID'], file=out)
    #             print(deriv, file=out)
    #         lex = deriv.get_lexical(ignore_attr=False)
    #         # combinators.update(deriv.count_combinators())
    #         for dln in lex:
    #             # atomic_categories.update(dln.category1.count_atomic_categories(concat_attr=True))
    #             # categories[dln.category1] += 1
    #             # words[dln.word.lower()] += 1
    #             # usages[dln.word.lower(), dln.category1] += 1
    #             cats_by_wordpos[dln.word.lower(), dln.pos1][dln.category1] += 1
    #             cats_by_word[dln.word.lower()][dln.category1] += 1
    #             cats_by_pos[dln.pos1][dln.category1] += 1

    if args.testing_format == 'ast':
        dr = ASTDerivationsReader
    elif args.testing_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    if not for_parser:
        print(
          # 'index', 'word', 'POS1', 'POS2',
          'gold category', 'depth', 'arguments', 'size',
          # 'word frequency', 'category frequency', 'usage frequency',
          sep='\t', file=out)
    i = 0
    for filepath in args.testing_files:
        ds = dr(filepath, print_err_msgs=True)
        for d in ds:
            deriv = d['DERIVATION']
            print(d['ID'], file=sys.stderr)
            if args.derivation:
                print(d['ID'], file=out)
                print(deriv, file=out)
            lex = deriv.get_lexical(ignore_attr=False)
            # combinators.update(deriv.count_combinators())
            if for_easyccg:
                print('|'.join(['\t'.join([dln.word, dln.pos1, '0', str(dln.category1), '1']) for dln in lex]), file=out)
                # print(' '.join([dln.word for dln in lex]), file=out)
            else:
                for dln in lex:
                    cat = dln.category1
                    if for_parser:
                        cat = str(cat)
                        # if cat == '-UNKNOWN-':
                        #     if (dln.word.lower(), dln.pos1) in cats_by_wordpos:
                        #         cat = cats_by_wordpos[dln.word.lower(), dln.pos1].most_common(1)[0][0]
                        #     elif dln.word.lower() in cats_by_word:
                        #         cat = cats_by_word[dln.word.lower()].most_common(1)[0][0]
                        #     else:
                        #         cat = cats_by_pos[dln.pos1].most_common(1)[0][0]
                        # print(dln.word, dln.pos1, 1, cat, 1, sep='\t', file=out)
                        print(dln.word, dln.pos1, 1, cat, 1, sep='\t', file=out)
                        # print(dln.word, dln.pos1, cat, sep='\t', file=out)
                    else:
                        print(
                          # i, dln.word, dln.pos1, dln.pos2,
                          cat, cat.depth(), cat.nargs(), cat.size(),
                          # words[dln.word.lower()], categories[cat], usages[dln.word.lower(), cat],
                          sep='\t', file=out)
                    i += 1
            # if for_parser:
            #     print(file=out)

        ds.close()

    if args.out:
        out.close()


if __name__ == '__main__':
    args = argparse.main()
    main(args)
