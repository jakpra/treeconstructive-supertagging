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

    if args.testing_format == 'ast':
        dr = ASTDerivationsReader
    elif args.testing_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    for filepath in args.testing_files:
        ds = dr(filepath, print_err_msgs=True)
        for d in ds:
            deriv = d['DERIVATION']

            if args.output_format == 'ast':
                raise NotImplementedError
            elif args.output_format == 'stagged':
                print(deriv.print_stagged(), file=out)
            elif args.output_format == 'pretty':
                print(deriv.pretty_print(), file=out)
            else:
                print(deriv, file=out)

        ds.close()

    if args.out:
        out.close()


if __name__ == '__main__':
    argparse.argparser.add_argument('--output-format', type=str)
    args = argparse.main()
    main(args)
