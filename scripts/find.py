'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys, argparse
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import random
import pandas
import json

from tree_st.util import argparse
from tree_st.util.reader import AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader
from tree_st.util.statistics import print_counter_stats
from tree_st.ccg.category import Slashes as sl


def main(queries, keys, invert, train_files, dev_files, test_files, args):
    out = open(args.out, 'w') if args.out else sys.stdout

    if args.training_format == 'ast':
        dr = ASTDerivationsReader
    elif args.training_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    results = []
    combinators = Counter()
    atomic_categories = Counter()
    categories = Counter()
    category_shapes = Counter()
    depth_counts = Counter()
    slashes = Counter()
    tl_slashes = Counter()
    addresses = Counter()
    unary = Counter()
    unary_levels = Counter()
    sentence_unary = Counter()
    for filepath in tqdm(train_files):
        ds = dr(filepath)
        for d in ds:
            deriv = d['DERIVATION']
            combs = deriv.count_combinators()
            combinators.update(combs)

            atom_cats = deriv.count_atomic_categories()
            atomic_categories.update(atom_cats)

            cats = deriv.count_categories()
            categories.update(cats)

            cat_shapes = deriv.count_category_shapes()
            category_shapes.update(cat_shapes)

            depth_counts.update(deriv.count_depths())

            sls = deriv.count_slashes()
            slashes.update(sls)

            tl_slashes.update(deriv.count_tl_slashes())

            addrs = deriv.count_addresses(labels=(sl.F, sl.B))
            addresses.update(addrs)

            unary.update(deriv.count_multiple_unary())
            unary_levels.update(deriv.count_unary_levels())
            sentence_unary.update(deriv.count_sentence_unary())

            words = deriv.count_words()

            all_keys = {'x': combs,
                        'a': atom_cats,
                        'c': cats,
                        'u': cat_shapes,
                        's': sls,
                        'p': addrs,
                        'w': words}
            values = [all_keys[k] for k in keys]
            for v in values:
                if len(queries) == 1:
                    query = queries[0].lower()
                    check = (query in map(lambda s: str(s).lower(), v)) == (not invert)
                    match = query
                elif len(queries) > 1:
                    for value in map(lambda s: str(s), v):
                        if value != 'None':
                            check = (value in queries) == (not invert)
                            if check:
                                match = value
                                break
                        match = None
                else:
                    pass

                if check:
                    d['FILE'] = filepath
                    d['MATCH'] = match
                    results.append(d)
                    break
        ds.close()

    for result in results:
        print(result['FILE'], result['ID'], result['DERIVATION'])
        print(result['MATCH'])
        print(result['DERIVATION'].pretty_print())


if __name__ == '__main__':
    argparse.argparser.add_argument('query', type=str)
    argparse.argparser.add_argument('--keys', type=str, default='acpsuwx')
    argparse.argparser.add_argument('-f', '--query-is-file', action='store_true')
    argparse.argparser.add_argument('-v', '--invert', action='store_true')
    args = argparse.main()

    if args.query_is_file:
        with open(args.query) as f:
            query = json.load(f)
    else:
        query = [args.query]

    main(query, args.keys, args.invert, args.training_files, args.development_files, args.testing_files, args)
