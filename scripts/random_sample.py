'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys, argparse
import random
import math
import json
from tqdm import tqdm

from tree_st.util import argparse
from tree_st.util.reader import ASTDerivationsReader, AUTODerivationsReader, StaggedDerivationsReader
from tree_st.util.statistics import print_counter_stats
from tree_st.ccg.category import Slashes as sl


def main(args):

    if args.training_format == 'ast':
        dr = ASTDerivationsReader
    elif args.training_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    # categories = Counter()
    # words = Counter()
    # usages = Counter()
    sent_ids = []
    for filepath in tqdm(args.training_files):
        ds = dr(filepath)
        for d in ds:
            sent_ids.append(d['ID'])
        ds.close()

    total = len(sent_ids)
    sent_ids = set(sent_ids)
    for i in tqdm(range(args.n_samples)):
        smpl = random.sample(sent_ids, math.floor(total*args.proportion))
        with open(f'{args.out}_{i}.json', 'w') as f:
            json.dump(smpl, f, indent=2)
        sent_ids -= set(smpl)


if __name__ == '__main__':
    ap = argparse.argparser
    ap.add_argument('--proportion', type=float, default=0.1)
    ap.add_argument('--n-samples', type=int, default=10)
    args = argparse.main()
    main(args)
