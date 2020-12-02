"""
Extract super-atomic sub-categories from a CCG corpus, e.g., by means of frequency, byte-pair encoding (have to generalize to trees),
fixed-size subtrees (bottom-up vs top-down)...

baseline mode: 425 most frequent complete categories
extreme atomic mode: slashes and atomic categories
intermediate modes: slashes and super-atomic sub-categories in terms of frequency, byte-pair encoding

returns json


@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
"""


import sys
import json

from collections import Counter

import tree_st.util.argparse as ap
from tree_st.util.reader import DerivationsReader
from tree_st.util.functions import bottom_up

# extract all tags as many times as they occur (preprocessing for extracting subtags with tfreq)
extr = ap.mode.add_parser('extract')

# extract tags by frequency
freq = ap.mode.add_parser('extract_freq')
freq.add_argument('--top-k', type=int, default=425)

args = ap.main()


cat_freqs = Counter()
for filepath in args.training_files:
    for d in DerivationsReader(filepath):
        deriv = d['DERIVATION']
        bu = bottom_up(len(deriv.sentence))

        for ((i, j), u), (_, cat), comb in sorted(deriv.categories(), key=lambda x: bu(*x[0][0])):

            if j - i == 1:
                if u == 0:
                    cat_freqs[cat] += 1

if args.mode == 'extract':
    with open(args.out, 'w', newline='\n') as f:
        for cat, freq in cat_freqs.most_common():
            for _ in range(freq):
                print(cat.s_expr(), file=f)

elif args.mode == 'extract_freq':
    labels = {}
    for i, (cat, _) in enumerate(cat_freqs.most_common()[:args.top_k]):
        labels[str(cat)] = i

    with open(args.out, 'w') as f:
        json.dump(labels, f, indent=2)
