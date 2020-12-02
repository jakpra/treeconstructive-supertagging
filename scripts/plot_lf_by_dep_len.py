'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import glob

from collections import Counter, defaultdict
import json

import math
import pandas
from decimal import Decimal, ROUND_HALF_UP

from matplotlib import pyplot as plt
import seaborn as sns

from tree_st.util.reader import AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader


def round_to_nearest_multiple(x, n):
    return n * (x // n)


def bin_offset(x):
    # if x < 10:
    #     return 1
    if x < 5:
        return 5
    # elif x < 40:
    #     return 10
    else:
        return 15


infiles = glob.glob(sys.argv[1])
out_prefix = sys.argv[2]

# data_runs = defaultdict(list)
# bins = []


binned_raw_eval = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for infile in infiles:
    with open(infile) as f:
        raw_eval = json.load(f)
    name = infile.split('.')[1]
    # name, run = name.rsplit('-', maxsplit=1)
    eval = defaultdict(float)
    for bin, bin_eval in raw_eval.items():
        bin = int(bin)
        binned_raw_eval[name][round_to_nearest_multiple(bin, bin_offset(bin))]['correct'] += bin_eval['correct']
        binned_raw_eval[name][round_to_nearest_multiple(bin, bin_offset(bin))]['incorrect'] += bin_eval['incorrect']
        binned_raw_eval[name][round_to_nearest_multiple(bin, bin_offset(bin))]['missing'] += bin_eval['missing']


data = defaultdict(list)
for _name, raw_eval in binned_raw_eval.items():
    name, run = _name.rsplit('-', maxsplit=1)
    eval = defaultdict(float)
    for bin, bin_eval in raw_eval.items():
        if bin_eval['correct'] + bin_eval['incorrect'] + bin_eval['missing'] == 0:
            continue
        p = 1.0 if bin_eval['correct'] + bin_eval['incorrect'] == 0 else bin_eval['correct']/(bin_eval['correct'] + bin_eval['incorrect'])
        r = 0.0 if bin_eval['correct'] + bin_eval['missing'] == 0 else bin_eval['correct']/(bin_eval['correct'] + bin_eval['missing'])
        f = 0.0 if p + r == 0 else (2*p*r)/(p+r)

        data['model'].append(name)
        data['length'].append(bin)
        data['avg F1'].append(f)

    #     eval[bin] = f
    # data_runs[name].append(eval)

# data = defaultdict(list)
# for name, runs in data_runs.items():
#     keys = set.intersection(*[set(r.keys()) for r in runs])
#     # for r in runs[1:]:
#     #     assert r.keys() == keys, f'{name} bins divergence! {set(r.keys()).difference(set(keys))} | {set(keys).difference(set(r.keys()))}'
#     for key in keys:
#         avg_f = sum(run[key] for run in runs)/len(runs)
#
#         data['model'].append(name)
#         data['length'].append(key)
#         data['avg F1'].append(avg_f)

data = pandas.DataFrame(data)


# x_lim = math.ceil(max(data[complex]))
# subplot_kws = {'yscale': 'log',
#                # 'xscale': 'log',
#                # 'xscale_kws': {'basex': 2},
#                # 'xlim': (-0.5, 6.5),
#                'ylim': (0.5, 1000000),
#                'xticks': range(0, x_lim, 1)
#                }


sns.lineplot(x="length", y="avg F1",
              hue="model", style="model",
              markers=True, data=data, ci='sd')

plt.savefig(f'{out_prefix}.dep_len_f1.pdf', format='pdf')
plt.show()

# sns.relplot(x=complex, y='freq', hue='train freq', size='cat types', col='split',
#             data=data, sizes=(1, 200),
#             alpha=0.5, palette=sns.xkcd_palette(['green', 'orange', 'purple']),
#             linewidth=0.5, edgecolor='face', col_wrap=2,
#             col_order=order,
#             kind='scatter', aspect=1, facet_kws={'subplot_kws': subplot_kws})
