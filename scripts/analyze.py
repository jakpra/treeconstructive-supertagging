'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys, argparse
from collections import Counter, defaultdict
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import random
import pandas
from decimal import Decimal, ROUND_HALF_UP

from tree_st.util.reader import AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader
from tree_st.util.statistics import print_counter_stats
from tree_st.ccg.category import Slashes as sl


def round_to_nearest_quantile(x, n):
    lg = math.floor(math.log(x, n))
    m = Decimal(f'1e{lg}')
    return int(Decimal(f'{(Decimal(x) / m).quantize(Decimal("1."), rounding=ROUND_HALF_UP)}e{lg}'))


def main(train_files, dev_files, test_files,
         unaries=0, generate_figure=False, lexical=True, complex='size', oov_sent_ids=[],
         output_new_split=None, generate_new_split=None,
         args=None):

    out = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout

    if args.training_format == 'ast':
        dr = ASTDerivationsReader
    elif args.training_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    combinators = Counter()
    atomic_categories = Counter()
    categories = Counter()
    categories_remaining = Counter()
    categories_selected = defaultdict(Counter)
    category_shapes = Counter()
    depth_counts = Counter()
    size_counts = Counter()
    arg_counts = Counter()
    slashes = Counter()
    tl_slashes = Counter()
    addresses = Counter()
    unary = Counter()
    unary_levels = Counter()
    sentence_unary = Counter()
    pos_uniq_cat = defaultdict(Counter)
    file_iter = tqdm(train_files)
    for filepath in file_iter:
        ds = dr(filepath)
        try:
            for d in ds:
                deriv = d['DERIVATION']
                if args.derivation:
                    print(f'ID={d["ID"]}', file=out)
                    print(deriv, file=out)

                if lexical:
                    deriv = deriv.lexical_deriv()

                combs = deriv.count_combinators()
                combinators.update(combs)

                atom_cats = deriv.count_atomic_categories(concat_attr=True, u=unaries)
                atomic_categories.update(atom_cats)

                cats = deriv.count_categories(u=unaries)
                categories.update(cats)
                id_selected = False
                for i, ids in enumerate(oov_sent_ids):
                    if d['ID'] in ids:
                        id_selected = True
                        if output_new_split is not None:
                            with open(f'{output_new_split}/oov/{d["ID"]}.auto', 'w') as f:
                                print(' '.join('='.join([k, v]) for k, v in d.items() if k != 'DERIVATION'), file=f)
                                print(deriv, file=f)
                        categories_selected[i].update(cats)

                if not id_selected:
                    if output_new_split is not None:
                        with open(f'{output_new_split}/iv/{d["ID"]}.auto', 'w') as f:
                            print(' '.join('='.join([k, v]) for k, v in d.items() if k != 'DERIVATION'), file=f)
                            print(deriv, file=f)
                    categories_remaining.update(cats)

                cat_shapes = deriv.count_category_shapes(u=unaries)
                category_shapes.update(cat_shapes)

                depth_counts.update(deriv.count_depths(u=unaries))
                size_counts.update(deriv.count_sizes(u=unaries))
                arg_counts.update(deriv.count_args(u=unaries))

                sls = deriv.count_slashes(u=unaries)
                slashes.update(sls)

                tl_slashes.update(deriv.count_tl_slashes(u=unaries))

                addrs = deriv.count_addresses(labels=(sl.F, sl.B), u=unaries)
                addresses.update(addrs)

                _pos_uniq_cats = deriv.get_unique_cats_by_pos()
                for k in _pos_uniq_cats:
                    pos_uniq_cat[k].update(_pos_uniq_cats[k])

                unary.update(deriv.count_multiple_unary())
                unary_levels.update(deriv.count_unary_levels())
                sentence_unary.update(deriv.count_sentence_unary())
        except (UnicodeDecodeError, ArithmeticError) as e:
            ds.close()
            raise

        ds.close()

    print('Category depths', '---------------', sep='\n', file=out)
    print_counter_stats(depth_counts, 'depth', None, file=out)

    print(file=out)
    print('Category sizes', '---------------', sep='\n', file=out)
    print_counter_stats(size_counts, 'size', None, file=out)

    print(file=out)
    print('Categories', '-----------------', sep='\n', file=out)
    print_counter_stats(categories, 'category', None, file=out)

    print(file=out)
    print('Category shapes', '-----------------', sep='\n', file=out)
    print_counter_stats(category_shapes, 'shape', None, file=out)

    print(file=out)
    print('Top-level slashes', '-----------', sep='\n', file=out)
    print_counter_stats(tl_slashes, 'slash', None, file=out)

    print(file=out)
    print('Slash addresses', '-----------', sep='\n', file=out)
    print_counter_stats(addresses, 'address', None, file=out)

    print(file=out)
    print('Slashes', '-----------', sep='\n', file=out)
    print_counter_stats(slashes, 'slash', None, file=out)

    print(file=out)
    print('Atomic categories', '-----------------', sep='\n', file=out)
    print_counter_stats(atomic_categories, 'category', None, file=out)

    print(file=out)
    print('Combinators', '-----------', sep='\n', file=out)
    print_counter_stats(combinators, 'combinator', None, file=out)

    print(file=out)
    print('# unary combinators in a row', '-----------', sep='\n', file=out)
    print_counter_stats(unary, 'unaries in a row', None, file=out)

    print(file=out)
    print('Level of unary combinators', '-----------', sep='\n', file=out)
    print_counter_stats(unary_levels, 'level', None, file=out)

    print(file=out)
    print('Unary combinators per sentence', '-----------', sep='\n', file=out)
    print_counter_stats(sentence_unary, 'unaries per sentence', None, file=out)

    print(file=out)
    print('Tokens by POS', '-----------', sep='\n', file=out)
    print_counter_stats(Counter({k: sum(v.values()) for k, v in pos_uniq_cat.items()}), 'tokens per POS', None, file=out)

    print(file=out)
    print('Unique categories by POS', '-----------', sep='\n', file=out)
    print_counter_stats(Counter({k: len(v) for k, v in pos_uniq_cat.items()}), 'unique cats per POS', None, file=out)

    if args.out:
        out.close()

    if generate_figure:

        if args.development_format == 'ast':
            dr = ASTDerivationsReader
        elif args.development_format == 'stagged':
            dr = StaggedDerivationsReader
        else:
            dr = AUTODerivationsReader

        dev_categories = Counter()
        for filepath in tqdm(dev_files):
            ds = dr(filepath)
            try:
                for d in ds:
                    deriv = d['DERIVATION'].lexical_deriv() if lexical else d['DERIVATION']
                    if args.derivation:
                        print(d['ID'], file=out)
                        print(deriv, file=out)
                    dev_categories.update(deriv.count_categories())
            except (UnicodeDecodeError, ArithmeticError):
                ds.close()
                continue
            ds.close()

        if args.testing_format == 'ast':
            dr = ASTDerivationsReader
        elif args.testing_format == 'stagged':
            dr = StaggedDerivationsReader
        else:
            dr = AUTODerivationsReader

        test_categories = Counter()
        for filepath in tqdm(test_files):
            ds = dr(filepath)
            try:
                for d in ds:
                    deriv = d['DERIVATION'].lexical_deriv() if lexical else d['DERIVATION']
                    if args.derivation:
                        print(d['ID'], file=out)
                        print(deriv, file=out)
                    test_categories.update(deriv.count_categories())
            except (UnicodeDecodeError, ArithmeticError):
                ds.close()
                continue
            ds.close()

        freqs = []
        sizes = []
        depths = []
        thresh_freqs = []
        thresh_depths = []
        thresh_sizes = []
        train_sizes = Counter()
        nothresh_freqs = []
        nothresh_depths = []
        nothresh_sizes = []
        threshold = []
        depths_dict = defaultdict(list)
        for i, (c, freq) in enumerate(categories.most_common(len(categories))):
            freqs.append(freq)
            d = c.depth()
            depths.append(d)
            s = c.size()
            sizes.append(s)
            a = c.nargs()
            depths_dict[d].append(i)
            if categories[c] >= 100:
                thresh_freqs.append(freq)
                thresh_depths.append(d)
                thresh_sizes.append(s)
                threshold.append('\u226510')
                train_sizes[d if complex == 'depth' else a if complex == 'args' else s//2 if complex=='slashes' else s, freq, freq, 'n \u2265 100'] += 1  #  + .1
            elif categories[c] >= 10:
                # nothresh_freqs.append(freq + 0.2 * (random.random() - 0.5))
                # nothresh_depths.append(d + 0.2 * (random.random() - 0.5))
                # nothresh_sizes.append(s + 0.2 * (random.random() - 0.5))
                # threshold.append('<10')
                train_sizes[d if complex=='depth' else a if complex == 'args' else s//2 if complex=='slashes' else s, freq, freq, '10 \u2264 n < 100'] += 1  #  - .1
            else:
                nothresh_freqs.append(freq + 0.2 * (random.random() - 0.5))
                nothresh_depths.append(d + 0.2 * (random.random() - 0.5))
                nothresh_sizes.append(s + 0.2 * (random.random() - 0.5))
                threshold.append('<10')
                train_sizes[d if complex=='depth' else a if complex == 'args' else s//2 if complex=='slashes' else s, freq, freq, '1 \u2264 n < 10'] += 1  #  - .1
            # types
        #     if i > bins[bin]:
        #         types_count_bins.append(types_count)
        #         types_count = 0
        #         bin += 1
        #     types_count += 1
        # while len(types_count_bins) < len(bins):
        #     types_count_bins.append(types_count)
        #     types_count = 0

        train_remaining_sizes = Counter()
        for i, (c, freq) in enumerate(categories_remaining.most_common(len(categories_remaining))):
            d = c.depth()
            s = c.size()
            a = c.nargs()
            train_freq = categories[c]
            # test_sizes.append(s)

            # if categories_remaining[c] == 0:
            #     if 1 <= categories[c] < 10:
            #         train_remaining_sizes[(
            #                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '<10'] += 1
            #     else:
            #         train_remaining_sizes[(
            #                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
            if categories_remaining[c] == 0:
                train_remaining_sizes[(
                                      d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
            elif categories_remaining[c] < 10:
                train_remaining_sizes[(
                                      d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '1 \u2264 n < 10'] += 1
            elif categories[c] < 100:
                train_remaining_sizes[d if complex == 'depth' else a if complex == 'args' else s//2 if complex == 'slashes' else s, freq, freq, '10 \u2264 n < 100'] += 1

            else:
                train_remaining_sizes[(
                                     d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'n \u2265 100'] += 1

        train_selected_sizes = defaultdict(Counter)
        train_selected_remaining_sizes = defaultdict(Counter)
        train_selected_self_sizes = defaultdict(Counter)
        for j, cats_selected in categories_selected.items():
            for i, (c, freq) in enumerate(cats_selected.most_common(len(cats_selected))):
                d = c.depth()
                s = c.size()
                a = c.nargs()
                train_freq = categories[c]

                if categories[c] == 0:
                    train_selected_sizes[j][(
                                   d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
                elif categories[c] < 10:
                    train_selected_sizes[j][(
                                  d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '<10'] += 1
                else:
                    train_selected_sizes[j][(
                                  d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '\u226510'] += 1

                if categories_remaining[c] == 0:
                    train_selected_remaining_sizes[j][(
                                       d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
                elif categories_remaining[c] < 10:
                    train_selected_remaining_sizes[j][(
                                       d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '1 \u2264 n < 10'] += 1
                elif categories_remaining[c] < 100:
                    train_selected_remaining_sizes[j][
                        d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s, freq, freq, '10 \u2264 n < 100'] += 1

                else:
                    train_selected_remaining_sizes[j][(
                                       d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'n \u2265 100'] += 1

                if categories_selected[j][c] == 0:
                    train_selected_self_sizes[j][(
                                      d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
                elif categories_selected[j][c] < 10:
                    train_selected_self_sizes[j][(
                                      d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '<10'] += 1
                else:
                    train_selected_self_sizes[j][(
                                      d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '\u226510'] += 1

        dev_train_freqs = []
        dev_train_depths = []
        dev_train_sizes = []
        dev_sizes = Counter()
        dev_half_sizes = Counter()
        dev_notrain_freqs = []
        dev_notrain_depths = []
        dev_notrain_sizes = []
        dev_freqs = []
        dev_depths = []
        # dev_sizes = []
        dev_train = []
        for i, (c, freq) in enumerate(dev_categories.most_common(len(dev_categories))):
            d = c.depth()
            s = c.size()
            a = c.nargs()
            dev_freqs.append(freq)
            dev_depths.append(d)
            train_freq = categories[c]
            # dev_sizes.append(s)
            if categories[c] == 0:
                dev_sizes[(
                               d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
            elif categories[c] < 10:
                dev_sizes[(
                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '<10'] += 1
            else:
                dev_sizes[(
                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '\u226510'] += 1

            if categories_remaining[c] == 0:
                dev_half_sizes[(
                                   d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
            else:
                dev_half_sizes[(
                                   d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '\u226510'] += 1

        test_sizes = Counter()
        test_half_sizes = Counter()
        for i, (c, freq) in enumerate(test_categories.most_common(len(test_categories))):
            d = c.depth()
            s = c.size()
            a = c.nargs()
            train_freq = categories[c]
            # test_sizes.append(s)
            if categories[c] == 0:
                test_sizes[(
                               d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
            elif categories[c] < 10:
                test_sizes[(
                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '1 \u2264 n < 10'] += 1
            elif categories[c] < 100:
                test_sizes[(
                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '10 \u2264 n < 100'] += 1
            else:
                test_sizes[(
                              d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'n \u2265 100'] += 1

            if categories_remaining[c] == 0:
                test_half_sizes[(
                                   d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, 'unseen'] += 1
            else:
                test_half_sizes[(
                                   d if complex == 'depth' else a if complex == 'args' else s // 2 if complex == 'slashes' else s), freq, train_freq, '\u226510'] += 1

        # freqs = np.array(freqs)
        # thresh_freqs = np.array(thresh_freqs)
        # dev_train_freqs = np.array(dev_train_freqs)
        # test_train_freqs = np.array(test_train_freqs)
        # dev_notrain_freqs = np.array(dev_notrain_freqs)
        # test_notrain_freqs = np.array(test_notrain_freqs)

        # fig, ax1 = plt.subplots(1, 1)
        # sns.set(style="whitegrid")

        # color = 'tab:red'

        # ax1.set_xlabel('frequency rank')
        # ax1.set_ylabel('log freq / depth')
        # ax1.scatter(t, np.log10(freqs), s=15, color='tab:red')
        # ax1.scatter(t, depths, s=15, color='tab:blue')
        # depths_viol = [idxs for i, idxs in sorted(depths_dict.items())]
        # ax1.violinplot(depths_viol, sorted(depths_dict.keys()), vert=False, widths=0.7)
        # ax1.axvline(425, color='k')


        # col_order = ['train',
        #              'train_oov seen', 'train_oov unseen',
        #              'dev seen', 'dev unseen',
        #              'test seen', 'test unseen',
        #              'train >10',
        #              'train_oov >10 seen', 'train_oov >10 unseen',
        #              'dev >10 seen', 'dev >10 unseen',
        #              'test >10 seen', 'test >10 unseen',
        #              ]


        # for (d, f, t), s in dev_sizes.items():
        #     data['depth' if use_depth else 'size'].append(math.log2(d))  # + 0.3*(random.random()-0.5)
        #     data['freq'].append(f)
        #     data['Train freq'].append(t)
        #     data['# Cat types'].append(s)
        #     data['split'].append('test')

        # max_x =
        # ax1.margins(0.2, tight=False)
        # ax1.set_title('train')
        # ax1.set_xlabel('depth')
        # ax1.set_ylabel('freq')
        # ax1.set_yscale('log')

        if complex == 'none':

            data = defaultdict(list)
            all_cats = sorted(categories.keys() | dev_categories.keys() | test_categories.keys(), key=lambda x: categories.get(x, 0), reverse=True)

            # for i, cat in enumerate(all_cats):
                # cat = str(cat)
            for split, counts in zip(['train', 'dev', 'test'], [categories, dev_categories, test_categories]):
                # print(split, counts)
                for cat, count in counts.most_common(len(counts)):
                    data['train_rank'].append(categories[cat])
                    data['split'].append(split)
                # data['train_freq'].append(categories[cat])
                # data['freq'].append(counts[cat])
                # data['split'].append(split)

            # print(data, file=sys.stderr)

            # order = ['train',
            #          # 'dev',
            #          'test',
            #          # 'train \u226510',
            #          # 'train <10',
            #          # 'train >10',
            #          # 'train_oov >10',
            #          # 'dev >10',
            #          # 'test >10',
            #          ]

            subplot_kws = {'xscale': 'log',
                           # 'xscale': 'log',
                           # 'xscale_kws': {'basex': 2},
                           # 'xlim': (-0.5, 6.5),
                           # 'ylim': (0.5, 1000000),
                           # 'xticks': range(0, x_lim, 2)
                           }

            print('ok1')
            sns.violinplot(x='train_rank', y='split', data=data, order=order,
                           bw=0.2, cut=0, inner='point', ax=plt.subplot(xscale='log'))  # scale='width',
                            # , subplot_kws=subplot_kws)
            print('ok2')
        else:

            data = defaultdict(list)

            offsets = {'n \u2265 100': -0.3, '10 \u2264 n < 100': -0.1, '1 \u2264 n < 10': 0.1, 'unseen': 0.3}

            binned = Counter()
            for (d, f, tf, t), s in train_remaining_sizes.items():
                binned[d, round_to_nearest_quantile(f, 10), t] += s
            for (d, f, t), s in binned.items():
                data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
                data['freq'].append(f)
                # data['train_freq'].append(tf)
                data['Train freq'].append(t)
                data['# Cat types'].append(s)
                # data['split'].append(f'train \u226510')
                data['split'].append(f'non-tail sentences')

            # for (d, f, tf, t), s in train_oov_half_sizes.items():
            #     data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
            #     data['freq'].append(f)
            #     data['train_freq'].append(tf)
            #     data['Train freq'].append(t)
            #     data['# Cat types'].append(s)
            #     data['split'].append(f'train_oov >=10')  # {t if t=="seen" else "unseen"}

            # binned = Counter()
            # for (d, f, tf, t), s in dev_half_sizes.items():
            #     binned[d, round_to_nearest_quantile(f, 10), t] += s
            #     # binned[d, round_to_nearest_quantile(tf, 10), t] += s
            # for (d, f, t), s in binned.items():
            #     d = d + offsets[t]
            #     data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
            #     data['freq'].append(f)
            #     # data['train_freq'].append(tf)
            #     data['Train freq'].append(t)
            #     data['# Cat types'].append(s)
            #     data['split'].append(f'dev >=10')

            # binned = Counter()
            # for (d, f, tf, t), s in test_half_sizes.items():
            #     binned[d, round_to_nearest_quantile(f, 10), t] += s
            # for (d, f, t), s in binned.items():
            #     d = d + offsets[t]
            #     data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
            #     data['freq'].append(f)
            #     # data['train_freq'].append(tf)
            #     data['Train freq'].append(t)
            #     data['# Cat types'].append(s)
            #     data['split'].append(f'test >=10')

            binned = Counter()
            for (d, f, tf, t), s in train_sizes.items():
                binned[d, round_to_nearest_quantile(f, 10), t] += s
            for (d, f, t), s in binned.items():
                data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
                data['freq'].append(f)
                # data['train_freq'].append(tf)
                data['Train freq'].append(t)
                data['# Cat types'].append(s)
                data['split'].append(f'train')

            for i in categories_selected:
                binned = Counter()
                # for (d, f, tf, t), s in train_selected_sizes[i].items():
                #     binned[d, round_to_nearest_quantile(f, 10), t] += s
                for (d, f, tf, t), s in train_selected_remaining_sizes[i].items():
                    binned[d, round_to_nearest_quantile(f, 10), t] += s
                # for (d, f, tf, t), s in train_selected_sizes[i].items():
                for (d, f, t), s in binned.items():
                    d = d + offsets[t]
                    data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
                    data['freq'].append(f)
                    # data['train_freq'].append(tf)
                    data['Train freq'].append(t)  # {'\u226510': '\u226510', 'unseen': 'unseen', '<10': 'unseen'}[t])
                    data['# Cat types'].append(s)
                    # data['split'].append(f'train <10')
                    # data['split'].append(f'sample {i+1}')
                    data['split'].append(f'tail sentences')

            # binned = Counter()
            # for (d, f, tf, t), s in dev_sizes.items():
            #     binned[d, round_to_nearest_quantile(f, 10), t] += s
            # for (d, f, t), s in binned.items():
            #     d = d + offsets[t]
            #     data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
            #     data['freq'].append(f)
            #     # data['train_freq'].append(tf)
            #     data['Train freq'].append(t)
            #     data['# Cat types'].append(s)
            #     data['split'].append(f'dev')

            binned = Counter()
            for (d, f, tf, t), s in test_sizes.items():
                binned[d, round_to_nearest_quantile(f, 10), t] += s
            for (d, f, t), s in binned.items():
                d = d + offsets[t]
                data[complex].append(math.log2(d) if complex.endswith('_log') else d)  # + 0.3*(random.random()-0.5)
                data['freq'].append(f)
                # data['train_freq'].append(tf)
                data['Train freq'].append(t)
                data['# Cat types'].append(s)
                data['split'].append(f'test')

            data = pandas.DataFrame(data)

            order = [
                         'train',
                         # 'dev',
                         'test',
                         # 'train \u226510',
                         # 'train <10',
                        # 'non-tail sentences',
                        # 'tail sentences',

                         # 'train >10',
                         # 'train_oov >10',
                         # 'dev >10',
                         # 'test >10',
                         ]

            # order = [f'sample {i+1}' for i in range(len(categories_selected))]


            #########################################################################################

            x_lim = math.ceil(max(data[complex]))
            subplot_kws = {'yscale': 'log',
                           # 'xscale': 'log',
                           # 'xscale_kws': {'basex': 2},
                           # 'xlim': (-0.5, 6.5),
                           'ylim': (0.5, 1000000),
                           # 'legend': {'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05)},
                           'xticks': range(0, x_lim, 1)
                           # 'grid': dict(b=True, which='major', color='w', linewidth=1.0)
                           }

            if complex.endswith('_log'):
                exp_str = dict(enumerate(['\u2070', '\u00B9', '\u00B2', '\u00B3', '\u2074', '\u2075', '\u2076']))
                subplot_kws['xticklabels'] = [f'2{exp_str[i]}' for i in subplot_kws['xticks']]

            # sns.set_style('whitegrid')  # font_scale=1
            sns.set(style='whitegrid', font_scale=1.0)  # , rc={"xtick.bottom": False, "ytick.left": True})

            # fig, ax = plt.subplots(1, 1)

            # ax.scatter(x, y)

            g = sns.relplot(x=complex, y='freq', hue='Train freq', size='# Cat types', col='split',
                        data=data, sizes=(3, 200),
                        # alpha=0.5,
                            palette=sns.color_palette(['#6eff95', '#fbbc04', '#ea4335', '#4285f4']),  # 'orange', 'red'
                            # ['#34a853', '#ff9900
                            # ', '#ea4335', '#4285f4']   # colorblind-friendly
                            # sns.xkcd_palette(['green', 'dark yellow', 'magenta', 'purple'])
                        linewidth=0.5, edgecolor='face', col_wrap=2,  # edgecolor='face'
                        col_order=order,
                        kind='scatter', aspect=1, facet_kws={'subplot_kws': subplot_kws, 'legend_out': False},
                        # legend=False
                            )
                # .add_legend(loc='upper right', bbox_to_anchor=(50, 50), fancybox=True, frameon=True)

            ax1 = g.facet_axis(0, 0)
            ax1.get_xaxis().set_minor_locator(mpl.ticker.FixedLocator([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])) # mpl.ticker.AutoMinorLocator()
            # ax1.get_xaxis().set_view_interval(-0.5, 4.5, ignore=True)
            # ax1.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax1.grid(b=False, axis='x', which='major', color='lightgrey', linewidth=0.0)
            ax1.grid(b=True, which='minor', color='lightgrey', alpha=0.5, linewidth=0.8)
            ax1.tick_params(left=True, axis='y', which='major')

            ax2 = g.facet_axis(0, 1)
            ax2.get_xaxis().set_minor_locator(mpl.ticker.FixedLocator([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
            # ax2.get_xaxis().set_view_interval(-0.5, 6.5, ignore=True)
            # ax2.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax2.grid(b=False, axis='x', which='major', color='lightgrey', linewidth=0.0)
            ax2.grid(b=True, which='minor', color='lightgrey', alpha=0.3, linewidth=0.8)
            ax2.tick_params(left=True, axis='y', which='major')

            # h, l = ax2.get_legend_handles_labels()

            # keep same handles, edit labels with names of choice
            # ax2.legend(loc='upper right', handles=h, labels=l)

            # g.despine(left=True)
            # g.add_legend(loc='upper right', bbox_to_anchor=(0, 0))

            h, l = ax1.get_legend_handles_labels()
            ax2.legend(h, l)
            ax1.get_legend().remove()
            # g._legend.set_draggable(True)
            # g._legend.set_loc('upper right')
            # g._legend.set_bbox_to_anchor()
            # g._legend.draw_frame(True)
            # # g.set_ylabels("survival probability")
            # ax2.legend(loc='upper right')  # , bbox_to_anchor=(0.5, -0.05))

            ###############################################################################################

            # # x_lim = max(data['freq'])
            # y_lim = math.ceil(max(data[complex]))
            # subplot_kws = {'xscale': 'log',
            #                # 'xscale': 'log',
            #                # 'xscale_kws': {'basex': 2},
            #                # 'xlim': (-0.5, 6.5),
            #                'xlim': (0.5, 1000000),
            #                # 'xticks': range(1000000, 0.5),
            #                'yticks': range(0, y_lim+1, 1)
            #                }
            #
            # if complex.endswith('_log'):
            #     exp_str = dict(enumerate(['\u2070', '\u00B9', '\u00B2', '\u00B3', '\u2074', '\u2075', '\u2076']))
            #     subplot_kws['yticklabels'] = [f'2{exp_str[i]}' for i in subplot_kws['yticks']]
            #
            # g = sns.relplot(x='train_freq', y=complex, hue='Train freq', size='# Cat types', col='split',
            #             data=data, sizes=(3, 600),
            #             alpha=0.5, palette=sns.xkcd_palette(['green', 'orange', 'purple']),
            #             linewidth=0.5, edgecolor='face', col_wrap=1,
            #             col_order=order,
            #             kind='scatter', aspect=2, facet_kws={'subplot_kws': subplot_kws})
            #
            # for i in range(math.ceil(len(order)/col_wrap)):
            #     for j in range(col_wrap):
            #         try:
            #             ax = g.facet_axis(i, j)
            #             ax.axvline(10, color='r')
            #         except:
            #             pass

            ###############################################################################################
            # violinplot

            # data = defaultdict(list)
            # all_cats = sorted(categories.keys() | dev_categories.keys() | test_categories.keys(),
            #                   key=lambda x: categories.get(x, 0), reverse=True)
            #
            # # for i, cat in enumerate(all_cats):
            # # cat = str(cat)
            # for split, counts in zip(['train', 'dev', 'test'], [categories, dev_categories, test_categories]):
            #     # print(split, counts)
            #     for cat, count in counts.most_common(len(counts)):
            #         data['freq'].append(categories[cat])
            #         data['split'].append(split)
            #
            # sns.violinplot(x='freq', y=complex, hue='Train freq', size='# Cat types', col='split', split=True,
            #                data=data, sizes=(1, 200),
            #                alpha=0.5, palette=sns.xkcd_palette(['green', 'orange', 'purple']),
            #                linewidth=0.5, edgecolor='face', col_wrap=3,
            #                col_order=order, aspect=1)  # , facet_kws={'subplot_kws': subplot_kws})

            ################################################################################################

        # ax1 = sns.swarmplot(x='depth', y='freq', color='#15b01a', data={'depth': thresh_depths, 'freq': thresh_freqs},
        #                                                             # 'train': threshold}
        #                     size=1, ax=ax1,
        #                     marker='o')
        # ax1 = sns.scatterplot(x='depth', y='freq', color='#e50000', data={'depth': nothresh_depths, 'freq': nothresh_freqs},
        #                                                                   # 'train': threshold},
        #                     # palette=['#e50000', '#929591'],
        #                     size=1, ax=ax1,
        #                     marker='o')
        # ax1.set_xlim(xmin=-0.5, xmax=6.5)
        # ax1.set_ylim(ymin=0.5, ymax=1000000)
        # ax.set_xticks(range(7))
        # ax.set_xticklabels(range(7))
        # ax1.set_ylim(ymin=0, ymax=1000000)
        # ax1.scatter(depths, np.log10(freqs), s=15, color='tab:blue')
        # ax1.scatter(t, depths, s=15, color='tab:blue')
        # depths_viol = [idxs for i, idxs in sorted(depths_dict.items())]
        # ax1.violinplot(depths_viol, sorted(depths_dict.keys()), vert=False, widths=0.7)
        # ax.axhline(10, color='k', linewidth=0.5)
        # for i in range(6):
        #     ax.axvline(i + 0.5, color='.9', linewidth=0.5)
        # ax1.margins(0.1)

        # ax2.set_title('dev')
        # ax2.set_xlabel('depth')
        # # ax2.set_yscale('log')
        # # ax2.set_ylabel('freq')
        # # ax2 = sns.swarmplot(x='depth', y='freq', hue='train', data={'depth': dev_depths, 'freq': dev_freqs, 'train': dev_train}, palette='Set2', size=1, ax=ax2)
        # ax2 = sns.scatterplot(x='d', y='f', hue='Train freq', size='# Cat types', data=dev_data, sizes=(5, 20), alpha=0.5, ax=ax2)
        # # ax2 = sns.swarmplot(x='depth', y='freq', color='#15b01a', data={'depth': dev_train_depths, 'freq': dev_train_freqs},
        # #                     # 'train': threshold}
        # #                     size=1, ax=ax2,
        # #                     marker='o')
        # # ax2 = sns.scatterplot(x='depth', y='freq', hue='train', data={'depth': dev_notrain_depths,
        # #                                                               'freq': dev_notrain_freqs,
        # #                                                               'train': dev_train},
        # #                       # 'train': threshold},
        # #                       palette=['#e50000', '#929591'],
        # #                       size=1, ax=ax2,
        # #                       marker='o')
        # ax2.set_xlim(xmin=-0.5, xmax=6.5)
        # ax2.set_xticks(range(7))
        # ax2.set_xticklabels(range(7))
        # ax2.set_ylim(ymin=0.5, ymax=1000000)
        # for i in range(6):
        #     ax2.axvline(i + 0.5, color='.9', linewidth=0.5)
        # # ax2.scatter(dev_train_depths, np.log10(dev_train_freqs), s=15, color='tab:blue')
        # # ax2.scatter(dev_notrain_depths, np.log10(dev_notrain_freqs), s=15, color='tab:red')
        #
        # ax3.set_title('test')
        # # ax3.set_xlabel('depth')
        # # ax3.set_ylabel('freq')
        # # ax3.set_yscale('log')
        # ax3.set_xlim(xmin=-0.5, xmax=6.5)
        # ax3.set_xticks(range(7))
        # ax3.set_xticklabels(range(7))
        # ax3.set_ylim(ymin=0.5, ymax=1000000)
        # for i in range(6):
        #     ax3.axvline(i + 0.5, color='.9', linewidth=0.5)
        # ax3.scatter(test_train_depths, np.log10(test_train_freqs), s=15, color='tab:blue')
        # ax3.scatter(test_notrain_depths, np.log10(test_notrain_freqs), s=15, color='tab:red')

        # ax1.tick_params(axis='y', labelcolor=color)

        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #
        # # color = 'tab:blue'
        # ax2.set_ylabel('types', color='tab:green')  # we already handled the x-label with ax1
        # ax2.hist(t, data2, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)

        # plt.margins(0.2)
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped

        # try:
        plt.savefig(f'stats_20_shifted_rebank_{complex}{"_lexical" if lexical else ""}.pdf', format='pdf')
        # except PermissionError:
        #     pass

        plt.show()


if __name__ == '__main__':
    import json

    from ccg.util.argparse import argparser, str2bool, load_json, main as ap_main

    argparser.add_argument('--unaries', type=int, default=0)
    argparser.add_argument('--generate-figure', action='store_true')
    argparser.add_argument('--lexical', type=str2bool, nargs='?', const=True, default=True)
    argparser.add_argument('--complex', type=str, default='depth')
    argparser.add_argument('--oov-sent-ids', nargs='+', type=load_json, default=['ccg/parser/evaluation/oov_10_sentences.json'])
    argparser.add_argument('--output-new-split', type=str, default=None)
    # argparse.argparser.add_argument('--generate-new-split', type=str, default=None)
    args = ap_main()


    # for ids in args.oov_sent_ids:
    # with open(args.oov_sent_ids) as f:
    #     oov_sent_ids = json.load(f)

    main(args.training_files, args.development_files, args.testing_files,
         unaries=args.unaries,
         generate_figure=args.generate_figure,
         lexical=args.lexical,
         complex=args.complex,
         oov_sent_ids=args.oov_sent_ids,
         # generate_new_split=args.generate_new_split,
         output_new_split=args.output_new_split,
         args=args)
