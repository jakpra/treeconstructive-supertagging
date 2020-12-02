'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import math

from collections import Counter, defaultdict

from tqdm import tqdm

from ccg.representation.category import Slashes as sl
from ccg.representation.derivation import Derivation
from ccg.util.reader import AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader, SCategoryReader
from ccg.util.functions import harmonic_mean
from ccg.util.statistics import print_counter_stats
from ccg.parser.scoring.nn import UNK


class Evaluator:
    def __init__(self, train_files, train_ids=None, dev_files=[], pred=[], gold=[], pred_deps=[], gold_deps=[],
                 max_depth=6, training_format='auto', development_format='auto', testing_format='auto'):
        self.pred_derivs = pred
        self.gold_derivs = gold
        self.pred_deps = pred_deps
        self.gold_deps = gold_deps
        self.max_depth = max_depth

        if training_format == 'ast':
            dr = ASTDerivationsReader
        elif training_format == 'stagged':
            dr = StaggedDerivationsReader
        else:
            dr = AUTODerivationsReader

        category_sentences = defaultdict(set)
        word_sentences = defaultdict(set)
        usage_sentences = defaultdict(set)
        self.n_train_sentences = 0
        self.n_train_tokens = 0
        self.train_words = Counter()
        self.train_usages = Counter()
        # self.train_atomic_categories = Counter()
        self.train_categories = Counter()
        # self.train_category_shapes = Counter()
        self.train_depths = Counter()
        # self.train_slashes = Counter()
        # self.train_tl_slashes = Counter()
        # self.train_addresses = Counter()
        for filepath in tqdm(train_files):
            ds = dr(filepath, validate=False)
            for d in ds:
                if train_ids is not None and d['ID'] not in train_ids:
                    continue

                deriv = d['DERIVATION']
                deriv = deriv.lexical_deriv()
                self.n_train_sentences += 1
                # self.train_atomic_categories.update(deriv.count_atomic_categories())
                cat_counts = deriv.count_categories()
                self.train_categories.update(cat_counts)
                for cat in cat_counts:
                    self.n_train_tokens += cat_counts[cat]
                    category_sentences[cat].add(d['ID'])
                # self.train_category_shapes.update(deriv.count_category_shapes())
                self.train_depths.update(deriv.count_depths())

                word_counts = deriv.count_words()
                self.train_words.update(word_counts)
                for word in word_counts:
                    word_sentences[word].add(d['ID'])

                usage_counts = deriv.count_usages()
                self.train_usages.update(usage_counts)
                for usage in usage_counts:
                    usage_sentences[usage].add(d['ID'])
                # self.train_slashes.update(deriv.count_slashes())
                # self.train_tl_slashes.update(deriv.count_tl_slashes())
                # self.train_addresses.update(deriv.count_addresses(labels=(sl.F, sl.B)))

            ds.close()

        # self.gt10 = dict(self.train_categories.most_common(425))
        self.gt10 = {k: v for k, v in self.train_categories.items() if v >= 10}

        # cat_freq_iter = iter(self.train_categories.most_common(len(self.train_categories)))
        # self.freq_cats = defaultdict(Counter)
        # for ex in range(math.floor(math.log10(self.train_categories.most_common(1)[0][1])), -1, -1):
        #     theta = 10 ** ex
        #     while True:
        #         try:
        #             cat, freq = next(cat_freq_iter)
        #         except StopIteration:
        #             break
        #         if freq >= theta:
        #             self.freq_cats[ex][cat] = freq
        #         else:
        #             break
        self.freq_cats = defaultdict(Counter)
        self.freq_cats[0] = Counter()
        self.depth_cats = defaultdict(Counter)
        for cat in self.train_categories:
            floor_log_freq = math.floor(math.log10(self.train_categories[cat]))
            self.freq_cats[10**floor_log_freq if floor_log_freq >= 1 else self.train_categories[cat]][cat] = self.train_categories[cat]
            self.depth_cats[cat.depth()][cat] = self.train_categories[cat]

        self.train_freq_sents = defaultdict(set)
        for k in sorted(self.freq_cats.keys()):
            for cat in self.freq_cats[k]:
                for _k in self.freq_cats:
                    if k <= _k:
                        self.train_freq_sents[_k] |= category_sentences[cat]
            # self.train_freq_sents[k] = len(freq_sents[k])

        self.freq_words = defaultdict(Counter)
        self.freq_words[0] = Counter()
        for word in self.train_words:
            floor_log_freq = math.floor(math.log10(self.train_words[word]))
            self.freq_words[10 ** floor_log_freq if floor_log_freq >= 1 else self.train_words[word]][word] = \
            self.train_words[word]

        self.train_word_freq_sents = defaultdict(set)
        for k in sorted(self.freq_words.keys()):
            for word in self.freq_words[k]:
                for _k in self.freq_words:
                    if k <= _k:
                        self.train_word_freq_sents[_k] |= word_sentences[word]

        self.freq_usages = defaultdict(Counter)
        self.freq_usages[0] = Counter()
        for usage in self.train_usages:
            floor_log_freq = math.floor(math.log10(self.train_usages[usage]))
            self.freq_usages[10 ** floor_log_freq if floor_log_freq >= 1 else self.train_usages[usage]][usage] = \
            self.train_usages[usage]

        self.train_usage_freq_sents = defaultdict(set)
        for k in sorted(self.freq_usages.keys()):
            for usage in self.freq_usages[k]:
                for _k in self.freq_usages:
                    if k <= _k:
                        self.train_usage_freq_sents[_k] |= usage_sentences[usage]

    def get_train_freq_sents(self, n):
        result = set()
        for freq in sorted(self.train_freq_sents):
            if freq < n:
                result |= self.train_freq_sents[freq]
            else:
                break
        return sorted(result)

    def eval_supertags(self):
        '''
        Evaluate...
        ...on unseen words, tags, usages
        ...on atomic and complex sub-categories
        ...w.r.t. functionality (depth >= 1, slash direction, functionality of argument and result)
        ...w.r.t. complexity of sub-categories for training and prediction
        ...partial credit using tree kernels?
        '''
        # TODO: create Counters for words, tags, usages seen in training (and dev?)
        # TODO: create Counters for depth (res, arg), slashes, ... (reuse analyze.py)
        # TODO: pivot tag accuracy by these Counters

        total = 0
        cat_tp = Counter()
        cat_fp = Counter()
        cat_fn = Counter()
        word_tp = Counter()
        word_fp = Counter()
        word_fn = Counter()
        usage_tp = Counter()
        usage_fp = Counter()
        usage_fn = Counter()
        cat_tp_425 = Counter()
        cat_fp_425 = Counter()
        cat_fn_425 = Counter()
        word_tp_425 = Counter()
        word_fp_425 = Counter()
        word_fn_425 = Counter()
        usage_tp_425 = Counter()
        usage_fp_425 = Counter()
        usage_fn_425 = Counter()
        errors = {'correct': [],
                  'attr_err': [],
                  'slash_err': [],
                  'atom_err': [],
                  # 'consistent': [],
                  'corr_struct': [],
                  'wrong': [],
                  'unk': [],
                  'invalid': []}
        consistent = {'attr_err': [],
                      'slash_err': [],
                      'atom_err': [],
                      'corr_struct': [],
                      'other': []}
        invented_errors = {'correct': [],
                           'attr_err': [],
                           'slash_err': [],
                           'atom_err': [],
                           # 'consistent': [],
                           'corr_struct': [],
                           'wrong': [],
                           'unk': [],
                           'invalid': []}
        invented_consistent = {'attr_err': [],
                               'slash_err': [],
                               'atom_err': [],
                               'corr_struct': [],
                               'other': []}
        category_sentences = defaultdict(set)
        word_sentences = defaultdict(set)
        usage_sentences = defaultdict(set)
        n_test_sentences = 0
        for i, (pred_deriv, gold_deriv) in enumerate(zip(self.pred_derivs, self.gold_derivs)):
            n_test_sentences += 1
            gold_lex = gold_deriv.get_lexical()
            # gold_deriv = gold_deriv.lexical_deriv()
            pred_lex = pred_deriv.get_lexical()
            # pred_deriv = pred_deriv.lexical_deriv()
            assert len(pred_lex) == len(gold_lex), (pred_lex, gold_lex)
            for j, (pred, gold) in enumerate(zip(pred_lex, gold_lex)):
                total += 1
                category_sentences[gold.category1].add(i)
                word_sentences[gold.word].add(i)
                usage_sentences[gold.word, gold.category1].add(i)
                if gold.category1.equals(pred.category1):
                    cat_tp[gold.category1] += 1
                    word_tp[gold.word] += 1
                    usage_tp[gold.word, gold.category1] += 1
                    if gold.category1 in self.gt10:
                        cat_tp_425[gold.category1] += 1
                        word_tp_425[gold.word] += 1
                        usage_tp_425[gold.word, gold.category1] += 1
                    if pred.category1 not in self.train_categories:
                        invented_errors['correct'].append((i, j))
                    errors['correct'].append((i, j))
                else:
                    cat_fp[pred.category1] += 1
                    word_fp[gold.word] += 1
                    usage_fp[gold.word, pred.category1] += 1
                    if pred.category1 in self.gt10:
                        cat_fp_425[pred.category1] += 1
                        word_fp_425[gold.word] += 1
                        usage_fp_425[gold.word, pred.category1] += 1

                    cat_fn[gold.category1] += 1
                    word_fn[gold.word] += 1
                    usage_fn[gold.word, gold.category1] += 1
                    if gold.category1 in self.gt10:
                        cat_fn_425[gold.category1] += 1
                        word_fn_425[gold.word] += 1
                        usage_fn_425[gold.word, gold.category1] += 1

                    is_consistent = self.consistent(i, j)
                    if pred.category1 not in self.train_categories:
                        if pred.category1 is None or '-ERR-' in pred.category1.root:  # TODO: separate category for None?
                            invented_errors['invalid'].append((i, j))
                        elif pred.category1.root == '-UNKNOWN-':
                            invented_errors['unk'].append((i, j))
                        else:
                            if self.equal_but_attr(pred.category1, gold.category1):
                                invented_errors['attr_err'].append((i, j))
                                if is_consistent:
                                    invented_consistent['attr_err'].append((i, j))
                            elif self.equal_but_n_slashes(pred.category1, gold.category1):
                                invented_errors['slash_err'].append((i, j))
                                if is_consistent:
                                    invented_consistent['slash_err'].append((i, j))
                            elif self.equal_but_n_atoms(pred.category1, gold.category1):
                                invented_errors['atom_err'].append((i, j))
                                if is_consistent:
                                    invented_consistent['atom_err'].append((i, j))
                            elif self.same_shape(pred.category1, gold.category1):
                                invented_errors['corr_struct'].append((i, j))
                                if is_consistent:
                                    invented_consistent['corr_struct'].append((i, j))
                            else:
                                invented_errors['wrong'].append((i, j))
                                if is_consistent:
                                    invented_consistent['other'].append((i, j))

                    if '-ERR-' in pred.category1.root:
                        errors['invalid'].append((i, j))
                    elif pred.category1.root == '-UNKNOWN-':
                        errors['unk'].append((i, j))
                    else:
                        if self.equal_but_attr(pred.category1, gold.category1):
                            errors['attr_err'].append((i, j))
                            if is_consistent:
                                consistent['attr_err'].append((i, j))
                        elif self.equal_but_n_slashes(pred.category1, gold.category1):
                            errors['slash_err'].append((i, j))
                            if is_consistent:
                                consistent['slash_err'].append((i, j))
                        elif self.equal_but_n_atoms(pred.category1, gold.category1):
                            errors['atom_err'].append((i, j))
                            if is_consistent:
                                consistent['atom_err'].append((i, j))
                        elif self.same_shape(pred.category1, gold.category1):
                            errors['corr_struct'].append((i, j))
                            if is_consistent:
                                consistent['corr_struct'].append((i, j))
                        else:
                            errors['wrong'].append((i, j))
                            if is_consistent:
                                consistent['other'].append((i, j))

        overall_acc = sum(cat_tp.values()) / total
        gt10_correct = sum(cat_tp_425.values())
        gt10_fn = sum(cat_fn_425.values())
        gt10_fp = sum(cat_fp_425.values())
        gt10_rec = (gt10_correct / (gt10_correct + gt10_fn)) if gt10_correct + gt10_fn > 0 else 0.
        gt10_prec = (gt10_correct / (gt10_correct + gt10_fp)) if gt10_correct + gt10_fp > 0 else 0.

        gold_cats = cat_tp.keys() | cat_fn.keys()
        pred_cats = cat_tp.keys() | cat_fp.keys()
        rec_by_cat = Counter({cat: (cat_tp[cat] / (cat_tp[cat] + cat_fn[cat])) if cat_tp[cat] + cat_fn[cat] > 0 else 0. for cat in gold_cats})
        prec_by_cat = Counter({cat: (cat_tp[cat] / (cat_tp[cat] + cat_fp[cat])) if cat_tp[cat] + cat_fp[cat] > 0 else 0. for cat in pred_cats})

        freq_to_tp = Counter()
        freq_to_fn = Counter()
        freq_to_fp = Counter()
        rec_by_freq = {}
        prec_by_freq = {}
        test_freq_sents = defaultdict(set)
        for k, cat_freqs in self.freq_cats.items():
            tp = sum(cat_tp[cat] for cat in self.freq_cats[k])
            fn = sum(cat_fn[cat] for cat in self.freq_cats[k])
            fp = sum(cat_fp[cat] for cat in self.freq_cats[k])
            freq_to_tp[k] = tp
            freq_to_fn[k] = fn
            freq_to_fp[k] = fp
            rec_by_freq[k] = (tp / (tp + fn)) if tp + fn > 0 else 0.
            prec_by_freq[k] = (tp / (tp + fp)) if tp + fp > 0 else 0.

            for cat in self.freq_cats[k]:
                for _k in self.freq_cats:
                    if k <= _k:
                        test_freq_sents[_k] |= category_sentences[cat]
            # test_freq_sents[k] = len(freq_sents)

        unseen_cats = {cat for cat in gold_cats | pred_cats if cat not in self.train_categories}
        tp = sum(cat_tp[cat] for cat in unseen_cats)
        fn = sum(cat_fn[cat] for cat in unseen_cats)
        fp = sum(cat_fp[cat] for cat in unseen_cats)
        freq_to_tp[0] = tp
        freq_to_fn[0] = fn
        freq_to_fp[0] = fp
        rec_by_freq[0] = (tp / (tp + fn)) if tp + fn > 0 else 0.
        prec_by_freq[0] = (tp / (tp + fp)) if tp + fp > 0 else 0.

        for cat in unseen_cats:
            for _k in self.freq_cats:
                test_freq_sents[_k] |= category_sentences[cat]

        # WORDS ############################################################################

        gold_words = word_tp.keys() | word_fn.keys()
        pred_words = word_tp.keys() | word_fp.keys()
        rec_by_word = Counter(
            {word: (word_tp[word] / (word_tp[word] + word_fn[word])) if word_tp[word] + word_fn[word] > 0 else 0. for word in
             gold_words})
        prec_by_word = Counter(
            {word: (word_tp[word] / (word_tp[word] + word_fp[word])) if word_tp[word] + word_fp[word] > 0 else 0. for word in
             pred_words})

        word_freq_to_tp = Counter()
        word_freq_to_fn = Counter()
        word_freq_to_fp = Counter()
        word_rec_by_freq = {}
        word_prec_by_freq = {}
        test_word_freq_sents = defaultdict(set)
        for k, word_freqs in self.freq_words.items():
            tp = sum(word_tp[word] for word in self.freq_words[k])
            fn = sum(word_fn[word] for word in self.freq_words[k])
            fp = sum(word_fp[word] for word in self.freq_words[k])
            word_freq_to_tp[k] = tp
            word_freq_to_fn[k] = fn
            word_freq_to_fp[k] = fp
            word_rec_by_freq[k] = (tp / (tp + fn)) if tp + fn > 0 else 0.
            word_prec_by_freq[k] = (tp / (tp + fp)) if tp + fp > 0 else 0.

        for word in self.freq_words[k]:
            for _k in self.freq_words:
                if k <= _k:
                    test_word_freq_sents[_k] |= word_sentences[word]

        unseen_words = {word for word in gold_words | pred_words if word not in self.train_words}
        tp = sum(word_tp[word] for word in unseen_words)
        fn = sum(word_fn[word] for word in unseen_words)
        fp = sum(word_fp[word] for word in unseen_words)
        word_freq_to_tp[0] = tp
        word_freq_to_fn[0] = fn
        word_freq_to_fp[0] = fp
        word_rec_by_freq[0] = (tp / (tp + fn)) if tp + fn > 0 else 0.
        word_prec_by_freq[0] = (tp / (tp + fp)) if tp + fp > 0 else 0.

        for word in unseen_words:
            for _k in self.freq_words:
                test_word_freq_sents[_k] |= word_sentences[word]

        # USAGES ############################################################################

        gold_usages = usage_tp.keys() | usage_fn.keys()
        pred_usages = usage_tp.keys() | usage_fp.keys()
        rec_by_usage = Counter(
            {usage: (usage_tp[usage] / (usage_tp[usage] + usage_fn[usage])) if usage_tp[usage] + usage_fn[usage] > 0 else 0. for usage in
             gold_usages})
        prec_by_usage = Counter(
            {usage: (usage_tp[usage] / (usage_tp[usage] + usage_fp[usage])) if usage_tp[usage] + usage_fp[usage] > 0 else 0. for usage in
             pred_usages})

        usage_freq_to_tp = Counter()
        usage_freq_to_fn = Counter()
        usage_freq_to_fp = Counter()
        usage_rec_by_freq = {}
        usage_prec_by_freq = {}
        test_usage_freq_sents = defaultdict(set)
        for k, usage_freqs in self.freq_usages.items():
            tp = sum(usage_tp[usage] for usage in self.freq_usages[k])
            fn = sum(usage_fn[usage] for usage in self.freq_usages[k])
            fp = sum(usage_fp[usage] for usage in self.freq_usages[k])
            usage_freq_to_tp[k] = tp
            usage_freq_to_fn[k] = fn
            usage_freq_to_fp[k] = fp
            usage_rec_by_freq[k] = (tp / (tp + fn)) if tp + fn > 0 else 0.
            usage_prec_by_freq[k] = (tp / (tp + fp)) if tp + fp > 0 else 0.

        for usage in self.freq_usages[k]:
            for _k in self.freq_usages:
                if k <= _k:
                    test_usage_freq_sents[_k] |= usage_sentences[usage]

        unseen_usages = {usage for usage in gold_usages | pred_usages if usage not in self.train_usages}
        tp = sum(usage_tp[usage] for usage in unseen_usages)
        fn = sum(usage_fn[usage] for usage in unseen_usages)
        fp = sum(usage_fp[usage] for usage in unseen_usages)
        usage_freq_to_tp[0] = tp
        usage_freq_to_fn[0] = fn
        usage_freq_to_fp[0] = fp
        usage_rec_by_freq[0] = (tp / (tp + fn)) if tp + fn > 0 else 0.
        usage_prec_by_freq[0] = (tp / (tp + fp)) if tp + fp > 0 else 0.

        for usage in unseen_usages:
            for _k in self.freq_usages:
                test_usage_freq_sents[_k] |= usage_sentences[usage]

        # DEPTHS ############################################################################

        depth_to_tp = defaultdict(list)
        depth_to_fn = defaultdict(list)
        depth_to_fp = defaultdict(list)
        for cat in cat_tp:
            depth_to_tp[cat.depth()].append(cat)
        for cat in cat_fn:
            depth_to_fn[cat.depth()].append(cat)
        for cat in cat_fp:
            depth_to_fp[cat.depth()].append(cat)

        rec_by_depth = {}
        prec_by_depth = {}
        for d in range(self.max_depth + 1):
            tp = sum(cat_tp[cat] for cat in depth_to_tp[d])
            fn = sum(cat_fn[cat] for cat in depth_to_fn[d])
            fp = sum(cat_fp[cat] for cat in depth_to_fp[d])
            depth_to_tp[d] = tp
            depth_to_fn[d] = fn
            depth_to_fp[d] = fp
            rec_by_depth[d] = (tp / (tp + fn)) if tp + fn > 0 else 0.
            prec_by_depth[d] = (tp / (tp + fp)) if tp + fp > 0 else 0.

        unseen_depths = {d for d in depth_to_tp.keys() | depth_to_fn.keys() if d not in self.train_depths}

        print('# sentences train:', f'{self.n_train_sentences:10.0f}')
        print('# tokens train:', f'{self.n_train_tokens:10.0f}')
        print('# sentences test:', f'{n_test_sentences:10.0f}')
        print('# tokens test:', f'{total:10.0f}')

        print('\noverall acc:', f'{100*overall_acc:6.2f}')

        print('\n>= 10 p:', f'{100*gt10_prec:6.2f}')
        print('>= 10 r:', f'{100*gt10_rec:6.2f}')
        print('>= 10 f:', f'{100*harmonic_mean(gt10_rec, gt10_prec):6.2f}')

        print('\nby category frequency (each bin i contains items with n_i <= frequency < n_{i+1})')
        # max_freq = math.floor(math.log10(self.train_categories.most_common(1)[0][1]))
        freq_keys = sorted(self.freq_cats.keys(), reverse=True)
        print(' n             |', ' | '.join([f'{k:10.0f}' for k in freq_keys]))
        print(' # train types |', ' | '.join([f'{len(self.freq_cats[k]):10.0f}' for k in freq_keys]))
        print(' # train sents |', ' | '.join([f'{len(self.train_freq_sents[k]):10.0f}' for k in freq_keys]))
        print(' # train toks  |', ' | '.join([f'{sum(self.freq_cats[k].values()):10.0f}' for k in freq_keys]))
        print(' # gold sents  |', ' | '.join([f'{len(test_freq_sents[k]):10.0f}' for k in freq_keys]))
        print(' # gold toks   |', ' | '.join([f'{freq_to_tp[k]+freq_to_fn[k]:10.0f}' for k in freq_keys]))
        print(' # pred toks   |', ' | '.join([f'{freq_to_tp[k]+freq_to_fp[k]:10.0f}' for k in freq_keys]))
        print(' # correct     |', ' | '.join([f'{freq_to_tp[k]:10.0f}' for k in freq_keys]))
        print(' p             |', ' | '.join([f'{100 * prec_by_freq[k]:10.2f}' for k in freq_keys]))
        print(' r             |', ' | '.join([f'{100 * rec_by_freq[k]:10.2f}' for k in freq_keys]))
        print(' f             |', ' | '.join([f'{100 * harmonic_mean(prec_by_freq[k], rec_by_freq[k]):10.2f}' for k in freq_keys]))

        print(f'\nby depth (unseen depths: {sorted(unseen_depths)})')
        print(' d             |', ' | '.join([f'{d:10.0f}' for d in range(self.max_depth + 1)]))
        print(' # train types |', ' | '.join([f'{len(self.depth_cats[d]):10.0f}' for d in range(self.max_depth + 1)]))
        print(' # train toks  |', ' | '.join([f'{self.train_depths[d]:10.0f}' for d in range(self.max_depth + 1)]))
        print(' # gold  toks  |', ' | '.join([f'{depth_to_tp[d]+depth_to_fn[d]:10.0f}' for d in range(self.max_depth + 1)]))
        print(' # pred  toks  |', ' | '.join([f'{depth_to_tp[d]+depth_to_fp[d]:10.0f}' for d in range(self.max_depth + 1)]))
        print(' # correct     |', ' | '.join([f'{depth_to_tp[d]:10.0f}' for d in range(self.max_depth + 1)]))
        print(' p             |', ' | '.join([f'{100*prec_by_depth[d]:10.2f}' for d in range(self.max_depth + 1)]))
        print(' r             |', ' | '.join([f'{100*rec_by_depth[d]:10.2f}' for d in range(self.max_depth + 1)]))
        print(' f             |', ' | '.join([f'{100*harmonic_mean(prec_by_depth[d], rec_by_depth[d]):10.2f}' for d in
                                 range(self.max_depth + 1)]))

        print('\nby word frequency (each bin i contains items with n_i <= frequency < n_{i+1})')
        freq_keys = sorted(self.freq_words.keys(), reverse=True)
        print(' n             |', ' | '.join([f'{k:10.0f}' for k in freq_keys]))
        print(' # train types |', ' | '.join([f'{len(self.freq_words[k]):10.0f}' for k in freq_keys]))
        print(' # train sents |', ' | '.join([f'{len(self.train_word_freq_sents[k]):10.0f}' for k in freq_keys]))
        print(' # train toks  |', ' | '.join([f'{sum(self.freq_words[k].values()):10.0f}' for k in freq_keys]))
        print(' # gold sents  |', ' | '.join([f'{len(test_word_freq_sents[k]):10.0f}' for k in freq_keys]))
        print(' # gold toks   |', ' | '.join([f'{word_freq_to_tp[k] + word_freq_to_fn[k]:10.0f}' for k in freq_keys]))
        print(' # pred toks   |', ' | '.join([f'{word_freq_to_tp[k] + word_freq_to_fp[k]:10.0f}' for k in freq_keys]))
        print(' # correct     |', ' | '.join([f'{word_freq_to_tp[k]:10.0f}' for k in freq_keys]))
        print(' p             |', ' | '.join([f'{100 * word_prec_by_freq[k]:10.2f}' for k in freq_keys]))
        print(' r             |', ' | '.join([f'{100 * word_rec_by_freq[k]:10.2f}' for k in freq_keys]))
        print(' f             |',
              ' | '.join([f'{100 * harmonic_mean(word_prec_by_freq[k], word_rec_by_freq[k]):10.2f}' for k in freq_keys]))

        print('\nby usage frequency (each bin i contains items with n_i <= frequency < n_{i+1})')
        freq_keys = sorted(self.freq_usages.keys(), reverse=True)
        print(' n             |', ' | '.join([f'{k:10.0f}' for k in freq_keys]))
        print(' # train types |', ' | '.join([f'{len(self.freq_usages[k]):10.0f}' for k in freq_keys]))
        print(' # train sents |', ' | '.join([f'{len(self.train_usage_freq_sents[k]):10.0f}' for k in freq_keys]))
        print(' # train toks  |', ' | '.join([f'{sum(self.freq_usages[k].values()):10.0f}' for k in freq_keys]))
        print(' # gold sents  |', ' | '.join([f'{len(test_usage_freq_sents[k]):10.0f}' for k in freq_keys]))
        print(' # gold toks   |', ' | '.join([f'{usage_freq_to_tp[k] + usage_freq_to_fn[k]:10.0f}' for k in freq_keys]))
        print(' # pred toks   |', ' | '.join([f'{usage_freq_to_tp[k] + usage_freq_to_fp[k]:10.0f}' for k in freq_keys]))
        print(' # correct     |', ' | '.join([f'{usage_freq_to_tp[k]:10.0f}' for k in freq_keys]))
        print(' p             |', ' | '.join([f'{100 * usage_prec_by_freq[k]:10.2f}' for k in freq_keys]))
        print(' r             |', ' | '.join([f'{100 * usage_rec_by_freq[k]:10.2f}' for k in freq_keys]))
        print(' f             |',
              ' | '.join([f'{100 * harmonic_mean(usage_prec_by_freq[k], usage_rec_by_freq[k]):10.2f}' for k in freq_keys]))

        print('\nerrors')
        error_counter = Counter({k: len(v) for k, v in errors.items()})
        print_counter_stats(error_counter, 'error', len(error_counter))

        print('\nconsistent')
        consistent_counter = Counter({k: len(v) for k, v in consistent.items()})
        print_counter_stats(consistent_counter, 'consistent', len(consistent_counter))

        print('\ninvented errors')
        inv_error_counter = Counter({k: len(v) for k, v in invented_errors.items()})
        print_counter_stats(inv_error_counter, 'error', len(inv_error_counter))

        print('\ninvented consistent')
        inv_consistent_counter = Counter({k: len(v) for k, v in invented_consistent.items()})
        print_counter_stats(inv_consistent_counter, 'consistent', len(inv_consistent_counter))

        # print()
        # for k in invented:
        #     print(k)
        #     for i, j in invented[k]:
        #         deriv = self.pred_derivs[i]
        #         print('', i, j, deriv.get_node(j, j+1).category1, deriv.sentence[j], deriv.sentence)

        print(f'\nby cat')
        print('cat', '# train', '# gold', '# pred', '# correct', 'p', 'r', 'f')
        for cat in sorted(gold_cats | pred_cats, key=lambda c: (self.train_categories[c],
                                                                cat_tp[c]+cat_fn[c],
                                                                cat_tp[c]+cat_fp[c]), reverse=True):
            print(cat, f'{self.train_categories[cat]}',
                  f'{cat_tp[cat]+cat_fn[cat]}', f'{cat_tp[cat]+cat_fp[cat]}', f'{cat_tp[cat]}',
                  f'{100*prec_by_cat[cat]:6.2f}', f'{100*rec_by_cat[cat]:6.2f}',
                  f'{100*harmonic_mean(prec_by_cat[cat], rec_by_cat[cat]):6.2f}')

        return {'errors': errors, 'consistent': consistent,
                'invented_errors': invented_errors, 'invented_consistent': invented_consistent}

    def eval_dependencies(self):
        pass

    def add(self, pred, gold, pred_dep=None, gold_dep=None):
        self.pred_derivs.append(pred)
        self.gold_derivs.append(gold)
        self.pred_deps.append(pred_dep)
        self.gold_deps.append(gold_dep)

    def equal_but_attr(self, cat1, cat2):
        scr = SCategoryReader()
        _cat1 = scr.read(cat1.s_expr(), validate=False)
        _cat2 = scr.read(cat2.s_expr(), validate=False)
        return _cat1.equals(_cat2, ignore_attr=True)

    def consistent(self, i, j):
        if self.pred_deps[0] is None:
            return None

        deps = self.pred_deps[i]
        # gold_deps = self.gold_deps[i]
        # deriv = self.pred_derivs[i]
        # gds = set()
        # for dep in gold_deps:
        #     if j == dep.dep:
        #         gds.add(dep.head)
        #     elif j == dep.head:
        #         gds.add(dep.dep)
        # pds = set()
        for dep in deps:
            if j in (dep.head, dep.dep):
                # print(i, offs, deriv.get_node(offs, offs+1).category1, deriv.sentence[offs])
                # print(dep)
                # print('\n'.join(map(str, deps)))
                # print(deriv.pretty_print())
                # deriv_cat = deriv.get_node(offs, offs+1).category1
                # if offs == dep.head:
                #     dep_cat = dep.head_cat
                #     if not dep_cat.equals(deriv_cat, ignore_attr=False):
                #         print(dep_cat, deriv_cat, offs)
                #         print(str(dep))
                #         print(deriv.pretty_print())
                #         assert False
                return True
        #     if j == dep.dep:
        #         pds.add(dep.head)
        #     elif j == dep.head:
        #         pds.add(dep.dep)
        # return pds == gds
        return False

    def same_shape(self, cat1, cat2):
        return cat1.get_shape() == cat2.get_shape()
    
    def same_shape_and_slashes(self, cat1, cat2):
        return cat1.get_shape(keep_slashes=True) == cat2.get_shape(keep_slashes=True)

    def same_shape_and_atomcats(self, cat1, cat2):
        return cat1.get_shape(keep_atomcats=True) == cat2.get_shape(keep_atomcats=True)

    def equal_but_n_slashes(self, cat1, cat2, n=1):
        return self.same_shape_and_atomcats(cat1, cat2) and 0 < len(cat1 - cat2) <= n

    def equal_but_n_atoms(self, cat1, cat2, n=1):
        return self.same_shape_and_slashes(cat1, cat2) and 0 < len(cat1 - cat2) <= n


    # def finegrained_eval(self, cat1, cat2):
    #     nargs_correct = cat1.nargs() == cat2.nargs()
    #     size_correct = cat1.size() == cat2.size()
    #     slashes_correct = cat1.get_shape(keep_slashes=True) == cat2.get_shape(keep_slashes=True)
    #     shape_correct = cat1.get_shape() == cat2.get_shape()
    #     result_correct = cat1.leftmost_result() == cat2.leftmost_result()
    #     # TODO: F-scores over arguments, atomic categories, slashes
    #
    #     # TODO: do this as increasingly finegrained filters?
    #
    #     # consistent = ...  # TODO: engages in at least one dependency with its neighbors
    #
    #     # ignoring features
    #     # scr = SCategoryReader()
    #     # self_without_features = scr.read(cat1.s_expr(), validate=False).without_attr()
    #     # gold_without_features = scr.read(cat2.s_expr(), validate=False).without_attr()
    #     # result_correct_no_feat = self_without_features.leftmost_result() == gold_without_features.leftmost_result()
    #     #
    #     # # TODO: F-scores over arguments, atomic categories, slashes
