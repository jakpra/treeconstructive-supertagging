'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
from collections import defaultdict, Counter
from operator import itemgetter
import json

import pickle
import numpy as np
import random
import torch
from torch.nn.utils.rnn import pad_sequence

import tree_st.util.argparse
from .reader import AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader, DependenciesReader
from .functions import *
from ..tagger.nn import UNK, PAD, START, END


def batch_supertags(supertagger, _seq_batch, y_batch, max_sentence_length, batch_size,
                    span_batch=None, mask_batch=None, deps_batch=None,
                    load_phrasal=False, load_deps=False, train=False, lower=True, output_sentences=False):
    seq_batch = supertagger.prepare_inputs(_seq_batch, padding=max_sentence_length,
                                           train=train, lower=lower)
    y_batches, deps_batches = [], []
    for gen in supertagger.generators:
        y_batches.append(gen.prepare_outputs(y_batch, batch_size=batch_size, padding=max_sentence_length))
        if load_deps:
            deps_batches.append(gen.prepare_deps(deps_batch, padding=max_sentence_length))
    result = {'x': seq_batch, 'y': y_batches}
    if load_phrasal:
        span_batch = torch.cat(span_batch)
        mask_batch = pad_sequence(mask_batch, batch_first=True, padding_value=0.)

        result['span'] = span_batch
        result['mask'] = mask_batch

    if load_deps:
        result['dependencies'] = deps_batches

    if output_sentences:
        result['sentences'] = _seq_batch

    return result


def load_supertags(filepaths, supertagger, format='auto', ids=None,
                   load_phrasal=False, load_deps=False,
                   batch_size=1, train=False, lower=True, output_sentences=False):

    if format == 'ast':
        dr = ASTDerivationsReader
    elif format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    _seq_batch = []
    if load_phrasal:
        span_batch = []  # may or may not use these to included phrasal categories in training
        mask_batch = []  # may or may not use these to included phrasal categories in training
    y_batch = []
    if load_deps:
        deps_batch = []
    max_sentence_length = 0
    loaded_sents = 0
    loaded_toks = 0
    for filepath in filepaths:
        if load_deps:
            assert format == 'auto'
            deps_path = str(filepath).replace('AUTO', 'PARG').replace('auto', 'parg')
            iterator = zip(dr(filepath, validate=False), DependenciesReader(deps_path))
        else:
            iterator = dr(filepath, validate=False)
        for item in iterator:
            if ids is not None and item['ID'] not in ids:
                continue
            if load_deps:
                deriv, deps = item
                assert deriv['ID'] == deps['ID']
                deriv = deriv['DERIVATION']
                deriv.set_dependencies(deps['DEPENDENCIES'])
                deps_batch.append(deriv.dependencies)
            else:
                deriv = item['DERIVATION']

            bu = bottom_up(len(deriv.sentence))

            if load_phrasal:
                sentence_index_in_batch = len(seq_batch)
            _seq_batch.append(deriv.sentence)
            sentence_length = len(deriv.sentence)
            loaded_sents += 1
            loaded_toks += sentence_length
            if sentence_length > max_sentence_length:
                max_sentence_length = sentence_length
            y_seq = []
            for ((i, j), u), (args, cat), comb in sorted(deriv.categories(), key=lambda x: bu(*x[0][0])):

                if j - i == 1:
                    if u == 0:
                        y_seq.append(cat)  # TODO

                elif load_phrasal:
                    pass

                else:
                    break

            y_batch.append(y_seq)

            if batch_size is not None and len(_seq_batch) == batch_size:
                yield batch_supertags(supertagger, _seq_batch, y_batch, max_sentence_length,
                                      batch_size,
                                      span_batch=span_batch if load_phrasal else None,
                                      mask_batch=mask_batch if load_phrasal else None,
                                      deps_batch=deps_batch if load_deps else None,
                                      load_phrasal=load_phrasal, load_deps=load_deps,
                                      train=train, lower=lower,
                                      output_sentences=output_sentences)

                print(f'{loaded_sents} sentences; {loaded_toks} tokens', end='\r', file=sys.stderr)

                _seq_batch, y_batch = [], []
                if load_phrasal:
                    span_batch, mask_batch = [], []
                if load_deps:
                    deps_batch = []
                max_sentence_length = 0

    if _seq_batch:
        while batch_size is not None and len(_seq_batch) < batch_size:
            sentence_index_in_batch = len(_seq_batch)
            _seq_batch.append([])

        yield batch_supertags(supertagger, _seq_batch, y_batch, max_sentence_length,
                              batch_size,
                              span_batch=span_batch if load_phrasal else None,
                              mask_batch=mask_batch if load_phrasal else None,
                              deps_batch=deps_batch if load_deps else None,
                              load_phrasal=load_phrasal, load_deps=load_deps,
                              train=train, lower=lower,
                              output_sentences=output_sentences)

    print(f'{loaded_sents} sentences; {loaded_toks} tokens', file=sys.stderr)


def load_derivations(filepaths, batch_size=1):
    for filepath in filepaths:
        dr = AUTODerivationsReader(filepath)
        for d in dr:
            deriv = d['DERIVATION']
            ID = d['ID']
            yield ID, deriv


def get_target_vocab(filepaths, lower=True, n=0, feat_freq_cutoff=1, checkpoint=defaultdict(dict)):
    vocab = {f'{prepost}_{_n}': {} for prepost in ('pre', 'suf') for _n in range(1, n+1)}
    vocab['word'] = {}
    feat_freqs = {f'{prepost}_{_n}': Counter() for prepost in ('pre', 'suf') for _n in range(1, n+1)}
    indices = {k: len(checkpoint[k]) for k in vocab}
    for key in vocab:
        if UNK not in (checkpoint[key].keys() | vocab[key].keys()):
            vocab[key][UNK] = indices[key]
            indices[key] += 1
        if PAD not in (checkpoint[key].keys() | vocab[key].keys()):
            vocab[key][PAD] = indices[key]
            indices[key] += 1
    if START not in (checkpoint['word'].keys() | vocab['word'].keys()):
        vocab['word'][START] = indices['word']
        indices['word'] += 1
    if END not in (checkpoint['word'].keys() | vocab['word'].keys()):
        vocab['word'][END] = indices['word']
        indices['word'] += 1
    for filepath in filepaths:
        dr = AUTODerivationsReader(filepath)
        for d in dr:
            deriv = d['DERIVATION']
            for w in deriv.sentence:
                if lower:
                    w = w.lower()
                if w not in (checkpoint['word'].keys() | vocab['word'].keys()):
                    vocab['word'][w] = indices['word']
                    indices['word'] += 1
                for _n in range(1, min(n, len(w))+1):
                    feat_freqs[f'pre_{_n}'][w[:_n]] += 1
                    feat_freqs[f'suf_{_n}'][w[-_n:]] += 1

    for key in feat_freqs:
        for feat in feat_freqs[key]:
            if feat_freqs[key][feat] >= feat_freq_cutoff and feat not in (checkpoint[key].keys() | vocab[key].keys()):
                vocab[key][feat] = indices[key]
                indices[key] += 1

    return vocab


def load_pretrained_glove_pickle(path, target_vocab, emb_dim, seed=42):
    import bcolz

    np.random.seed(seed)
    vectors = bcolz.open(f'{path}.dat')[:]
    words = pickle.load(open(f'{path}_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{path}_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, (word, _) in enumerate(sorted(target_vocab.items(), key=itemgetter(1))):
        try:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,)) if word == UNK else glove[word.encode()]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    return weights_matrix


def load_pretrained_bert_jsonl(vector_path, seqs_path, layer_idx=-1):
    seq_to_ix = {}
    with open(seqs_path, 'r') as f:
        for i, line in enumerate(f):
            seq_to_ix[line] = i
    weights = []
    token_maps = []
    with open(vector_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            token_maps.append(data['token_map'])
            seq = []
            for token in data['features']:
                for layer in token['layers']:
                    if layer['index'] == layer_idx:
                        seq.append(np.array(layer['values']).astype(np.float))
                        break
                weights.append(seq)

    return seq_to_ix, weights, token_maps


if __name__ == '__main__':

    args = argparse.main()


    d = list(load_unaries(args.testing_files))

    # d = load_nodes(filepaths)
    #
    # lexicon = defaultdict(lambda: defaultdict(Counter))
    lexicon = Counter()
    for batch, _ in d:
        for (s, i, j, cat, a, l) in batch:
            lexicon[cat, l] += 1

    for l in lexicon.most_common(100):
        print('   ', l)

    # while True:
    #     q = tuple(input('Query lexicon: ').split())
    #     if not q:
    #         break
    #     print(q)
    #     for a in sorted(lexicon[q]):
    #         print(' ', a)
    #         for l in lexicon[q][a].most_common(10):
    #             print('   ', l)
