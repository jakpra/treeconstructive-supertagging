'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import os
import sys
import torch
import json

from operator import itemgetter
from collections import OrderedDict

from .nn import Scorer, Encoder, SequenceTagger, create_emb_layer, train, PAD, UNK
from .mlp import MLPDecoder
from .rnn import SeqRNNDecoder, TreeRNNDecoder, PointwiseSeqDecoder
from .addrmlp import AddrMLPDecoder
from .kogkalidis_transformer import TransformerSeqDecoder as KTransformerSeqDecoder
from .encoders.span import SpanEncoder, RNNSpanEncoder, BERTSpanEncoder, RoBERTaSpanEncoder, LookupSpanEncoder

from ..util.reader import CategoryReader
from ..util.loader import get_target_vocab, load_pretrained_bert_jsonl
from ..util.mode import Mode
from ..util.strings import slash

from ..ccg.category import Category, Slashes as sl


ATOMIC_LABELS = {
  "(S)": 0,
  "(S[dcl])": 1,
  "(S[wq])": 2,
  "(S[q])": 3,
  "(S[qem])": 4,
  "(S[em])": 5,
  "(S[bem])": 6,
  "(S[b])": 7,
  "(S[frg])": 8,
  "(S[for])": 9,
  "(S[intj])": 10,
  "(S[inv])": 11,
  "(S[to])": 12,
  "(S[pss])": 13,
  "(S[pt])": 14,
  "(S[ng])": 15,
  "(S[as])": 16,
  "(S[asup])": 17,
  "(S[poss])": 18,
  "(S[adj])": 19,
  "(NP)": 20,
  "(NP[nb])": 21,
  "(NP[expl])": 22,
  "(NP[thr])": 23,
  "(N)": 24,
  "(N[num])": 25,
  "(PP)": 26,
  "(,)": 27,
  "(.)": 28,
  "(conj)": 29,
  "(:)": 30,
  "(;)": 31,
  "(RRB)": 32,
  "(LRB)": 33,
  "(/)": 34,
  "(\\)": 35
}

ATOMIC = {'name': 'atomic',
          'tagset': 'atomic.json',
          'grow_labels': [34, 35],
          'complete': True,
          'max_address_depth': 6,
          'featurized': True,
          'attention': False}

ATOMIC_LSTM = {'name': 'atomic',
               'tagset': 'atomic.json',
               'complete': True,
               'max_address_depth': 6,
               'featurized': False,
          'attention': False}

TOP_425 = {'name': 'top_425',
           'tagset': 'top_425.json',
           'complete': False,
           'max_address_depth': 0,
           'featurized': True,
          'attention': False}

TOP_425_LSTM = {'name': 'top_425',
                'tagset': 'top_425.json',
                'complete': False,
                'max_address_depth': 0,
                'featurized': False,
          'attention': False}


class SuperTagger(SequenceTagger):
    def __init__(self, span_encoder, hidden_dims=[300, 300], dropout=[0.0, 0.0], tree_hidden_dim=100,
                 finetune=True, tasks=[ATOMIC],
                 oracle='gobal',
                 transformer_layers=12, attention_heads=12,
                 device=torch.device('cpu'), **kwargs):

        super(SequenceTagger, self).__init__(input_dim=span_encoder.hidden_dim,
                                             output_dim=0,
                                             hidden_dims=hidden_dims,
                                             dropout=dropout, **kwargs)

        self.device = device

        self.span_encoder = span_encoder
        self.finetune = finetune

        context_hidden_dim = hidden_dims[-1] if len(hidden_dims) >= 1 else span_encoder.hidden_dim
        self.tasks = tasks
        self.generators = torch.nn.ModuleList()
        for i, task in enumerate(tasks):
            task['tagset'] = f"{os.environ['CCG_TAGSETS']}/{task['tagset']}"
            with open(task['tagset']) as f:
                labels = json.load(f)
            if i == 0:
                if 'atomic_tagset' in task:
                    task['atomic_tagset'] = f"{os.environ['CCG_TAGSETS']}/{task['atomic_tagset']}"
                    with open(task['atomic_tagset']) as f:
                        self.atomic_labels = json.load(f)
                elif task['complete']:
                    self.atomic_labels = dict(labels)
                else:
                    self.atomic_labels = ATOMIC_LABELS
            if not task['complete'] and UNK not in labels:
                labels[UNK] = len(labels)
            decoder = task['decoder']
            if decoder == 'mlp':
                self.generators.append(MLPDecoder(tree_hidden_dim, context_hidden_dim,
                                                  labels, atomic_labels=self.atomic_labels,
                                                  mapped_grow_labels=task.get('mapped_grow_labels'),
                                                  max_mapped_depth=task.get('max_mapped_address_depth'),
                                                  enc_attention=task.get('enc_attention'),
                                                  dec_attention=task.get('dec_attention'),
                                                  is_sexpr=task.get('is_sexpr'),
                                                  device=device))
            elif decoder == 'addrmlp':
                self.generators.append(AddrMLPDecoder(tree_hidden_dim, context_hidden_dim,
                                                      labels, atomic_labels=self.atomic_labels,
                                                      grow_labels=task['grow_labels'],
                                                      mapped_grow_labels=task.get('mapped_grow_labels'),
                                                      max_depth=task['max_address_depth'],
                                                      max_mapped_depth=task.get('max_mapped_address_depth'),
                                                      oracle=oracle,
                                                      enc_attention=task.get('enc_attention'),
                                                      dec_attention=task.get('dec_attention'),
                                                      is_sexpr=task.get('is_sexpr'),
                                                      device=device))
            elif decoder == 'treernn':
                self.generators.append(TreeRNNDecoder(tree_hidden_dim, context_hidden_dim,
                                                      labels, atomic_labels=self.atomic_labels,
                                                      grow_labels=task['grow_labels'],
                                                      mapped_grow_labels=task.get('mapped_grow_labels'),
                                                      max_depth=task['max_address_depth'],
                                                      max_mapped_depth=task.get('max_mapped_address_depth'),
                                                      oracle=oracle,
                                                      enc_attention=task.get('enc_attention'),
                                                      dec_attention=task.get('dec_attention'),
                                                      is_sexpr=task.get('is_sexpr'),
                                                      device=device))
            elif decoder == 'rnn':
                self.generators.append(PointwiseSeqDecoder(tree_hidden_dim, context_hidden_dim,
                                                           labels,
                                                           max_size=task['max_len'],
                                                           enc_attention=task.get('enc_attention'),
                                                           dec_attention=task.get('dec_attention'),
                                                           is_sexpr=task.get('is_sexpr'),
                                                           device=device))
            elif decoder == 'k+19':
                self.generators.append(KTransformerSeqDecoder(tree_hidden_dim, context_hidden_dim,
                                                              labels,
                                                              max_len=task['max_len'],
                                                              transformer_layers=transformer_layers,
                                                              attention_heads=attention_heads,
                                                              activation='gelu', is_sexpr=task.get('is_sexpr', True),
                                                              with_sep=(not task.get('no_sep')),
                                                              dropout=dropout[0] if len(dropout) >= 1 else 0.0,
                                                              device=device))
            else:
                raise NotImplementedError(f'decoder={decoder}')

    def forward(self, x, ys=None, output_attention_weights=False, word_mask=None):
        if self.finetune:
            x, enc_cls = self.span_encoder(x, word_mask=word_mask, device=self.device)  # (batch, seq, in)
        else:
            with torch.no_grad():
                x, enc_cls = self.span_encoder(x, word_mask=word_mask, device=self.device)  # (batch, seq, in)
        if ys is None:
            ys = [None] * len(self.generators)

        result = self.generators[0](x, y=ys[0], output_attention_weights=output_attention_weights,
                                    word_mask=word_mask, enc_cls=enc_cls)

        del x, ys
        return result

    def prepare_output(self, y, device=torch.device('cpu')):  # y is a nodeblock dict
        return torch.tensor([self.out_to_ix[y.get(a, PAD)] for a in range(self.address_dim)], dtype=torch.long, device=device)

    def scores(self, y):
        '''
        :param y: an output tensor of shape (batch, seq, address, label)
        :return: a 3-fold nested list containing an OrderedDict({label: score}) for each address
        '''
        seq_len = y.size(1)
        scores = y.view(-1, seq_len, self.address_dim, self.label_dim)  # (batch, seq, address, label)
        return [[[OrderedDict(((self.ix_to_out[ix], score) for ix, score in enumerate(address)))
                  for address in word]
                 for word in seq]
                for seq in scores]

    def to_dict(self):
        result = super().to_dict()
        result.update({'tasks': self.tasks})
        return result


def build_supertagger(args, training_files=[], additional_files=[], # atomic_labels=ATOMIC_LABELS,
                      model_checkpoint={}, emb_weights=None, feat_emb_weights=None):
    assert training_files or 'span_encoder' in model_checkpoint, \
        'If training_files is not provided, a pretrained span_encoder must already exist in model_checkpoint.'
    if 'span_encoder' in model_checkpoint:
        if args.span_encoder in ('bert', 'roberta'):
            emb = True
        else:
            from ..util.loader import load_pretrained_glove_pickle

            emb_dim = model_checkpoint['span_encoder']['embedding_dim']
            vocab = model_checkpoint['span_encoder']['to_ix']
            vocab_sizes = {k: len(v) for k, v in vocab.items()}
            assert vocab_sizes == model_checkpoint['span_encoder']['vocab_sizes']
            if emb_weights is None:
                emb_weights = load_pretrained_glove_pickle(args.word_vectors, vocab['word'], args.embedding_dim)
                emb_weights = torch.from_numpy(emb_weights).to(torch.float32)
            assert emb_weights.size(0) == vocab_sizes['word']
            assert emb_weights.size(1) == emb_dim

            if feat_emb_weights is None:
                feat_emb_weights = {}
                for key in vocab:
                    if key != 'word':
                        feat_emb_weights[key] = torch.randn(vocab_sizes[key],
                                                            args.feat_embedding_dim,
                                                            dtype=torch.float32)
                        assert feat_emb_weights[key].size(1) == model_checkpoint['span_encoder']['feat_embedding_dim']

            new_vocab = get_target_vocab(additional_files, checkpoint=vocab, n=0)
            if new_vocab and set(new_vocab['word']) - set(vocab['word']):
                new_weights = load_pretrained_glove_pickle(args.word_vectors, new_vocab['word'], args.embedding_dim)
                new_weights = torch.from_numpy(new_weights).to(emb_weights)
                emb_weights = torch.cat((emb_weights, new_weights), dim=0)
                assert emb_weights.size(0) == vocab_sizes['word'] + new_weights.size(0), \
                    f'{emb_weights.size(0)} == {vocab_sizes["word"]} + {new_weights.size(0)}'
                assert emb_weights.size(1) == emb_dim, f'{emb_weights.size(1)} == {emb_dim}'

                vocab['word'].update(new_vocab['word'])  # TODO


            emb, vocab_size, emb_dim = create_emb_layer(emb_weights,
                                                        trainable=args.mode == Mode.train and args.finetune)
            feat_emb = torch.nn.ModuleDict()
            for key in feat_emb_weights:
                feat_emb[key], _, _ = create_emb_layer(feat_emb_weights[key],
                                                       trainable=args.mode == Mode.train and args.finetune)

            print(vocab_size, 'words in vocabulary', file=sys.stderr)
    else:
        if args.span_encoder in ('bert', 'roberta'):
            emb = False
        else:
            vocab = get_target_vocab(training_files, feat_freq_cutoff=args.feat_freq_cutoff, n=args.feat_chars)
            vocab_sizes = {k: len(v) for k, v in vocab.items()}
            weights = load_pretrained_glove_pickle(args.word_vectors, vocab['word'], args.embedding_dim)
            weights = torch.from_numpy(weights).to(torch.float32)

            feat_emb = torch.nn.ModuleDict()
            for key in vocab:
                if key != 'word':
                    feat_weights = torch.randn(vocab_sizes[key], args.feat_embedding_dim, dtype=torch.float32)
                    feat_emb[key], _, _ = create_emb_layer(feat_weights, trainable=args.mode == Mode.train and args.finetune)

            if additional_files:
                add_vocab = get_target_vocab(additional_files, checkpoint=vocab, n=0)
                vocab['word'].update(add_vocab['word'])
                add_weights = load_pretrained_glove_pickle(args.word_vectors, add_vocab['word'], args.embedding_dim)
                add_weights = torch.from_numpy(add_weights).to(torch.float32)
                weights = torch.cat((weights, add_weights), dim=0)

            emb, vocab_size, emb_dim = create_emb_layer(weights, trainable=args.mode == Mode.train and args.finetune)
            assert weights.size(0) == vocab_sizes['word'] + add_weights.size(0), \
                f'{emb_weights.size(0)} == {vocab_sizes["word"]} + {add_weights.size(0)}'
            assert args.embedding_dim == emb_dim

            print(vocab_size, 'words in vocabulary', file=sys.stderr)

    if args.span_encoder == 'rnn':  # TODO: and maybe LookupBERT?
        span_enc = RNNSpanEncoder(emb_dim, hidden_dims=args.span_hidden_dims, vocab=vocab, embed=emb,
                                  feat_embedding_dim=args.feat_embedding_dim,
                                  feat_chars=args.feat_chars, feat_embed=feat_emb,
                                  emb_dropout=args.embedding_dropout,
                                  bidirectional=args.bidirectional, dropout=args.span_dropout)

    elif args.span_encoder == 'transformer':
        span_enc = TransformerSpanEncoder(emb_dim, hidden_dim=args.span_hidden_dims[0], vocab=vocab, embed=emb,
                                          feat_embedding_dim=args.feat_embedding_dim,
                                          feat_chars=args.feat_chars, feat_embed=feat_emb,
                                          emb_dropout=args.embedding_dropout, dropout=args.span_dropout[0],
                                          n_layers=1, n_heads=3, activation='gelu')

    elif args.span_encoder == 'bert':
        span_enc = BERTSpanEncoder(emb=emb, pretrained=args.pretrained_bert)

    elif args.span_encoder == 'roberta':
        span_enc = RoBERTaSpanEncoder(emb=emb, pretrained=args.pretrained_bert)

    tasks = []
    for task in args.tasks:
        with open(f'{task}.task.json') as f:
            tasks.append(json.load(f))

    s = SuperTagger(span_enc, hidden_dims=args.hidden_dims, dropout=args.dropout,
                    finetune=args.finetune if hasattr(args, 'finetune') else None,
                    tasks=tasks,
                    tree_hidden_dim=args.tree_hidden_dim,
                    oracle=args.teacher_forcing if hasattr(args, 'teacher_forcing') else None,
                    transformer_layers=args.transformer_layers, attention_heads=args.attention_heads,
                    device=args.device)
    return s
