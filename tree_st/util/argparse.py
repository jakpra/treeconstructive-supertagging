'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import json
import argparse as ap
from pathlib import Path

from .mode import Mode


argparser = ap.ArgumentParser()
mode = argparser.add_subparsers(help='mode', dest='mode')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ap.ArgumentTypeError('Boolean value expected.')


def load_json(v):
    try:
        with open(v) as f:
            return json.load(f)
    except Exception as e:
        raise ap.ArgumentTypeError(e)


def parse_args():
    global argparser, mode

    argparser.add_argument('--format', type=str, default='auto')
    argparser.add_argument('-o', '--out', type=str, default=None)
    argparser.add_argument('--labels', type=str, default=None)
    argparser.add_argument('--freq-threshold', type=int, default=10)
    argparser.add_argument('-m', '--model', type=str, default='ccg-model')
    argparser.add_argument('--derivation', action='store_true', help='print derivations as they are read')
    argparser.add_argument('-i', '--interactive', action='store_true')
    argparser.add_argument('-O', '--oracle-scoring', action='store_true')
    argparser.add_argument('--oracle-structure', action='store_true')
    argparser.add_argument('--oracle-supertags', action='store_true')
    argparser.add_argument('-a', '--max-category-depth', type=int, default=6, help='maximum depth of categories')
    argparser.add_argument('-k', '--global-beam', type=int, default=None, help='log-2 beam size')
    argparser.add_argument('-K', '--local-beam', type=int, default=None, help='log-2 beam size')
    argparser.add_argument('--lbda', type=float, default=0.1, help='minimum cost / optimal heuristic factor lambda')
    argparser.add_argument('--cheap', type=float, default=1, help='cost multiplier')
    argparser.add_argument('--penalty', type=float, default=100, help='cost multiplier')
    argparser.add_argument('--high-penalty', type=float, default=1000, help='cost multiplier')

    test_files = argparser.add_argument_group('Testing files')
    test_files.add_argument('-T', '--testing-files', type=str, nargs='+', default=['sample_data/test.auto'])
    test_files.add_argument('--testing-format', type=str, default='auto')

    train_files = argparser.add_argument_group('Training files')
    train_files.add_argument('-t', '--training-files', type=str, nargs='+', default=['sample_data/train.auto'])
    train_files.add_argument('--training-format', type=str, default='auto')
    train_files.add_argument('--training-ids', type=load_json, default=None, help='json file containing list of sentence ids')

    train_files.add_argument('-D', '--development-files', type=str, nargs='+', default=['sample_data/train.auto'])
    train_files.add_argument('--development-format', type=str, default='auto')

    # learning architecture
    arch = argparser.add_argument_group('Learning Architecture')
    arch.add_argument('--span-encoder', type=str, choices=['rnn', 'transformer', 'bert', 'roberta'], default='roberta')
    arch.add_argument('--word-vectors', type=str, default='word_vectors/glove.6B/6B.50')
    arch.add_argument('--pretrained-bert', type=str, default='roberta-base', help='model identifier')
    arch.add_argument('--attention-heads', type=int, default=1)
    arch.add_argument('--transformer-layers', type=int, default=2)
    arch.add_argument('-d', '--embedding-dim', type=int, default=50)
    arch.add_argument('--feat-embedding-dim', type=int, default=12)
    arch.add_argument('--feat-chars', type=int, default=4)
    arch.add_argument('--feat-freq-cutoff', type=int, default=3)
    arch.add_argument('--embedding-dropout', type=float, default=0.2)

    arch.add_argument('--span-hidden-dims', type=int, nargs='+', default=[768, 768])
    arch.add_argument('--bidirectional', type=str2bool, nargs='?', const=True, default=True)
    arch.add_argument('--span-dropout', type=float, nargs='*', default=[0.2, 0.1])

    arch.add_argument('--hidden-dims', type=int, nargs='*', default=[])
    arch.add_argument('--dropout', type=float, nargs='*', default=[])

    arch.add_argument('--tasks', type=str, nargs='*', default=['tasks/addrmlp_att_rebank'])
    arch.add_argument('--tree-hidden-dim', type=int, default=64)
    arch.add_argument('--enc-attention', action='store_true')
    arch.add_argument('--dec-attention', action='store_true')

    arch.add_argument('-b', '--batch-size', type=int, default=1)

    arch.add_argument('--seed', type=int, default=42)

    # CUDA
    cuda = argparser.add_argument_group('CUDA')
    cuda.add_argument('--cuda', action='store_true')
    cuda.add_argument('--cuda-devices', type=int, nargs='*', default=[])


    argparser.add_argument('-n', '--n-print', type=int, default=100)

    train = mode.add_parser(Mode.train)

    # hyperparams
    hyp = train.add_argument_group('Hyperparameters')
    hyp.add_argument('-e', '--epochs', type=int, default=10)
    hyp.add_argument('--max-batches', type=int, default=None)

    hyp.add_argument('--loss-fxn', type=str, choices=['crossent', 'avg', 'all'],
                     default='crossent')
    hyp.add_argument('--teacher-forcing', type=str, choices=['global', 'dynamic_best', 'dynamic_random'],  # add local?
                     default='global')

    hyp.add_argument('--omega-native-atom', type=float, default=0.0)
    hyp.add_argument('--omega-atom', type=float, default=0.0)
    hyp.add_argument('--omega-full', type=float, default=0.0)
    hyp.add_argument('--lambda-enc', type=float, default=0.0)
    hyp.add_argument('--lambda-dec', type=float, default=0.0)

    hyp.add_argument('--optimizer', type=str, default='adamw')
    hyp.add_argument('--learning-rate', type=float, default=1e-4)
    hyp.add_argument('--bert-learning-rate', type=float, default=1e-5)
    hyp.add_argument('--momentum', type=float, default=0.7)
    hyp.add_argument('--epsilon', type=float, default=1e-6)
    hyp.add_argument('--decay', type=float, default=0.01)
    hyp.add_argument('--use-schedule', action='store_true', default=False)
    hyp.add_argument('--pct-start', type=float, default=0.3)
    hyp.add_argument('--anneal-strategy', type=str, default='cos')

    hyp.add_argument('--finetune', type=str2bool, nargs='?', const=True, default=True)

    return argparser.parse_args()


def get_filepaths_from_path(paths, filename, suffix):
    filepaths = []
    try:
        for path in paths:
            p = Path(path)
            for filepath in p.glob(f'**/{filename if filename else f"*.{suffix}"}'):
                filepaths.append(filepath)
    except AttributeError:
        pass

    return filepaths


def get_filepaths_from_glob(globs):
    filepaths = []
    try:
        p = Path()
        for glob in globs:
            for filepath in p.glob(glob):
                filepaths.append(filepath)
    except AttributeError:
        pass

    return sorted(filepaths)


def get_filepaths_from_args(args):
    model = Path(f'{args.model}.pt')
    args.model_exists = model.is_file()
    print(args, file=sys.stderr)

    args.testing_files = get_filepaths_from_glob(args.testing_files)
    args.training_files = get_filepaths_from_glob(args.training_files)
    args.development_files = get_filepaths_from_glob(args.development_files)


def main():
    args = parse_args()
    get_filepaths_from_args(args)

    import torch.cuda as cuda

    if args.cuda and cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    return args


if __name__ == '__main__':
    main()
