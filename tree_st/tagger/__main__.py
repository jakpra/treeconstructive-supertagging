'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
from operator import itemgetter
from collections import Counter
import json
import re
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

import tree_st.util.argparse as ap
from ..util.loader import load_derivations, load_supertags
from ..util.mode import Mode
from ..util.functions import bottom_up, top_down, left_corner
from ..util.statistics import print_counter_stats
from ..util.reader import CategoryReader, DerivationsReader, AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader

from ..ccg.category import ATOMIC, ATOMIC_WITH_ATTR, AtomicCategories as ac
from ..ccg.derivation import Derivation

from .eval.evaluation import Evaluator
from .supertagger import build_supertagger, PAD
from .nn import train, OPTIM, load_model_states


args = ap.main()

if not args.model_exists:
    assert args.mode == Mode.train, 'If model does not exist, mode has to be \'train\'.'

torch.manual_seed(args.seed)


loader = load_supertags
builder = build_supertagger

print('Device selected:', args.device, file=sys.stderr)

if args.mode == Mode.train:
    if args.model_exists:
        print('Found model. Loading parameters...', file=sys.stderr)
        checkpoint = torch.load(f'{args.model}.pt', map_location=args.device)

        checkpoint['model_state_dict'] = load_model_states(checkpoint['model_state_dict'])
        feat_emb_weights = {}
        if args.span_encoder not in ('bert', 'roberta'):
            for key in list(checkpoint['model_state_dict'].keys()):
                mo = re.match('span_encoder\.feat_embeddings\.([^.]+)\.weight', key)
                if mo:
                    feat_emb_weights[mo.group(1)] = checkpoint['model_state_dict'].pop(key)
        model = builder(args, training_files=args.training_files, additional_files=args.development_files,
                                        model_checkpoint=checkpoint['model'],
                                        emb_weights=None if args.span_encoder in ('bert', 'roberta') else checkpoint['model_state_dict'].pop('span_encoder.embeddings.weight'),
                                        feat_emb_weights=None if args.span_encoder in ('bert', 'roberta') else feat_emb_weights)

        print(model, file=sys.stderr)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=sys.stderr)
        print(sum(p.numel() for p in model.span_encoder.parameters() if p.requires_grad), ' parameters in encoder', file=sys.stderr)
        print(sum(p.numel() for p in model.generators.parameters() if p.requires_grad), ' parameters in decoder(s)', file=sys.stderr)

        optimizer_name = checkpoint['optimizer_name']
        optimizer_kwargs = checkpoint['optim_kwargs']
        epoch = checkpoint['epoch']
        dev_acc = checkpoint['dev_acc']
        dev_loss = checkpoint['dev_loss']
        print('best epoch:', epoch, file=sys.stderr)
        print('best dev acc:', dev_acc, file=sys.stderr)
        print('best dev loss:', dev_loss, file=sys.stderr)

        print('Resuming training...', file=sys.stderr)

    else:
        print('Model not found. Starting from scratch...', file=sys.stderr)
        optimizer_name = args.optimizer
        optimizer_kwargs = dict(lr=args.learning_rate, momentum=args.momentum, eps=args.epsilon,
                                weight_decay=args.decay, use_schedule=args.use_schedule,
                                pct_start=args.pct_start, anneal_strategy=args.anneal_strategy)
        epoch = 0
        dev_acc = 0.0
        dev_loss = None

        model = builder(args, training_files=args.training_files, additional_files=args.development_files)
        print(model, file=sys.stderr)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=sys.stderr)
        print(sum(p.numel() for p in model.span_encoder.parameters() if p.requires_grad), ' parameters in encoder', file=sys.stderr)
        print(sum(p.numel() for p in model.generators.parameters() if p.requires_grad), ' parameters in decoder(s)', file=sys.stderr)

    num_devices = len(args.cuda_devices)
    if args.cuda and cuda.device_count() >= num_devices > 1:
        tasks = model.tasks
        prepare_inputs = model.prepare_inputs
        span_encoder = model.span_encoder
        out_fxns = []
        for gen in model.generators:
            out_fxns.append(gen)
        to_dict = model.to_dict

        model = nn.DataParallel(model, device_ids=args.cuda_devices)
        print(cuda.device_count(), 'CUDA device(s) available', file=sys.stderr)

        model.tasks = tasks
        model.span_encoder = span_encoder
        model.prepare_inputs = prepare_inputs
        model.generators = nn.ModuleList(out_fxns)
        model.to_dict = to_dict

    model = model.to(args.device)

    param_groups = [{'params': gen.parameters(), 'lr': args.learning_rate} for gen in model.generators]
    if args.span_encoder in ('bert', 'roberta', 'albert'):
        param_groups.append({'params': model.span_encoder.parameters(), 'lr': args.bert_learning_rate})
    elif args.span_encoder in ('rnn',) and args.word_vectors is not None:
        param_groups.append({'params': model.span_encoder.embeddings.parameters(), 'lr': args.bert_learning_rate})
        param_groups.append({'params': model.span_encoder.feat_embeddings.parameters(), 'lr': args.learning_rate})
        param_groups.append({'params': model.span_encoder.rnn.parameters(), 'lr': args.learning_rate})
    optimizer = OPTIM.get(optimizer_name)(param_groups, **optimizer_kwargs)
    if args.model_exists:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)

    print('Loading training data...', file=sys.stderr)
    trainloader = list(loader(args.training_files, model,
                              format=args.training_format or args.format,
                              ids=args.training_ids,
                              batch_size=args.batch_size,
                              train=True,
                              load_deps=model.tasks[0].get('attention') and args.lambda_enc + args.lambda_dec > 0))
    print('Ok.', file=sys.stderr)

    print('Loading development data...', file=sys.stderr)
    devloader = list(loader(args.development_files, model,
                            format=args.development_format or args.format,
                            batch_size=args.batch_size))
    if len(devloader) == 0:
        devloader = None
        print('No data found.', file=sys.stderr)
    else:
        print('Ok.', file=sys.stderr)

    sys.stderr.flush()

    print('batch size:', args.batch_size, file=sys.stderr)
    print(len(trainloader), 'mini-batch(es), approx', len(trainloader) * args.batch_size, 'examples',
          file=sys.stderr)

    print('optimizer:', optimizer_name, file=sys.stderr)
    print('Training...', file=sys.stderr)
    sys.stderr.flush()

    train(model, trainloader,
          {'optimizer': optimizer, 'optimizer_name': optimizer_name, 'optimizer_kwargs': optimizer_kwargs},
          devloader=devloader, batch_size=args.batch_size, max_batches=args.max_batches,
          epochs=args.epochs, start_epoch=epoch, dev_acc=dev_acc, dev_loss=dev_loss, seed=args.seed,
          loss_fxn=args.loss_fxn,
          omega_native_atom=args.omega_native_atom, omega_atom=args.omega_atom, omega_full=args.omega_full,
          lambda_enc=args.lambda_enc, lambda_dec=args.lambda_dec,
          model=f'{args.model}.pt', n_print=args.n_print)

    del model
    torch.cuda.empty_cache()

print('Found model. Loading parameters...', file=sys.stderr)

checkpoint = torch.load(f'{args.model}.pt', map_location=args.device)

checkpoint['model_state_dict'] = load_model_states(checkpoint['model_state_dict'])
feat_emb_weights = {}
if args.span_encoder not in ('bert', 'roberta'):
    for key in list(checkpoint['model_state_dict'].keys()):
        mo = re.match('span_encoder\.feat_embeddings\.([^.]+)\.weight', key)
        if mo:
            feat_emb_weights[mo.group(1)] = checkpoint['model_state_dict'].pop(key)

model = builder(args, additional_files=args.testing_files,
                model_checkpoint=checkpoint['model'],
                emb_weights=None if args.span_encoder in ('bert', 'roberta') else checkpoint['model_state_dict'].pop('span_encoder.embeddings.weight'),
                feat_emb_weights=None if args.span_encoder in ('bert', 'roberta') else feat_emb_weights)

print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=sys.stderr)
print(sum(p.numel() for p in model.span_encoder.parameters() if p.requires_grad), ' parameters in encoder',
      file=sys.stderr)
print(sum(p.numel() for p in model.generators.parameters() if p.requires_grad), ' parameters in decoder(s)',
      file=sys.stderr)

epoch = checkpoint['epoch']
dev_acc = checkpoint['dev_acc']
dev_loss = checkpoint['dev_loss']
print('best epoch:', epoch, file=sys.stderr)
print('best dev acc:', dev_acc, file=sys.stderr)
print('best dev loss:', dev_loss, file=sys.stderr)

model.load_state_dict(checkpoint['model_state_dict'], strict=False)

num_devices = len(args.cuda_devices)
if args.cuda and cuda.device_count() >= num_devices > 1:
    tasks = model.tasks
    prepare_inputs = model.prepare_inputs
    span_encoder = model.span_encoder
    out_fxns = []
    for gen in model.generators:
        out_fxns.append(gen)
    to_dict = model.to_dict

    model = nn.DataParallel(model, device_ids=args.cuda_devices)
    print(cuda.device_count(), 'CUDA device(s) available', file=sys.stderr)

    model.tasks = tasks
    model.span_encoder = span_encoder
    model.prepare_inputs = prepare_inputs
    model.generators = nn.ModuleList(out_fxns)
    model.to_dict = to_dict

model = model.to(args.device)

model.eval()

if args.out is None:
    if args.oracle_scoring:
        args.out = 'oracle.auto'
    else:
        args.out = f'{args.model}.auto'

with open(args.out, 'w') as f:
    pass

testloader = list(loader(args.testing_files, model,
                         format=args.testing_format or args.format,
                         batch_size=args.batch_size,
                         output_sentences=True))

test_len = len(testloader)
cats = []
masks = []
xs = []
sentences = []
running_batch_time = 0.0
with torch.no_grad():
    for data in testloader:
        x, ys, s = data['x'], data['y'], data['sentences']
        batch_size, seq_len = x['word'].size(0), x['word'].size(1)
        word_mask = (x['word'] != model.span_encoder.pad_token_id)

        start_time = time.time()
        result = model(x, word_mask=word_mask)
        running_batch_time += time.time() - start_time
        y_hat, address_mask = result['y_hat'], result['mask']

        task, gen, y = model.tasks[0], model.generators[0], ys[0]
        output_dim = gen.output_dim
        address_dim = gen.address_dim

        xs.extend(x['word'].tolist())
        sentences.extend(s)

        y_hat = y_hat.view(batch_size, -1, address_dim, output_dim)
        address_mask = address_mask.view(batch_size, -1, address_dim)
        argmaxes = torch.argmax(y_hat, dim=3)

        word_mask = word_mask.to(address_mask)

        if hasattr(gen, 'address_map'):
            mask = (~word_mask).unsqueeze(-1).expand(-1, -1, address_dim) | (~address_mask)
        else:
            mask = ~address_mask

        argmaxes[mask] = gen.out_to_ix[PAD]

        masks.extend(word_mask.tolist())

        categories = gen.extract_outputs(argmaxes)
        cats.extend(categories)

        for k in x:
            if isinstance(x[k], torch.Tensor):
                x[k] = x[k].cpu()

        for k in result:
            if isinstance(result[k], torch.Tensor):
                result[k] = result[k].cpu()

        for i in range(len(ys)):
            _y = ys[i]
            if isinstance(_y, torch.Tensor):
                ys[i] = _y.cpu()

        del x, word_mask, ys, y_hat, address_mask, argmaxes
        torch.cuda.empty_cache()

print('[test summary] %.3f batches/s | %.3f expls/s' %
      (
          test_len / running_batch_time,
          (test_len * batch_size) / running_batch_time
      ),
      file=sys.stderr)

evl = Evaluator(args.training_files, train_ids=args.training_ids, max_depth=6)

cats = iter(cats)
xs = iter(xs)
masks = iter(masks)
sentences = iter(sentences)

testing_format = args.testing_format or args.format
if testing_format == 'ast':
    dr = ASTDerivationsReader
elif testing_format == 'stagged':
    dr = StaggedDerivationsReader
else:
    dr = AUTODerivationsReader

tab_out = f'{args.model}.tsv'
with open(tab_out, 'w') as f:
    f.write(f'{args.model}\n')

for filename in args.testing_files:
    ds = dr(filename)
    while True:
        try:
            deriv = ds.next()
        except StopIteration:
            break
        try:
            tags = next(cats)
            x = next(xs)
            s = next(sentences)
            mask = next(masks)
        except StopIteration:
            break
        deriv, ID = deriv['DERIVATION'], deriv['ID']
        if len(tags) != len(deriv.sentence):
            print(f'words {len(deriv.sentence)}  tags {len(tags)}', file=sys.stderr)
            print(f'sentence {s}', file=sys.stderr)
            print(f'x {x}', file=sys.stderr)
            print(f'PAD index', model.span_encoder.pad_token_id, file=sys.stderr)
            print(f'mask {mask}', file=sys.stderr)
            print(f'{deriv.sentence}\n{tags}', file=sys.stderr)
            print('\n'.join([f'{w}\t{t}' for w, t in zip(tags, deriv.sentence)]), file=sys.stderr)
            assert len(tags) == len(deriv.sentence)
        gold_lex = deriv.get_lexical()
        deriv_hat = Derivation.from_lexical(tags, gold_lex)
        evl.add(deriv_hat, deriv)
        with open(tab_out, 'a', newline='\n') as f:
            for gold_dln, pred_cat in zip(gold_lex, tags):
                f.write(f'{pred_cat}\t{int(gold_dln.category1.equals(pred_cat))}\n')
        with open(args.out, 'a', newline='\n') as f:
            f.write(f'ID={ID} PARSER={args.tasks[0]} NUMPARSE=1\n')
            f.write(f'{deriv_hat}\n')
    ds.close()

evl.eval_supertags()

print('Done.', file=sys.stderr)
