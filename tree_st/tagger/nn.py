'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys

import math
from operator import itemgetter
from collections import OrderedDict, Counter
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from .oracle.oracle import make_unordered_valid_loss

# import ExAssist as EA


UNK = '<UNKNOWN>'
PAD = '<PADDING>'
START = '<START>'
END = '<END>'
SEP = '<SEP>'


def create_emb_layer(weights_matrix, trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


OPTIM = {'sgd': lambda params, **kwargs: optim.SGD(params, lr=kwargs['lr'], momentum=kwargs['momentum']),
         'adam': lambda params, **kwargs: optim.Adam(params, lr=kwargs['lr'], eps=kwargs['eps'],
                                                     weight_decay=kwargs['weight_decay']),
         'adamw': lambda params, **kwargs: optim.AdamW(params, lr=kwargs['lr'], eps=kwargs['eps'],
                                                       weight_decay=kwargs['weight_decay']),
         'adagrad': optim.Adagrad}


def load_model_states(state_dict):
    # state_dict = torch.load(filename)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'O_T' in k:  # otherwise the shared trained parameters will be overwritten with untrained ones?
            continue
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # checkpoint = torch.load(args.model, map_location=device)
    # net.load_state_dict(new_state_dict)
    return new_state_dict


def compute_acc_and_loss(task, gen, y, y_hat, address_mask, word_mask, criterion, mapped_criterion,
                         batch_size, seq_len, address_dim, output_dim, loss_fxn=False, oracle=None,
                         omega_native_atom=0., omega_atom=0., omega_full=0., lambda_enc=0.1, lambda_dec=0.,
                         enc_attn=None, dec_attn=None, deps=None, dep_attn_criterion=None,
                         dep_norm=lambda t: F.normalize(t, p=1, dim=1),
                         output_correct_bool=False):
    y = y.to(y_hat).long()
    # print('y', y.size(), y[0, 3], [(i+1, gen.ix_to_out[j.item()]) for i, j in enumerate(y[0, 3])], file=sys.stderr)
    word_mask = word_mask.to(address_mask)
    words = word_mask.float().sum()

    y_hat = y_hat.view(batch_size, -1, address_dim, output_dim)
    address_mask = address_mask.view(batch_size, -1, address_dim)
    y_hat_len = y_hat.size(1)
    if y_hat_len < seq_len:
        y_hat = torch.cat([y_hat, torch.zeros(batch_size, seq_len - y_hat_len, address_dim, output_dim).to(y_hat)], dim=1)
        address_mask = torch.cat([address_mask, torch.zeros(batch_size, seq_len - y_hat_len, address_dim).to(address_mask)], dim=1)
    elif y_hat_len > seq_len:
        y_hat = y_hat[:, :seq_len]
        address_mask = address_mask[:, :seq_len]

    argmaxes = torch.argmax(y_hat, dim=3)

    categories_gold = gen.extract_outputs(y.view(batch_size, seq_len, address_dim))
    categories_hat = gen.extract_outputs(argmaxes)

    if loss_fxn in ('avg', 'all'):
        # print('compute dynamic loss')
        loss = criterion[0](y_hat, categories_hat if oracle is None else gen.extract_outputs(oracle), categories_gold) / words

    else:
        # TODO: change SEP to PAD in y, so that loss ignores it

        y_hat = y_hat.reshape(-1, output_dim)
        address_mask = address_mask.reshape(batch_size, -1, address_dim)
        y = y.view(-1)

        # native loss
        # y = y.view(-1)
        # y_hat = y_hat.transpose(1, -1).reshape(batch_size, output_dim, -1)
        # sum everything, then normalize over batch and sequence, but not over addresses
        # print(criterion)
        # print(y_hat.shape, y.shape)
        native_loss = criterion[0](y_hat, y) / words  # (batch_size * words)  # y.view(-1)
        # average over everything (incl addresses)
        native_atomic_loss = criterion[1](y_hat, y)  # y.view(-1)

        # category-level loss
        # category_loss = atomic_loss / address_dim
        # TODO: check which one of these is really correct
        # # category-level loss
        # y = y.view(-1, address_dim)
        # # y_hat = y_hat.view(-1, output_dim, address_dim)
        # y_hat = y_hat.view(-1, address_dim, output_dim).transpose(1, 2)
        # category_loss = criterion(y_hat, y) / (batch_size * seq_len)

    if hasattr(gen, 'address_map'):
        address_mask = address_mask.view(batch_size, -1, address_dim)
        mask = (~word_mask).unsqueeze(-1).expand(-1, -1, address_dim) | (~address_mask)
        argmaxes[mask] = gen.out_to_ix[PAD]

        atomic_output_dim = gen.address_map.output_dim
        atomic_address_dim = gen.address_map.address_dim

        mapped_y = gen.address_map(y.view(-1, address_dim), indices=True, argmax=True)
        # print('mapped_y', mapped_y.size(), mapped_y.view(batch_size, seq_len, atomic_address_dim)[0, :, :6], file=sys.stderr)
        # exit(0)

        # print('mapped_y', mapped_y.size(), mapped_y[3, :6], [(i+1, gen.address_map.ix_to_out[j.item()]) for i, j in enumerate(mapped_y[3])], file=sys.stderr)
        # print('y_hat', y_hat.size(), y_hat[0, 3, :6], file=sys.stderr)
        mapped_y_hat = gen.address_map(y_hat.view(-1, address_dim, output_dim), norm=True)
        # print('mapped_y_hat', mapped_y_hat.size(), mapped_y_hat[3, :6], file=sys.stderr)

        if loss_fxn not in ('avg', 'all'):
            full_loss = mapped_criterion[0](mapped_y_hat.view(-1, atomic_output_dim), mapped_y.view(-1)) / words
            atomic_loss = mapped_criterion[1](mapped_y_hat.view(-1, atomic_output_dim), mapped_y.view(-1))
            # full_loss = criterion(mapped_y_hat.view(-1, atomic_address_dim, atomic_output_dim).transpose(1, 2), mapped_y.view(-1, atomic_address_dim)) / (batch_size * seq_len * atomic_address_dim)

        # mask = mask.view(-1, atomic_address_dim)
        # print('mask', mask.size(), file=sys.stderr)

        # print('argmaxes', argmaxes.size(), argmaxes[0, 3, :6], file=sys.stderr)
        mapped_argmaxes = gen.address_map(argmaxes.view(-1, address_dim), indices=True, argmax=True).view(batch_size, -1,
                                                                                                          atomic_address_dim)
        # print('mapped_argmaxes', mapped_argmaxes.size(), mapped_argmaxes[0, :, :6], file=sys.stderr)

        correct_bool = torch.all(torch.eq(mapped_argmaxes, mapped_y.view(batch_size, -1, atomic_address_dim)), dim=2)
    else:
        full_loss = atomic_loss = 0.
        argmaxes = argmaxes.view(batch_size, -1)
        address_mask = address_mask.view(batch_size, -1)
        argmaxes[~address_mask] = gen.out_to_ix[PAD]
        y_hat_seps = (argmaxes == gen.out_to_ix[SEP]).nonzero()  # indices of separators in pred: [[b0, s0], [b1, s1], ...]
        y = y.view(batch_size, -1)
        y_seps = (y == gen.out_to_ix[SEP]).nonzero()             # indices of separators in gold
        max_words = word_mask.size(1)
        correct_bool = torch.zeros(batch_size, max_words, dtype=torch.bool).to(word_mask)
        # correct_bool = torch.eq(argmaxes, y.view(batch_size, -1, address_dim))
        last_batch = 0
        last_y_hat_sep = 0
        last_y_sep = 0
        i = 0
        y_hat_seps = iter(y_hat_seps)
        try:
            for yb, ys in y_seps:
                yb, ys = yb.item(), ys.item()
                if yb != last_batch:
                    last_y_sep = 0
                    i = 0
                if i >= max_words:
                    continue
                try:
                    yhb, yhs = next(y_hat_seps)
                    yhb, yhs = yhb.item(), yhs.item()
                    while yhb != yb:
                        yhb, yhs = next(y_hat_seps)
                        yhb, yhs = yhb.item(), yhs.item()
                except StopIteration:
                    correct_bool[yb, i] = False
                else:
                    correct_bool[yb, i] = yhs-last_y_hat_sep == ys-last_y_sep and torch.all(torch.eq(argmaxes[yhb, last_y_hat_sep:yhs], y[yb, last_y_sep:ys]))
                    last_y_hat_sep = yhs
                last_batch, last_y_sep, i = yb, ys, i+1
        except ValueError as e:
            raise ValueError(*e.args, y_hat_seps, y_seps)
        except IndexError as e:
            raise IndexError(*e.args, f'yb={yb}, last_batch={last_batch}, ys={ys}, last_y_sep={last_y_sep}, i={i}')

    category_acc = (correct_bool & word_mask).float().sum() / words

    has_enc_attn = enc_attn is not None
    has_dec_attn = dec_attn is not None

    if loss_fxn not in ('avg', 'all'):
        loss = (1. - omega_native_atom - omega_atom - omega_full) * native_loss + \
            omega_native_atom * native_atomic_loss + \
            omega_atom * atomic_loss + \
            omega_full * full_loss

    lbda = 1. - int(has_enc_attn) * lambda_enc - int(has_dec_attn) * lambda_dec
    loss = loss.clone() * lbda

    if deps is None:
        # loss += torch.sum(torch.abs(dec_attn)) / (batch_size * seq_len * address_dim)
        pass

    else:
        if has_dec_attn:
            dec_deps = torch.diagflat(torch.ones(seq_len * address_dim, dtype=torch.float32)
                                      ).view(seq_len, address_dim, seq_len, address_dim)
            dec_deps = dec_deps.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        if has_enc_attn:
            enc_deps = torch.diagflat(torch.ones(seq_len, dtype=torch.float32))
            enc_deps = enc_deps.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, address_dim, 1)

        for n, seq in enumerate(deps):
            for i, args in enumerate(seq):
                if not word_mask[n, i].item():
                    break
                for a, j, b in args:
                    if not address_mask[n, i, a].item():
                        continue
                    d_a = math.floor(math.log2(a+1))
                    p_a = (a+1) // 2 - 1
                    if has_enc_attn:
                        enc_deps[n, i, a, i] = 0.
                        enc_deps[n, i, a, j] += 1.
                        # parent slash
                        enc_deps[n, i, p_a, i] = 0.
                        enc_deps[n, i, p_a, j] += 1.
                        # children and descendents
                        for log_breadth, depth in enumerate(range(d_a+1, gen.max_depth), start=1):
                            first_addr = 2 ** depth - 1
                            any_at_depth = False
                            for c_a in range(first_addr, first_addr+2**log_breadth):
                                if address_mask[n, i, c_a].item():
                                    any_at_depth = True
                                    enc_deps[n, i, c_a, i] = 0.
                                    enc_deps[n, i, c_a, j] += 1.
                            if not any_at_depth:
                                break
                        # TODO: not sure about this one
                        # enc_deps[j, n, j, b] = 0.
                        # enc_deps[i, n, j, b] += 1.
                    if has_dec_attn:
                        d_b = math.floor(math.log2(b+1))
                        if d_b < d_a:
                            # head's attn to deps (note that key of attn has to be in first dim for KLLoss)
                            # (key_token, key_addr, batch, query_token, query_addr)
                            dec_deps[n, i, a, i, a] = 0.
                            dec_deps[n, i, a, j, b] = 1.
                        elif d_a < d_b:
                            # dep's attn to heads (note that key of attn has to be in first dim for KLLoss)
                            # (key_token, key_addr, batch, query_token, query_addr)
                            dec_deps[n, j, b, j, b] = 0.
                            dec_deps[n, j, b, i, a] = 1.

        if has_dec_attn:
            dec_deps = dec_deps.view(-1, seq_len*address_dim).to(dec_attn)
            # total_batch_size, self.address_dim, seq_len, self.address_dim
            dec_attn = dec_attn.view(-1, seq_len*address_dim)  # .permute(2, 3, 0, 1).reshape
            dec_attn_loss = dep_attn_criterion(F.log_softmax(dec_attn, dim=1), dep_norm(dec_deps))
            loss += lambda_dec * dec_attn_loss
            del dec_attn, dec_deps

        if has_enc_attn:
            enc_deps = enc_deps.view(-1, seq_len).to(enc_attn)
            # total_batch_size, self.address_dim, seq_len
            enc_attn = enc_attn.view(-1, seq_len)  # .permute(2, 0, 1).reshape
            enc_attn_loss = dep_attn_criterion(F.log_softmax(enc_attn, dim=1), dep_norm(enc_deps))
            loss += lambda_enc * enc_attn_loss
            del enc_attn, enc_deps

    result = category_acc, loss
    if output_correct_bool:
        result = (*result, (argmaxes, correct_bool, categories_hat, categories_gold))

    del word_mask, address_mask, y, y_hat

    return result


# TODO: append accs and losses to a file so they don't get overwritten by subsequent runs
def train(net, trainloader, optimizer, devloader=None,
          criterion=torch.nn.CrossEntropyLoss, dep_attn_criterion=torch.nn.KLDivLoss,
          # optimizer_name='sgd', learning_rate=0.001, momentum=0.9,
          epochs=1, start_epoch=0, dev_acc=0.0, dev_loss=None, seed=42,
          loss_fxn='crossent',
          omega_native_atom=0., omega_atom=0., omega_full=0., lambda_enc=0.1, lambda_dec=0.,
          batch_size=4, max_batches=None, n_print=100, model='ccg-glove', device='cpu', device_ids=[0]):
    # , device='cpu', device_ids=[0]):
    random.seed(seed)
    # device = device  # torch.device(f'cuda:{cuda_device}' if cuda.is_available() else 'cpu')
    #
    # _optimizer = optimizer if optimizer is not None \
    #     else OPTIM.get(optimizer_name, optim.SGD)(net.parameters(), lr=learning_rate, momentum=momentum)

    torch.autograd.set_detect_anomaly(True)

    # atomic_criterion = criterion(ignore_index=net.out_to_ix[PAD], reduction='sum')
    # category_criterion = criterion(ignore_index=net.out_to_ix[PAD], reduction='sum')
    criteria = []
    mapped_criteria = []
    dep_attn_criteria = []
    for gen in net.generators:
        # weight = torch.ones(gen.output_dim, dtype=torch.float32)
        # weight.index_fill_(0, gen.grow_label_ix, 2.)
        if loss_fxn in ('avg', 'all'):
            # print('instantiate dynamic loss')
            criteria.append((make_unordered_valid_loss(gen.out_to_ix, fxn=loss_fxn), None))
        else:
            criteria.append((criterion(ignore_index=gen.out_to_ix[PAD], reduction='sum'),
                             criterion(ignore_index=gen.out_to_ix[PAD], reduction='mean')))
        # crt = lambda x, y: torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(x, dim=1), y)
        # criteria.append((FuzzyLoss(crt, gen.output_dim+1, 0.2, gen.out_to_ix[PAD]),
        #                  FuzzyLoss(crt, gen.output_dim+1, 0.2, gen.out_to_ix[PAD])))
        # TODO: maybe indent this under the `else`?
        if hasattr(gen, 'address_map'):
            mapped_criteria.append((criterion(ignore_index=gen.address_map.out_to_ix[PAD], reduction='sum'),
                                    criterion(ignore_index=gen.address_map.out_to_ix[PAD], reduction='mean')))
        if getattr(gen, 'attention', False):
            dep_attn_criteria.append(dep_attn_criterion(reduction='batchmean'))
        else:
            dep_attn_criteria.append(None)

    best_dev_loss = dev_loss
    best_dev_acc = dev_acc
    best_epoch = start_epoch

    train_len = len(trainloader)
    if max_batches is not None:
        train_len = min(max_batches, train_len)

    if devloader is not None:
        dev_len = len(devloader)

    _optimizer = optimizer['optimizer']
    optimizer_name = optimizer['optimizer_name']
    optimizer_kwargs = optimizer['optimizer_kwargs']
    if optimizer_kwargs['use_schedule']:
        steps_per_epoch = train_len // batch_size + int(train_len % batch_size != 0)
        scheduler = optim.lr_scheduler.OneCycleLR(_optimizer,
                                                  optimizer_kwargs['lr'],
                                                  epochs=epochs,
                                                  steps_per_epoch=steps_per_epoch,
                                                  pct_start=optimizer_kwargs['pct_start'],
                                                  anneal_strategy=optimizer_kwargs['anneal_strategy'],
                                                  last_epoch=start_epoch * steps_per_epoch - 1)

    # with EA.start(_assist) as assist:
        # try:
    epoch_running_loss = 0.0
    epoch_running_acc = 0.0

    # assist.info['epoch'] = 0
    # assist.info['train_loss'] = 0
    # assist.info['train_acc'] = 0
    # assist.info['dev_loss'] = 0
    # assist.info['dev_acc'] = 0
    # assist.info['batch'] = 0
    # assist.info['batch_loss'] = 0
    # assist.info['batch_acc'] = 0
    # assist.info['ex_per_s'] = 0
    # assist.step()

    for epoch in range(start_epoch, start_epoch+epochs):  # loop over the dataset multiple times

        if True:  # epoch > 0:
            net.eval()
            if devloader is None:
                dev_acc = dev_loss = 0.0
            else:
                with torch.no_grad():
                    running_dev_acc = 0.0
                    running_dev_loss = 0.0
                    running_batch_time = 0.0
                    n_words = 0
                    gold_categories = Counter()
                    generated_categories = Counter()
                    correct_categories = Counter()
                    for data in devloader:
                        x, ys = data['x'], data['y']
                        word_mask = (x['word'] != net.span_encoder.pad_token_id)

                        start_time = time.time()
                        result = net(x, word_mask=word_mask)
                        running_batch_time += time.time() - start_time

                        # y_hats, masks = net(x)
                        y_hat, address_mask = result['y_hat'], result['mask']

                        task, gen, y, criterion = net.tasks[0], \
                                                  net.generators[0], \
                                                  ys[0], \
                                                  criteria[0]
                        mapped_criterion = mapped_criteria[0] if hasattr(gen, 'address_map') else None

                        seq_len = y.size(1)

                        output_dim = gen.output_dim
                        address_dim = gen.address_dim
                        # y_hat = y_hat.view(batch_size, -1, address_dim, output_dim)

                        # argmaxes = torch.argmax(y_hat, dim=3)

                        acc, loss, (argmaxes, correct_bool, categories_hat, categories_gold) = \
                            compute_acc_and_loss(task, gen, y, y_hat,
                                                 address_mask, word_mask,
                                                 criterion, mapped_criterion,
                                                 batch_size, seq_len, address_dim,
                                                 output_dim,
                                                 loss_fxn=loss_fxn,
                                                 omega_native_atom=omega_native_atom,
                                                 omega_atom=omega_atom, omega_full=omega_full,
                                                 lambda_enc=lambda_enc, lambda_dec=lambda_dec,
                                                 output_correct_bool=True)

                        # print(argmaxes)

                        running_dev_acc += acc.item()
                        running_dev_loss += loss.item()

                        correct_indices = correct_bool.nonzero().tolist()
                        # print('gold', end=' ')
                        # print(y.size(), y[0, 0].tolist())
                        # categories_gold = gen.extract_outputs(y.view(batch_size, -1, address_dim))
                        # print(categories_gold)
                        # print('pred', end=' ')
                        # print(argmaxes.size(), argmaxes[0, 0].tolist())
                        # categories_hat = gen.extract_outputs(argmaxes)
                        # print(len(categories_gold), len(categories_hat))
                        for b, (sequence, sequence_gold) in enumerate(zip(categories_hat, categories_gold)):
                            # print('gold', sequence_gold, file=sys.stderr)
                            # print('pred', sequence, file=sys.stderr)
                            for s, cat_gold in enumerate(sequence_gold):
                                cat = sequence[s] if s < len(sequence) else None
                                n_words += 1
                                correct_index = [b, s] in correct_indices
                                # assert (cat == cat_gold) == (correct_index), (
                                #     b, s, cat, cat_gold, argmaxes[b, s], mask[b, s], y[b, s],
                                #     f'[{b}, {s}] {"not " if not correct_index else ""}in correct_indices')
                                if cat is None:
                                    cat = 'None'
                                else:
                                    msg = cat.validate()
                                    if msg != 0:
                                        if hasattr(gen, 'max_depth') and cat.depth() >= gen.max_depth:
                                            msg = 'Max depth reached'
                                            # print(b, s, msg, str(cat), cat.s_expr())
                                            cat = msg
                                        elif hasattr(gen, 'max_len') and argmaxes.size(1) >= gen.max_len:
                                            msg = 'Max length reached'
                                            # print(b, s, msg, str(cat), cat.s_expr())
                                            cat = msg
                                        else:
                                            # print(b, s, msg[0], str(cat), cat.s_expr(), file=sys.stderr)
                                            # print(argmaxes[b, s], file=sys.stderr)
                                            cat = msg[0]
                                gold_categories[str(cat_gold)] += 1
                                generated_categories[str(cat)] += 1
                                if correct_index:
                                    correct_categories[str(cat)] += 1

                        for k in x:
                            if isinstance(x[k], torch.Tensor):
                                x[k] = x[k].cpu()

                        for i in range(len(ys)):
                            _y = ys[i]
                            if isinstance(_y, torch.Tensor):
                                ys[i] = _y.cpu()

                        del x, word_mask, ys, y_hat, address_mask, argmaxes, correct_bool, acc, loss
                        torch.cuda.empty_cache()

                    dev_acc = running_dev_acc / dev_len
                    dev_loss = running_dev_loss / dev_len

            epoch_loss = epoch_running_loss / train_len
            epoch_acc = epoch_running_acc / train_len

            # assist.info['epoch'] = epoch
            # assist.info['train_loss'] = epoch_loss
            # assist.info['train_acc'] = epoch_acc
            # assist.info['dev_loss'] = dev_loss
            # assist.info['dev_acc'] = dev_acc
            # assist.step()

            print('[epoch %d summary] train loss: %.3f | train acc: %.3f | dev loss: %.3f | dev acc: %.3f | %.3f batches/s | %.3f expls/s' %
                  (epoch,
                   epoch_loss,
                   epoch_acc,
                   dev_loss,
                   dev_acc,
                   dev_len / running_batch_time,
                   (dev_len * batch_size) / running_batch_time
                  ),
                  file=sys.stderr)

            if devloader is not None:
                print(f'most common gold categories (out of {n_words} in dev): '
                      f'{" | ".join(str(item) for item in gold_categories.most_common(10))}',
                      file=sys.stderr)
                print(f'most common generated categories (out of {n_words} in dev): '
                      f'{" | ".join(str(item) for item in generated_categories.most_common(10))}',
                      file=sys.stderr)
                print(f'most common correct categories (out of {n_words} in dev): '
                      f'{" | ".join(str(item) for item in correct_categories.most_common(10))}',
                      file=sys.stderr)

            sys.stderr.flush()

            if devloader is None \
                    or dev_acc > best_dev_acc \
                    or dev_acc == best_dev_acc and (best_dev_loss is None or dev_loss < best_dev_loss):
                best_dev_acc = dev_acc
                best_dev_loss = dev_loss
                best_epoch = epoch
                torch.save({'model': net.to_dict(),
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': _optimizer.state_dict(),
                            'optimizer_name': optimizer_name,
                            'optim_kwargs': optimizer_kwargs,
                            'epoch': best_epoch,
                            'dev_acc': best_dev_acc,
                            'dev_loss': best_dev_loss}, model)

        net.train()

        epoch_running_loss = 0.0
        epoch_running_acc = 0.0

        # running_atom_loss = 0.0
        # running_cat_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        running_batch_time = 0.0
        # batch_indices = random.sample(range(train_len), train_len)
        random.shuffle(trainloader)
        for train_i, data in enumerate(trainloader):
            start_time = time.time()
            if max_batches is not None and train_i > max_batches:
                break
            # get the inputs; data is a list of [inputs, labels]
            # *x, y = data
            # x, y = data  # supertagger
            # print('x', x, file=sys.stderr)

            x, ys = data['x'], data['y']
            # seq_len = x['word'].size(1)
            word_mask = (x['word'] != net.span_encoder.pad_token_id)

            task, gen, y, criterion, dep_attn_criterion = net.tasks[0], \
                                                          net.generators[0], \
                                                          ys[0], \
                                                          criteria[0], \
                                                          dep_attn_criteria[0]
            mapped_criterion = mapped_criteria[0] if hasattr(gen, 'address_map') else None
            deps = data.get('dependencies', [None])[0]

            seq_len = y.size(1)

            result = net(x, ys=ys, word_mask=word_mask)
            y_hat, address_mask = result['y_hat'], result['mask']

            # y_hats, masks = y_hats.transpose(0, 1), masks.transpose(0, 1)

            # mtl_loss = 0.
            # zero the parameter gradients
            _optimizer.zero_grad()


            # print('y', y.size(), file=sys.stderr)
            # print('y_hat', y_hat.size(), file=sys.stderr)

            output_dim = gen.output_dim
            address_dim = gen.address_dim
            # y_hat = y_hat.view(batch_size, -1, address_dim, output_dim)
            # seq_len = y_hat.size(1)
            # with torch.no_grad():
            # argmaxes = torch.argmax(y_hat, dim=3)

            acc, loss = compute_acc_and_loss(task, gen, y, y_hat,
                                             address_mask, word_mask,
                                             criterion, mapped_criterion,
                                             batch_size, seq_len, address_dim,
                                             output_dim,
                                             loss_fxn=loss_fxn, oracle=result.get('y'),
                                             omega_native_atom=omega_native_atom,
                                             omega_atom=omega_atom, omega_full=omega_full,
                                             lambda_enc=lambda_enc, lambda_dec=lambda_dec,
                                             enc_attn=result.get('enc_attn'),
                                             dec_attn=result.get('dec_attn'),
                                             deps=deps,
                                             dep_attn_criterion=dep_attn_criterion)

            # with torch.no_grad():
            running_acc += acc.item()
            running_loss += loss.item()
            epoch_running_loss += loss.item()
            epoch_running_acc += acc.item()

            # categories_hat = gen.extract_outputs(argmaxes)
            # val = [(cat.validate(), cat) for s in categories_hat for cat in s]
            # assert all([not v[0] for v in val]), val

            # mtl_loss = loss
            #
            # if len(net.tasks) > 1:
            #     # y_hat, address_mask = net(x)
            #     for i, (task, gen, y, y_hat, address_mask, criterion) in enumerate(zip(net.tasks, net.generators,
            #                                                                            ys, y_hats, masks, criteria)
            #                                                                        )[1:]:
            #         output_dim = gen.output_dim
            #         address_dim = gen.address_dim
            #         # print('y_hat', y_hat.size(), file=sys.stderr)
            #         # y_hat = y_hat.view(batch_size, seq_len, address_dim, output_dim)
            #         # print('y_hat', y_hat.size(), file=sys.stderr)
            #         # y_hat = y_hat.view(-1, output_dim)
            #         # print('y_hat', y_hat.size(), file=sys.stderr)
            #         argmaxes = torch.argmax(y_hat, dim=3)
            #         acc, loss = compute_acc_and_loss(task, gen, y, y_hat, argmaxes,
            #                                          address_mask, word_mask,
            #                                          criterion,
            #                                          batch_size, seq_len, address_dim,
            #                                          output_dim)
            #
            #         mtl_loss += loss

            running_batch_time += time.time() - start_time

            # mtl_loss = torch.sum(torch.cat(losses, dim=0), dim=0, keepdim=True)
            # mtl_loss.backward()
            loss.backward()
            _optimizer.step()

            if train_i % n_print == n_print - 1:  # print every n mini-batches

                batch_time = running_batch_time / n_print
                print('[%d, %5d] loss: %.3f | acc: %.3f | %.1f %s | %.1f %s' % (epoch + 1, train_i + 1,
                                                                      running_loss / n_print,
                                                                      running_acc / n_print,
                                                                      batch_time if batch_time >= 1 else 1 / batch_time,
                                                                      's/batch' if batch_time >= 1 else 'batch(es)/s',
                                                                      batch_time/batch_size if batch_time/batch_size >= 1 else batch_size/batch_time,
                                                                      's/expl' if batch_time/batch_size >= 1 else 'expl(s)/s'),
                      file=sys.stderr)
                # if str(device).startswith('cuda'):
                #     print(torch.cuda.memory_summary(abbreviated=False), file=sys.stderr)

                # assist.info['batch'] = train_i + 1
                # assist.info['batch_loss'] = running_loss / n_print
                # assist.info['batch_acc'] = running_acc / n_print
                # assist.info['ex_per_s'] = batch_size / batch_time
                # assist.step()

                running_loss = 0.0
                running_acc = 0.0
                running_batch_time = 0.0

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

            del x, word_mask, ys, y_hat, address_mask, acc, loss, result
            torch.cuda.empty_cache()

            if optimizer_kwargs['use_schedule']:
                scheduler.step()

        # except:
        #     # raise
        #     exit(1)

    net.eval()
    if devloader is None:
        dev_acc = dev_loss = 0.0
    else:
        with torch.no_grad():
            running_dev_acc = 0.0
            running_dev_loss = 0.0
            n_words = 0
            gold_categories = Counter()
            generated_categories = Counter()
            correct_categories = Counter()
            for data in devloader:
                x, ys = data['x'], data['y']
                word_mask = (x['word'] != net.span_encoder.pad_token_id)

                result = net(x, word_mask=word_mask)
                # y_hats, masks = net(x)
                y_hat, address_mask = result['y_hat'], result['mask']

                # on dev, only compute acc and loss for primary task
                task, gen, y, criterion = net.tasks[0], \
                                          net.generators[0], \
                                          ys[0], \
                                          criteria[0]
                mapped_criterion = mapped_criteria[0] if hasattr(gen, 'address_map') else None

                seq_len = y.size(1)

                output_dim = gen.output_dim
                address_dim = gen.address_dim
                y_hat = y_hat.view(batch_size, -1, address_dim, output_dim)

                # argmaxes = torch.argmax(y_hat, dim=3)
                acc, loss, (argmaxes, correct_bool, categories_hat, categories_gold) = \
                    compute_acc_and_loss(task, gen, y, y_hat,
                                         address_mask, word_mask,
                                         criterion, mapped_criterion,
                                         batch_size, seq_len, address_dim,
                                         output_dim,
                                         loss_fxn=loss_fxn,
                                         omega_native_atom=omega_native_atom,
                                         omega_atom=omega_atom, omega_full=omega_full,
                                         lambda_enc=lambda_enc, lambda_dec=lambda_dec,
                                         output_correct_bool=True)

                running_dev_acc += acc.item()
                running_dev_loss += loss.item()

                correct_indices = correct_bool.nonzero().tolist()
                # categories_gold = gen.extract_outputs(y.view(batch_size, -1, address_dim))
                # categories_hat = gen.extract_outputs(argmaxes)
                for b, (sequence, sequence_gold) in enumerate(zip(categories_hat, categories_gold)):
                    # print(sequence, sequence_gold, file=sys.stderr)
                    # print('gold', sequence_gold, file=sys.stderr)
                    # print('pred', sequence, file=sys.stderr)
                    for s, (cat, cat_gold) in enumerate(zip(sequence, sequence_gold)):
                        n_words += 1
                        correct_index = [b, s] in correct_indices
                        # assert (cat == cat_gold) == (correct_index), (
                        #     b, s, cat, cat_gold, argmaxes[b, s], mask[b, s], y[b, s],
                        #     f'[{b}, {s}] {"not " if not correct_index else ""}in correct_indices')
                        if cat is None:
                            cat = 'None'
                        else:
                            msg = cat.validate()
                            if msg != 0:
                                if hasattr(gen, 'max_depth') and cat.depth() >= gen.max_depth:
                                    msg = 'Max depth reached'
                                    # print(b, s, msg, str(cat), cat.s_expr())
                                    cat = msg
                                elif hasattr(gen, 'max_len') and argmaxes.size(1) >= gen.max_len:
                                    msg = 'Max length reached'
                                    # print(b, s, msg, str(cat), cat.s_expr())
                                    cat = msg
                                else:
                                    print(b, s, msg[0], str(cat), cat.s_expr(), file=sys.stderr)
                                    print(argmaxes[b, s], file=sys.stderr)
                                    cat = msg[0]
                        gold_categories[str(cat_gold)] += 1
                        generated_categories[str(cat)] += 1
                        if correct_index:
                            correct_categories[str(cat)] += 1

                for k in x:
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].cpu()

                for i in range(len(ys)):
                    _y = ys[i]
                    if isinstance(_y, torch.Tensor):
                        ys[i] = _y.cpu()

                del x, word_mask, ys, y_hat, address_mask, argmaxes, correct_bool, acc, loss
                torch.cuda.empty_cache()

        dev_acc = running_dev_acc / dev_len
        dev_loss = running_dev_loss / dev_len

    if devloader is None \
            or dev_acc > best_dev_acc \
            or dev_acc == best_dev_acc and (best_dev_loss is None or dev_loss < best_dev_loss):
        best_dev_acc = dev_acc
        best_dev_loss = dev_loss
        best_epoch = epoch + 1
        torch.save({'model': net.to_dict(),
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': _optimizer.state_dict(),
                'optimizer_name': optimizer_name,
                'optim_kwargs': optimizer_kwargs,
                'epoch': best_epoch,
                'dev_acc': best_dev_acc,
                'dev_loss': best_dev_loss
                }, model)

    epoch_loss = epoch_running_loss / train_len
    epoch_acc = epoch_running_acc / train_len

    # assist.info['epoch'] = epoch + 1
    # assist.info['train_loss'] = epoch_loss
    # assist.info['train_acc'] = epoch_acc
    # assist.info['dev_loss'] = dev_loss
    # assist.info['dev_acc'] = dev_acc
    # assist.step()

    print('[epoch %d summary] train loss: %.3f | train acc: %.3f | dev loss: %.3f | dev acc: %.3f' %
          (epoch + 1,
           epoch_loss,
           epoch_acc,
           dev_loss,
           dev_acc),
          file=sys.stderr)

    if devloader is not None:
        print(f'most common gold categories (out of {n_words} in dev): '
              f'{" | ".join(str(item) for item in gold_categories.most_common(10))}',
              file=sys.stderr)
        print(f'most common generated categories (out of {n_words} in dev): '
              f'{" | ".join(str(item) for item in generated_categories.most_common(10))}',
              file=sys.stderr)
        print(f'most common correct categories (out of {n_words} in dev): '
              f'{" | ".join(str(item) for item in correct_categories.most_common(10))}',
              file=sys.stderr)

    print('Finished Training', file=sys.stderr)
    sys.stderr.flush()


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.activation = torch.nn.ModuleList()
        self.hidden = torch.nn.ModuleList()

    def prepare_input(self, *args, **kwargs):
        pass

    def prepare_output(self, *args, **kwargs):
        pass

    def forward(self, x):
        pass

    def to_dict(self):
        return {}


class Encoder(NN):
    def __init__(self, *args, hidden_dims=[0], **kwargs):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dims[-1]
        # self._cache = None

    # @property
    # def cache(self):
    #     return self._cache
    #
    # @cache.setter
    # def cache(self, new):
    #     del self._cache
    #     self._cache = new

    def prepare_input(self, *args, **kwargs):
        pass

    def prepare_inputs(self, seqs, padding=None, train=False, lower=True, device=torch.device('cpu')):
        _seqs = {k: [] for k in self.to_ix}
        _seqs['idx'] = []
        padding = (max(len(s) for s in seqs) if padding is None else padding) + 2
        for seq in seqs:
            idxs = {k: [] for k in _seqs}
            seq = [START] + seq + [END]
            for w in seq:
                if lower and w not in (START, END):
                    w = w.lower()
                if train and w not in self.to_ix['word'] and len(self.to_ix['word']) < self.vocab_sizes['word']:
                    self.to_ix['word'][w] = len(self.to_ix['word'])
                idxs['idx'].append(self.to_ix['word'].get(w, self.to_ix['word'][UNK]))

                for _n in range(1, self.feat_chars+1):
                    pre_key = f'pre_{_n}'
                    suf_key = f'suf_{_n}'
                    if w not in (START, END) and _n <= len(w):
                        pre = w[:_n]
                        if train and pre not in self.to_ix[pre_key] and len(self.to_ix[pre_key]) < self.vocab_sizes[pre_key]:
                            self.to_ix[pre_key][pre] = len(self.to_ix[pre_key])
                        idxs[pre_key].append(self.to_ix[pre_key].get(pre, self.to_ix[pre_key][UNK]))
                        suf = w[-_n:]
                        if train and suf not in self.to_ix[suf_key] and len(self.to_ix[suf_key]) < self.vocab_sizes[suf_key]:
                            self.to_ix[suf_key][suf] = len(self.to_ix[suf_key])
                        idxs[suf_key].append(self.to_ix[suf_key].get(suf, self.to_ix[suf_key][UNK]))
                    else:
                        idxs[pre_key].append(self.to_ix[pre_key][PAD])
                        idxs[suf_key].append(self.to_ix[suf_key][PAD])
            while len(idxs['idx']) < padding:
                for key in idxs:
                    idxs[key].append(self.to_ix.get(key, self.to_ix['word'])[PAD])
            for key in idxs:
                _seqs[key].append(idxs[key])
        _seqs['idx'] = torch.tensor(_seqs['idx'], dtype=torch.long, device=device)
        word_mask = ((_seqs['idx'] != self.bos_token_id) & (_seqs['idx'] != self.eos_token_id)).to(device)
        _seqs['word'] = _seqs['idx'][word_mask].view(-1, padding - 2)
        # print('idx', _seqs['idx'].size(), _seqs['idx'])
        # print('word', _seqs['word'].size(), _seqs['word'])
        for k in _seqs:
            if k not in ('idx', 'word'):
                _seqs[k] = torch.tensor(_seqs[k], dtype=torch.long, device=device)
        return _seqs

    def forward(self, x):
        pass

    def to_dict(self):
        result = super().to_dict()
        result.update({'hidden_dim': self.hidden_dim})
        return result


class Scorer(NN):
    def __init__(self, input_dim, output_dim, hidden_dims=[300, 300], dropout=[0.0, 0.0],
                 activation=F.gelu, norm=F.log_softmax):  # TODO: try cube activation
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.norm = norm

        # if isinstance(activation, list):
        #     assert len(activation) == len(hidden_dims) + 1
        #     self.activation = torch.nn.ModuleList(activation)
        # else:
        #     self.activation = torch.nn.ModuleList([activation()] * len(hidden_dims))
        self.activation = activation

        # assert len(dropout) == len(hidden_dims)

        self.dropout = torch.nn.ModuleList([torch.nn.Dropout(d) for d in dropout])

        if len(hidden_dims) > 0:
            self.hidden.append(torch.nn.Linear(input_dim, hidden_dims[0]))
            for i in range(len(hidden_dims) - 1):
                self.hidden.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))

            # The activation layer that maps from hidden state space to output space
            self.hidden2out = torch.nn.Linear(hidden_dims[-1], output_dim)

        else:
            # self.in2hidden = lambda x: x
            self.hidden2out = torch.nn.Linear(input_dim, output_dim)

    def prepare_input(self, x, *args, **kwargs):
        return torch.tensor(x, dtype=torch.float32)

    def prepare_output(self, y, *args, **kwargs):
        return torch.tensor(y, dtype=torch.long)

    def _forward(self, x):
        # print('Scorer._forward', file=sys.stderr)
        # print(x, file=sys.stderr)
        # print(self.in2hidden, file=sys.stderr)
        # print(self.activation[0], file=sys.stderr)
        # x = self.activation[0](self.in2hidden(x))
        # x = self.dropout[0](self.activation[0](self.in2hidden(x)))
        # print(x, file=sys.stderr)
        for i, h in enumerate(self.hidden):
            # print(h, file=sys.stderr)
            # print(self.activation[i+1], file=sys.stderr)
            x = self.dropout[i](self.activation(h(x)))
        # out = x if self.hidden_dims == 0 else self.hidden2out(x)
        # print(out.shape)
        # scores = self.norm(out, dim=1)
        # print(scores.shape)
        return x

    def forward(self, x):
        # print(out.shape)
        # scores = self.norm(out, dim=1)
        x = self._forward(x)
        out = self.hidden2out(x)
        scores = self.norm(out, dim=1)
        return scores

    def score(self, outcome, x) -> float:
        with torch.no_grad():
            scores = self.forward(x).view(-1)
            # scores = self.forward(*x).view(-1)
            return scores[outcome]

    def classify(self, x):
        with torch.no_grad():
            scores = self.forward(x).view(-1)
            return max(enumerate(scores), key=itemgetter(1)), scores

    def to_dict(self):
        result = super().to_dict()
        result.update({'input_dim': self.input_dim,
                       'output_dim': self.output_dim,
                       'hidden_dims': self.hidden_dims})  # ,
                       # 'hidden': self.hidden})
        return result


class SequenceTagger(Scorer):
    def __init__(self, span_encoder, labels, hidden_dims=[300, 300], dropout=[0.0, 0.0], **kwargs):
        super(SequenceTagger, self).__init__(input_dim=span_encoder.hidden_dim,
                                             output_dim=len(labels),
                                             hidden_dims=hidden_dims,
                                             dropout=dropout, **kwargs)

        self.span_encoder = span_encoder

        self.out_to_ix = labels
        self.ix_to_out = {v: k for k, v in self.out_to_ix.items()}

    def prepare_inputs(self, *args, **kwargs):
        return self.span_encoder.prepare_inputs(*args, **kwargs)

    def forward(self, x):
        x = self.span_encoder(x)                                       # (seq, batch, in)
        seq_len = x.size(0)
        x = x.view(-1, self.span_encoder.hidden_dim)                   # (seq*batch, in)
        h = super(SequenceTagger, self)._forward(x)                    # (seq*batch, out)
        y = self.hidden2out(h)
        return self.norm(y.view(-1, seq_len, self.output_dim), dim=2)  # (batch, seq, out)

    def to_dict(self):
        result = super().to_dict()
        result.update({'span_encoder': self.span_encoder.to_dict()})
                       # 'labels': self.out_to_ix})
        return result


if __name__ == '__main__':
    pass
