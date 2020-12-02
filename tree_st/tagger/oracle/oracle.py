'''
Need oracle for two things:

Loss (train model to make good current prediction)
- static global: penalize if doesn't match "globally" optimal chunking (e.g., least & largest chunks for each supertag)
- dynamic local: penalize if doesn't match "locally" optimal chunking (e.g., largest possible chunk with current position as root (purely local), or least chunks still achievable (local-global))
- dynamic goal-oriented: penalize (only) if supertag cannot be completed correctly (but ignore optimality of chunking) (can easily be translated into beam-search optimization a la Wiseman & Rush, 2016)

Teacher forcing (don't negatively affect subsequent predictions during training)
- always static
- predicted chunk if supertag can be completed correctly, otherwise local
- predicted chunk if supertag can be completed correctly, otherwise highest ranked chunk which supertag can still be completed correctly
- predicted chunk if supertag can be completed correctly, otherwise random chunk with which supertag can still be completed correctly
- make all valid options available

@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''


from collections import defaultdict

from tree_st.ccg.category import Category
from tree_st.ccg.util import sexpr_nodeblock_to_cat



def find_all_decompositions(cat: Category, tagset, start=set(), k=None):
    result = []

    agenda = [start]
    while agenda:
        nb = agenda.pop(0)
        current = sexpr_nodeblock_to_cat(nb, validate=False)
        if current is None:
            current = Category(None, validate=False)
        elif current.equals(cat):
            result.append(nb)
            continue
        addr = current.next_open_address()
        fillers = find_valid_fillers(current, addr, cat, tagset, k=k)
        for filler in fillers:
            _nb = nb.copy()
            _nb.add((int(addr), filler))
            agenda.append(_nb)

    return result


def find_valid_fillers(current: Category, address, gold: Category, tagset, k=8, binary=True):
    if current is None:
        current = Category(None, validate=False)

    nb = dict(current.to_dense_nodeblock(concat_attr=True, binary=False))
    if not nb.keys().isdisjoint(gold - current):
        # current category not compatible with gold
        return None

    if current.get_node(address, binary=binary) is not None:
        return None

    gold_partial = gold.get_category(address, binary=binary)
    if gold_partial is None:
        return None

    top = Category(None, validate=False)
    open = defaultdict(set)
    open['1'].add(top)
    result = set()

    for i, (addr, node) in enumerate(gold_partial.traverse(), start=1):

        to_remove = defaultdict(set)
        to_add = defaultdict(set)
        for partial in open[addr]:
            if partial.root is None:
                new_partial = Category(node, validate=False)
            else:
                new_partial = partial.copy()
                try:
                    new_partial.set_node(addr, node, validate=False)
                except ValueError:
                    pass

            new_partial_s = new_partial.s_expr()
            if tagset is None or new_partial_s in tagset:
                result.add(new_partial_s)
            for a in open:
                if a != addr and partial in open[a]:
                    to_add[a].add(new_partial)
            to_add[f'{addr}0'].add(new_partial)
            to_add[f'{addr}1'].add(new_partial)

        changed = False

        for a in to_add:
            for p in to_add[a]:
                open[a].add(p)
                changed = True

        for a in to_remove:
            for p in to_remove[a]:
                open[a].remove(p)
                changed = True

        if not changed:
            pass

        if k is not None and i >= k:
            break

    return result


def compatible(partial_decomp1, partial_decomp2):
    pass


def make_unordered_valid_loss(out_to_ix, fxn='avg_good_above_best_bad', eps1=1.0, eps2=1e-7):
    import torch
    import math

    _eps1_exp = math.exp(eps1)
    _eps1_log = math.log(eps1)
    _eps2_log = math.log(eps2)

    def _loss_all_good_above_all_bad(good_ix, scores):
        good = scores[sorted(good_ix)]
        not_good = scores[sorted(set(range(len(scores))) - good_ix)]
        max_ix = torch.argmax(scores, 0).item()
        if max_ix not in good_ix:
            return torch.sum(
                torch.clamp_min(torch.log(torch.ger(1 / torch.exp(good), torch.exp(not_good) * _eps1_exp)), 0.0))
        else:
            return torch.zeros(1, requires_grad=True).to(scores)

    def _loss_avg_good_above_best_bad(good_ix, scores):
        good = scores[sorted(good_ix)]
        max_ix = torch.argmax(scores, 0).item()
        if max_ix not in good_ix:
            return len(scores) * torch.clamp_min(scores[max_ix] + eps1 - torch.mean(good), 0.0)
        else:
            return torch.zeros(1, requires_grad=True).to(scores)

    def unordered_valid_loss(y_hat, categories_hat, categories):
        loss = torch.zeros(1, requires_grad=True).to(y_hat)
        for sent_scores_hat, sent_tags_hat, sent_tags in zip(y_hat, categories_hat, categories):
            for tag_scores_hat, tag_hat, tag in zip(sent_scores_hat, sent_tags_hat, sent_tags):
                if tag is not None and tag.root is not None:
                    for a in range(tag.size()):
                        current = tag_hat.copy()
                        try:
                            current.set_category(a, None, binary=False, validate=False)
                        except ValueError:
                            continue
                        # TODO: after a mistake has been made in prediction, the teacher-forced history that the next
                        #  predictions are conditioned on should also be reflected in this loss
                        # print('[LOSS] looking for fillers', current, '(', a, ')', '->', tag)
                        fillers = find_valid_fillers(current, a+1, tag, out_to_ix, k=8, binary=False)
                        if fillers is None:
                            # TODO: have to handle this?
                            continue
                        # print('found:', fillers)
                        filler_ix = {out_to_ix[l] for l in fillers}
                        address_scores_hat = torch.log_softmax(tag_scores_hat[a], 0)
                        if fxn == 'avg':
                            l = _loss_avg_good_above_best_bad(filler_ix, address_scores_hat)
                        elif fxn == 'all':
                            l = _loss_all_good_above_all_bad(filler_ix, address_scores_hat)
                        else:
                            raise NotImplementedError
                        loss = loss.clone() + l

        return loss
    return unordered_valid_loss


if __name__ == '__main__':

    import json
    from ccg.util.reader import CategoryReader, SCategoryReader

    scr = SCategoryReader()

    cr = CategoryReader()


    ts = '''(/)
         (\\)
         (C)
         (D)
         (E)
         (\\(~R(D))(~A(E)))
         (/(~R(B))(~A(C)))
         (/(~R(B)))
         (/(~A(C)))
         (/(~A(/(~R(B))(~A(C)))))
         (/(~A(/(~R(B)))))
         (/(~A(/(~A(C)))))'''.split()
    # ts = ts[:5] + ts[6:]

    with open('atomic_100.json') as f:
        ts = json.load(f)

    print('tagset', ts)
    print()

    # gold = Category('/', Category('\\', Category('D'), Category('E')), Category('/', Category('B'), Category('C')))
    gold = cr.read('(((S\\NP)\\(S\\NP))/((S\\NP)\\(S\\NP)))/(((S\\NP)\\(S\\NP))/((S\\NP)\\(S\\NP)))')
    # gold = Category('/', Category('/', Category('/', Category('/', Category('/', Category('/', Category('/', Category('A'), Category('B')), Category('C')), Category('D')), Category('E')), Category('F')), Category('G')), Category('H'))
    # partial = Category('/', arg=Category('C'), validate=False)  # Category('\\', Category('D'), Category('E'))
    partial = Category(None, validate=False)

    print(gold)
    print(list(gold.traverse()))
    fill = find_valid_fillers(partial, 1, gold, ts, k=8)
    print('fill', fill)

    # decomps = find_all_decompositions(gold, ts, start=set([(1, '(/)')]))
    decomps = find_all_decompositions(gold, ts, k=8)
    print('decomps', decomps)



