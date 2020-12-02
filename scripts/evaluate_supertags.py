"""
Evaluate ...

on unseen words, tags, usages

on atomic and complex sub-categories

w.r.t. functionality (depth >= 1, slash direction, functionality of argument and result)

w.r.t. complexity of sub-categories for training and prediction

@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
"""
import sys

import json

from tree_st.util.argparse import get_filepaths_from_glob
from tree_st.util.reader import DerivationsReader, AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader, DependenciesReader, OutputDependenciesReader
from tree_st.tagger.eval.evaluation import Evaluator

if __name__ == '__main__':

    print(sys.argv, file=sys.stderr)
    preds = get_filepaths_from_glob([sys.argv[1]])
    golds = get_filepaths_from_glob([sys.argv[2]])
    if sys.argv[3] != '-':
        pred_deps = get_filepaths_from_glob([sys.argv[3]])
    else:
        pred_deps = None
    if sys.argv[4] != '-':
        gold_deps = get_filepaths_from_glob([sys.argv[4]])
    else:
        gold_deps = None
    testing_format = sys.argv[5]
    train = get_filepaths_from_glob(sys.argv[6:])
    max_depth = 6  # int(sys.argv[4])

    evl = Evaluator(train, max_depth=max_depth)

    print(preds, golds, file=sys.stderr)

    preds, golds = iter(preds), iter(golds)

    if testing_format == 'ast':
        dr = ASTDerivationsReader
    elif testing_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    pdr, gdr = dr(next(preds), validate=True, print_err_msgs=True), dr(next(golds), validate=True, print_err_msgs=True)

    if pred_deps:
        pred_deps = iter(pred_deps)
        odepr = OutputDependenciesReader
        pdepr = odepr(next(pred_deps))

    if gold_deps:
        gold_deps = iter(gold_deps)
        depr = DependenciesReader
        gdepr = depr(next(gold_deps))

    while True:
        try:
            pred = next(pdr)
        except StopIteration:
            try:
                pf = next(preds)
            except StopIteration:
                break
            else:
                pdr = dr(pf)
                pred = next(pdr)

        try:
            gold = next(gdr)
        except StopIteration:
            try:
                gf = next(golds)
            except StopIteration:
                break
            else:
                gdr = dr(gf)
                gold = next(gdr)

        if pred_deps:
            try:
                pred_dep = next(pdepr)
            except StopIteration:
                try:
                    pdep = next(pred_deps)
                except StopIteration:
                    break
                else:
                    pdepr = odepr(pdep)
                    pred_dep = next(pdepr)

        if gold_deps:
            try:
                gold_dep = next(gdepr)
            except StopIteration:
                try:
                    gdep = next(gold_deps)
                except StopIteration:
                    break
                else:
                    gdepr = depr(gdep)
                    gold_dep = next(gdepr)

        pred, pID = pred['DERIVATION'], pred['ID']
        gold, gID = gold['DERIVATION'], gold['ID']
        pred = pred.lexical_deriv()
        gold = gold.lexical_deriv()

        if pred_deps:
            pred.set_dependencies(pred_dep['DEPENDENCIES'])
        # print('\n'.join(map(str, pred_dep['DEPENDENCIES'])))
        # print(pred.pretty_print())

        evl.add(pred, gold,
                pred_dep=pred_dep['DEPENDENCIES'] if pred_deps else None,
                gold_dep=gold_dep['DEPENDENCIES'] if gold_deps else None)

    # print(len(evl.pred_deps), '==', len(evl.pred_derivs))

    errors = evl.eval_supertags()

    with open(f'{sys.argv[1]}.err.log', 'w') as f:
        json.dump(errors, f, indent=2)
