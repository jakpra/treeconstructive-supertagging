'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

import sys
import stanza

from tree_st.util import argparse
from tree_st.util.reader import ASTDerivationsReader, AUTODerivationsReader, StaggedDerivationsReader


def main(args):
    out = open(args.out, 'w', newline='\n', encoding='utf-8', errors='ignore') if args.out else sys.stdout

    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', tokenize_pretokenized=True)

    if args.testing_format == 'ast':
        dr = ASTDerivationsReader
    elif args.testing_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    # sents = []
    # for filepath in args.testing_files:
    #     ds = dr(filepath, print_err_msgs=True)
    #     for d in ds:
    #         deriv = d['DERIVATION']
    #         print(d['ID'], file=sys.stderr)
    #         lex = deriv.get_lexical(ignore_attr=False)
    #         sents.append(' '.join([dln.word for dln in lex]))
    #
    # doc = nlp('\n'.join(sents))

    # if not for_parser:
    #     print(
    #       # 'index', 'word', 'POS1', 'POS2',
    #       'gold category', 'depth', 'arguments', 'size',
    #       # 'word frequency', 'category frequency', 'usage frequency',
    #       sep='\t', file=out)
    i = 0
    for filepath in args.testing_files:
        ds = dr(filepath, print_err_msgs=True)
        for d in ds:
            deriv = d['DERIVATION']
            print(d['ID'], file=sys.stderr)
            lex = deriv.get_lexical(ignore_attr=False)

            doc = nlp(' '.join([dln.word for dln in lex]))
            # print(
            #     *[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}'
            #       for sent in doc.sentences for word in sent.words], sep='\n')
            words = doc.sentences[0].words
            assert len(words) == len(lex)

            for dln, word in zip(lex, words):
                cat = dln.category1
                if dln.word.lower() in ('so', 'too') and '[adj]' in str(dln.category1.arg):
                    pos1 = 'SO'
                elif dln.word.lower() == 'as' and '[adj]' in str(dln.category1.arg):
                    pos1 = 'AS'
                else:
                    pos1 = word.xpos

                cat = str(cat)
                # if cat == '-UNKNOWN-':
                #     if (dln.word.lower(), dln.pos1) in cats_by_wordpos:
                #         cat = cats_by_wordpos[dln.word.lower(), dln.pos1].most_common(1)[0][0]
                #     elif dln.word.lower() in cats_by_word:
                #         cat = cats_by_word[dln.word.lower()].most_common(1)[0][0]
                #     else:
                #         cat = cats_by_pos[dln.pos1].most_common(1)[0][0]
                print(dln.word, pos1, 1, cat, 1, sep='\t', file=out)
                i += 1

            print(file=out)

        ds.close()

    if args.out:
        out.close()


if __name__ == '__main__':
    args = argparse.main()
    main(args)
