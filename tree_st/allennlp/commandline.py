'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from .reader import CCGReader
from .model import CCGModel
from .predictor import CCGSupertaggerPredictor

import tree_st.util.argparse as ap

args = ap.main()

model = CCGModel.load(args)
vocab = model.vocab
reader = CCGReader(word_tag_delimiter='|')
predictor = CCGSupertaggerPredictor(model, reader)


confirm_map = {
    '': True,
    'y': True,
    'yes': True,
    'n': False,
    'no': False
}

while True:
    text = ''
    new_input = input('\nType a sentence or multiple and hit [ENTER]!\n')
    while new_input.strip():
        text += '\n' + new_input
        new_input = input()
    if not text.strip():
        confirm = input('\nDid not receive input. Do you want to exit? [Y|n]').lower()
        if confirm in confirm_map:
            if confirm_map[confirm]:
                break
            continue
        print('\nDid not recognize input. To exit, hit [ENTER] without typing a sentence and then hit [ENTER] again.')
        continue

    print(f'Tagging input:\n{text}\n')
    print(*predictor.json_to_labeled_instances({"sentence": text}), sep='\n')
