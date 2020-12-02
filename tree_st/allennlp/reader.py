'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from typing import Dict, List, Iterable, Optional

from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary, Instance
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader

from .model import UNK

@DatasetReader.register("ccg_reader")
class CCGReader(SequenceTaggingDatasetReader):
    def __init__(self, *args, **kwargs):
        super(CCGReader, self).__init__(*args, **kwargs)
        self.tokenizer = WhitespaceTokenizer()

    def text_to_instances(self, text: str) -> Iterable[Instance]:
        instances = []
        for line in text.split('\n'):
            instance = self.text_to_instance(line)
            if instance:
                instances.append(instance)
        return instances

    def text_to_instance(self, line: str) -> Optional[Instance]:
        tokens = []
        tags = []
        toks_tags = self.tokenizer.tokenize(line)
        if not toks_tags:
            return None
        for tok_tag in toks_tags:
            tok, *tag = tok_tag.text.split(self._word_tag_delimiter)
            tokens.append(Token(tok))
            tags.append(tag or UNK)

        inst = Instance({'tokens': TextField(tokens, {})})
        inst.add_field('tags', SequenceLabelField(tags, inst['tokens']))
        return inst
