'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''

from os import PathLike
from typing import Dict, Optional, List, Any, Union, Iterable
from collections import OrderedDict, Counter

import re
import numpy

import torch
import torch.cuda as cuda

from allennlp.common.params import Params
from allennlp.data import Vocabulary, Instance, TextFieldTensors
from allennlp.data.batch import Batch
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn import util

from ..util.loader import batch_supertags
from ..tagger.supertagger import SuperTagger, build_supertagger
from ..tagger.nn import create_emb_layer, load_model_states, UNK, PAD, START, END, SEP
from ..tagger.encoders.span import BERTLikeSpanEncoder


class TextFieldEmbedderWrapper(TextFieldEmbedder):
    def __init__(self, span_encoder: BERTLikeSpanEncoder):
        super(TextFieldEmbedderWrapper, self).__init__()
        self.wrapped_encoder = span_encoder
        self.wrapped_encoder.register_forward_hook(self.add_token_offsets_and_passthrough)

    def forward(self, *args, **kwargs):
        return

    def add_token_offsets_and_passthrough(self, module, inp, outp):
        idxs = inp[0]['idxs']  # list (batch_size) of lists (num_spans) of idx lists (span_len)
        padding = inp[0]['padding']
        offsets = [[([seq[i][0], seq[i][-1]] if i < len(seq) else [0, 0]) for i in range(padding)] for seq in idxs]
        offsets = torch.tensor(offsets, dtype=torch.long)
        self({'': {'offsets': offsets}})  # dict that contains dict with key 'offsets' and value Tensor of shape (batch_size, num_spans, 2)

    def get_output_dim(self):
        return self.wrapped_encoder.hidden_dim


@Model.register("ccg_model")
class CCGModel(Model):

    default_predictor = "ccg_predictor"

    def __init__(self,
                 wrapped_model: SuperTagger,
                 vocab: Vocabulary,
                 label_namespace: str = "labels",
                 verbose_metrics: bool = False,
                 **kwargs) -> None:
        super(CCGModel, self).__init__(vocab)
        self.wrapped_model = wrapped_model
        self.text_field_embedder_wrapper = TextFieldEmbedderWrapper(self.wrapped_model.span_encoder)
        self.label_namespace = label_namespace
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.pad_token_index = self.wrapped_model.generators[0].out_to_ix[PAD]
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_index, reduction='sum')
        self._verbose_metrics = verbose_metrics
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

    def forward(self,
                instances: Iterable[Instance],
                metadata: List[Dict[str, Any]] = None,
                output_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        _tokens = []
        max_sentence_length = 0
        batch_size = 0
        for instance in instances:
            _tokens.append([t.text for t in instance['tokens']])
            max_sentence_length = max(max_sentence_length, instance['tokens'].sequence_length())
            batch_size += 1
        batched = batch_supertags(self.wrapped_model, _tokens, [], max_sentence_length, batch_size)
        tokens = batched['x']
        word_mask = (tokens['word'] != self.wrapped_model.span_encoder.pad_token_id)
        outputs = self.wrapped_model(tokens, ys=None, word_mask=word_mask,
                                     output_attention_weights=output_attention_weights)
        outputs['x'] = tokens
        outputs['word_mask'] = word_mask
        return outputs

    def forward_on_instance(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        return self.forward_on_instances([instance])

    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        # Parameters

        instances : `List[Instance]`, required
            The instances to run the model on.

        # Returns

        A list of the models output for each instance.
        """
        with torch.no_grad():
            return self.make_output_human_readable(self(instances))

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        all_predictions = output_dict['y_hat']
        address_mask = output_dict['mask']
        word_mask = output_dict['word_mask']
        seq_len = output_dict['x']['word'].size(1)

        gen = self.wrapped_model.generators[0]
        output_dim = gen.output_dim
        address_dim = gen.address_dim

        all_predictions = all_predictions.view(-1, seq_len, address_dim, output_dim)
        argmax_indices = torch.argmax(all_predictions, dim=-1)

        if hasattr(gen, 'address_map'):
            address_mask = address_mask.view(-1, seq_len, address_dim)
            mask = (~word_mask).unsqueeze(-1).expand(-1, -1, address_dim) | (~address_mask)
            argmax_indices[mask] = self.pad_token_index
        else:
            argmax_indices = argmax_indices.view(-1, seq_len)
            address_mask = address_mask.view(-1, seq_len)
            argmax_indices[~address_mask] = self.pad_token_index

        nice_output_dict = {}
        nice_output_dict['probs'] = torch.softmax(all_predictions, dim=-1).cpu().data.numpy()
        nice_output_dict['tags'] = [[str(cat) for cat in seq] for seq in gen.extract_outputs(argmax_indices.cpu())]
        nice_output_dict['loss'] = self.criterion(all_predictions.view(-1, output_dim), argmax_indices.view(-1)) / word_mask.float().sum()
        return nice_output_dict

    @classmethod
    def load(cls, args) -> "Model":
        checkpoint = torch.load(f'{args.model}.pt', map_location=args.device)
        checkpoint['model_state_dict'] = load_model_states(checkpoint['model_state_dict'])
        feat_emb_weights = {}
        if args.span_encoder not in ('bert', 'roberta'):
            for key in list(checkpoint['model_state_dict'].keys()):
                mo = re.match('span_encoder\.feat_embeddings\.([^.]+)\.weight', key)
                if mo:
                    feat_emb_weights[mo.group(1)] = checkpoint['model_state_dict'].pop(key)
        model = build_supertagger(args, additional_files=args.testing_files,
                        model_checkpoint=checkpoint['model'],
                        emb_weights=None if args.span_encoder in ('bert', 'roberta') else checkpoint[
                            'model_state_dict'].pop('span_encoder.embeddings.weight'),
                        feat_emb_weights=None if args.span_encoder in ('bert', 'roberta') else feat_emb_weights)

        epoch = checkpoint['epoch']
        dev_acc = checkpoint['dev_acc']
        dev_loss = checkpoint['dev_loss']

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
            model = torch.nn.DataParallel(model, device_ids=args.cuda_devices)

            model.tasks = tasks
            model.span_encoder = span_encoder
            model.prepare_inputs = prepare_inputs
            model.generators = torch.nn.ModuleList(out_fxns)
            model.to_dict = to_dict

        model = model.to(args.device)
        model.eval()

        vocab = Vocabulary(counter={"labels": model.generators[0].out_to_ix},
                           padding_token=PAD,
                           oov_token=UNK)

        return CCGModel(model, vocab)

