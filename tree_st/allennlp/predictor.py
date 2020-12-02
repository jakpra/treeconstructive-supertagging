'''
@author: Jakob Prange (jakpra)
@copyright: Copyright 2020, Jakob Prange
@license: Apache 2.0
'''


from typing import Dict, List, Iterable, Tuple, Any, Union

import numpy

import torch

import allennlp
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import FlagField, TextField, SequenceLabelField, ArrayField
from allennlp.data.tokenizers import Token
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor

from ..ccg.category import Category
from ..util.reader import CategoryReader

from .field import ConstructiveSupertagField, json_to_cat


@Predictor.register("ccg_supertagger")
class CCGSupertaggerPredictor(SentenceTaggerPredictor):
    def __init__(self, *args, **kwargs):
        super(CCGSupertaggerPredictor, self).__init__(*args, **kwargs)
        self.ix_to_out = self._model.wrapped_model.generators[0].ix_to_out

    def predict(self, text: str) -> JsonDict:
        return self.predict_json({"sentence": text})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["sentence"])

    def _json_to_instances(self, json_dict: JsonDict) -> Iterable[Instance]:
        return self._dataset_reader.text_to_instances(json_dict["sentence"])

    def _instance_to_json(self, instance: Instance) -> JsonDict:
        return {
                'words': list(instance['tokens']),
                'tags': instance['tags'].as_json(),
                'probs': [
                    [
                        sorted(
                            [(p, self.ix_to_out[j]) for j, p in enumerate(atom.tolist())],
                            reverse=True
                        ) for atom in tag
                    ] for tag in instance['probs'].array
                ],
            }

    def _instances_to_json(self, instances: Iterable[Instance]) -> JsonDict:
        return {str(i): self._instance_to_json(instance) for i, instance in enumerate(instances)}

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return self._instances_to_json(self.json_to_labeled_instances(inputs))

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        instances = self._json_to_instances(inputs)
        return self.predictions_to_labeled_instances(
                   instances,
                   self.predict_instances(instances, raw=True)
               )

    def labeled_json_to_labeled_instances(self, json_dict: JsonDict) -> Dict[int, Instance]:
        seq_offset = 0
        seq_len = -1
        adhoc_vocab = Vocabulary()
        instances = {}
        for i, str_i in sorted(map((lambda x: (int(x), x)), json_dict.keys())):
            inst_obj = json_dict[str_i]
            if seq_len == -1:
                seq_len = len(inst_obj['words'])
                text_field = TextField([Token(tok['text']) for tok in inst_obj['words']], {})
                instance = Instance({'tokens': text_field})

            new_instance = instance.duplicate()

            tags_field = ConstructiveSupertagField(
                             [json_to_cat(tag) for tag in inst_obj['tags']],
                             text_field,
                             [i - seq_offset]
                         )
            adhoc_vocab.add_tokens_to_namespace(tags_field.labels, 'labels')
            new_instance.add_field(
                'tags', tags_field
            )
            new_instance.index_fields(adhoc_vocab)

            instances[i] = new_instance

            if i + 1 - seq_offset == seq_len:
                seq_offset += seq_len
                seq_len = -1

        return instances

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.
        # Parameters
        instances : `List[Instance]`
        # Returns
        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.
        # Notes
        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_layer = allennlp.nn.util.find_embedding_layer(self._model)

        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with torch.backends.cudnn.flags(enabled=False):
            outputs = self._model.make_output_human_readable(
                self._model.forward(instances)  # type: ignore
            )

            loss = outputs["loss"]

            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self._model.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in self._model.parameters():
                p.grad = None
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return grad_dict, outputs

    def predict_instance(self, instance: Instance) -> Dict[str, Union[Iterable, numpy.ndarray, torch.Tensor]]:
        return self.predict_instances([instance])

    def predict_instances(self, instances: Iterable[Instance], raw=False) -> Dict[str, Union[Iterable, numpy.ndarray, torch.Tensor]]:
        outputs = self._model.forward_on_instances(instances)
        return outputs if raw else sanitize(outputs)

    def predictions_to_labeled_instances(self, instances: Iterable[Instance],
                                         outputs: Dict[str, Union[numpy.ndarray, torch.Tensor, Iterable[Union[str, Category]]]]) -> List[Instance]:
        predicted_tags = outputs['tags']
        predicted_probs = outputs['probs']

        adhoc_vocab = Vocabulary()
        new_instances = []

        cr = CategoryReader()
        gen = self._model.wrapped_model.generators[0]

        for instance, tags, probs in zip(instances, predicted_tags, predicted_probs):
            text_field: TextField = instance['tokens']
            length = text_field.sequence_length()
            for i in range(length):
                new_instance = instance.duplicate()

                if all(map((lambda x: isinstance(x, Category)), tags)):
                    cat = tags[i:i+1]
                elif all(map((lambda x: isinstance(x, str)), tags)):
                    cat = [cr.read(tag) for tag in tags[i:i+1]]
                else:
                    cat = gen.extract_outputs(numpy.expand_dims(tags[i:i+1], 0))[0]

                tags_field = ConstructiveSupertagField(
                                 cat,
                                 text_field,
                                 [i]
                             )
                adhoc_vocab.add_tokens_to_namespace(tags_field.labels, 'labels')
                new_instance.add_field(
                    'tags', tags_field
                )
                new_instance.add_field(
                    'probs', ArrayField(probs[i:i+1])
                )
                new_instance.index_fields(adhoc_vocab)
                new_instances.append(new_instance)

        return new_instances

