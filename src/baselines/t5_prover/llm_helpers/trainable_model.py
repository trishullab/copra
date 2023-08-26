#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import enum
import typing
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import (PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, PretrainedConfig, TrainingArguments)

class ObjectiveTypes(enum.Enum):
    AutoRegressive = 1
    MaskedLM = 2
    BothAutoRegressiveAndMaskedLM = 3

class PaddingPolicy(enum.Enum):
    MAX_LENGTH = 0
    MAX_BATCH_LENGTH = 1

@dataclass
class TrainingDataArguments:
    padding: bool = False
    truncation: bool = True
    max_length: int = 512
    max_len_x: int = 256
    max_len_y: int = 256
    padding_policy: PaddingPolicy = PaddingPolicy.MAX_BATCH_LENGTH
    ignore_input_longer_than_block_size: bool = True
    shuffle: bool = True

class TheoremProvingTrainableModelWrapper(object):
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer, 
                 config : PretrainedConfig,
                 block_size: int = 512):
        assert model is not None, "Model cannot be None"
        assert isinstance(model, PreTrainedModel), "Model must be a PreTrainedModel"
        assert tokenizer is not None, "Tokenizer cannot be None"
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), "Tokenizer must be a PreTrainedTokenizer and PreTrainedTokenizerFast"
        assert config is not None, "Config cannot be None"
        assert isinstance(config, PretrainedConfig), "Config must be a PretrainedConfig"
        assert block_size is not None, "Block size cannot be None"
        assert isinstance(block_size, int), "Block size must be an integer"
        assert block_size > 0, "Block size must be greater than 0"
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.block_size = block_size

    def get_pretrained_model(self) -> PreTrainedModel:
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_config(self):
        return self.config
    
    def get_max_length(self):
        return self.tokenizer.model_max_length

    def get_block_size(self) -> int:
        return self.block_size
    
    def is_generative_model(self) -> bool:
        return False
    
    def tokenize(self, texts: typing.Union[str, typing.List[str]], truncation: bool = True, padding: bool = False, max_length: int = None, model: torch.nn.Module = None) -> typing.Dict[str, typing.Any]:
        """
        model: If None, use self.model (This is  useful for distributed training while using DataParallel)
        """
        raise Exception("Implement the trainable model")
    
    def get_metrics(self, dataloader: DataLoader, metric_key_prefix: str = "eval", objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive) -> typing.Dict[str, float]:
        raise Exception("Implement the trainable model")

    def get_predictions(self, dataloader: DataLoader, metric_key_prefix: str = "test", objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive) -> typing.Any:
        """
        This should return namedtuple with the following fields:
        - predictions: List of predictions
        - label_ids: List of labels if available
        - metrics: Dict of metrics
        """
        raise Exception("Implement the trainable model")

    def get_number_of_tokens(self, inputs: typing.Dict[str, typing.Any], labels: typing.Any) -> int:
        raise Exception("Implement the trainable model")

    def get_output_with_loss(self, inputs: typing.Dict[str, typing.Any], labels: typing.Any, objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive, model: torch.nn.Module = None):
        """
        model: If None, use self.model (This is  useful for distributed training while using DataParallel)
        """
        raise Exception("Implement the trainable model")
    
    def generate_and_compare(self, inputs: typing.Dict[str, typing.Any], labels: typing.Any, top_k: int = 5, objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive, metrics : typing.Dict[str, float] = {}, good_examples: list = [], bad_examples: list = []) -> list:
        """
        This should return a list of namedtuple with the following fields:
        - input: Input text
        - label: Label text
        - generated: List of Generated texts
        - metrics: Dict of metrics"""
        raise Exception("Implement the trainable model")

    def get_training_data_formatter(self) -> typing.Callable[[typing.Any], typing.Tuple[str, typing.Any]]:
        raise Exception("Implement the trainable model")
    
    def get_training_data_parser(self) -> typing.Callable[[str, str], typing.Any]:
        raise Exception("Implement the trainable model")
    
    def training_args(self) -> TrainingArguments:
        raise Exception("Implement the trainable model")
    
    def get_training_data_arguments(self) -> TrainingDataArguments:
        raise Exception("Implement the trainable model")