#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import torch
import typing
import enum
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class DecodingStrategy(enum.Enum):
    BEAM_SEARCH = 0
    NUCLEUS_SAMPLING = 1

class DecodingParameters:
    def __init__(self, decoding_strategy: DecodingStrategy, num_samples: int = 5, generation_max_length: int = 512, padding: bool = False):
        self.num_samples = num_samples
        self.decoding_strategy = decoding_strategy
        self.generation_max_length = generation_max_length
        self.padding = padding

class BeamSearchParameters(DecodingParameters):
    def __init__(self, num_samples: int = 5, generation_max_length: int = 512, padding: bool = False, beam_width: int = 10):
        super().__init__(DecodingStrategy.BEAM_SEARCH, num_samples, generation_max_length, padding)
        self.beam_width = beam_width

class NucleusSamplingParameters(DecodingParameters):
    def __init__(self, num_samples: int = 5, generation_max_length: int = 512, padding: bool = False, top_k: int = 20, top_p: float = 0.75, temperature: float = 1.0):
        super().__init__(DecodingStrategy.NUCLEUS_SAMPLING, num_samples, generation_max_length, padding)
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

@dataclass_json
@dataclass
class GenerationResult:
    input: str
    output: typing.List[str] = field(default_factory=list)
    probabilities: typing.List[float] = field(default_factory=list)

class BaseLM:
    def load_model(self, model_name : str, model_max_length: int = 512):
        raise NotImplementedError("load_model() not implemented")
    
    def get_embedding(self, snippet: str) -> torch.Tensor:
        raise NotImplementedError("get_embedding() not implemented")
    
    def get_embedding_batch(self, snippets: typing.List[str]) -> torch.Tensor:
        raise NotImplementedError("get_embedding_batch() not implemented")
    
    def generate(self, inputs: typing.Union[typing.List[str], str], decoding_params: DecodingParameters) -> typing.Union[GenerationResult, typing.List[GenerationResult]]:
        raise NotImplementedError("generate() not implemented")

class BaseLMBasedClassifier:
    def load_model(self, model_name : str, model_max_length: int = 512):
        raise NotImplementedError("load_model() not implemented")
    
    def get_prediction(self, inp: typing.Any) -> torch.Tensor:
        raise NotImplementedError("get_embedding() not implemented")
    
    def get_prediction_batch(self, inps: typing.List[typing.Any]) -> torch.Tensor:
        raise NotImplementedError("get_embedding_batch() not implemented")