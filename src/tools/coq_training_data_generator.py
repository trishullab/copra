#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
import logging
import enum
from src.tools.training_data_format import MergableCollection, TrainingDataFormat
from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
logger = logging.getLogger("CoqTrainingGenerator")

class TrainingDataGenerationType(enum.Enum):
    FULL = 0
    RETRIEVER = 1
    GENERATION_BM25 = 2
    GENERATION_DPR = 3
    TEXT = 4
    LOCAL = 5

class GenericTheoremRetriever:
    def __init__(self):
        self.logger = None
        pass

    def filter_best_context(self, partial_data: TrainingDataFormat) -> TrainingDataFormat:
        raise NotImplementedError("retrieve_best_context must be implemented")

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

class GenericTrainingDataGenerationTransform(object):

    """Class to generate training data for for coq based automatic theorem provers.

    This class is responsible for generating training data for coq based automatic theorem provers.
    NOTE: This class is not thread safe!!
    """
    def __init__(self,
                 training_data_type: TrainingDataGenerationType,
                 buffer_size : int = 10000,
                 logger = None):
        """Initialize the training data generator.
        """
        assert buffer_size > 0, "Buffer size must be greater than 0"
        self.buffer_size = buffer_size
        self.logger : logging.Logger = logger if logger is not None else logging.getLogger()
        self.training_data_generation_type : TrainingDataGenerationType = training_data_type

    def get_meta_object(self) -> MergableCollection:
        raise NotImplementedError("get_meta_object must be implemented")

    def get_data_collection_object(self) -> MergableCollection:
        raise NotImplementedError("get_data_collection_object must be implemented")
    
    def load_meta_from_file(self, file_path) -> MergableCollection:
        raise NotImplementedError("load_meta_from_file must be implemented")
    
    def load_data_from_file(self, file_path) -> MergableCollection:
        raise NotImplementedError("load_data_from_file must be implemented")
    
    def get_proof_id(self, project_id: str, file_namespace: str, line_number: int, lemma_name: str) -> str:
        return f"<{project_id}>.{file_namespace}.{line_number:06d}.{lemma_name}"
    
    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any:
        raise NotImplementedError("__call__ must be implemented")
    
    @property
    def name(self):
        return self.training_data_generation_type.name.lower()