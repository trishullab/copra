#!/usr/bin/env python3

import os
import logging
import typing
import copy
from rank_bm25 import BM25Okapi
from itp_interface.tools.coq_executor import CoqExecutor
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.training_data_format import TrainingDataFormat
from copra.retrieval.abstraction import ReRanker

class CoqBm25ReRanker(ReRanker):
    def __init__(self, k1: float = 1.0, b: float = 0.75, epsilon: float = 0.25, language: str = '') -> None:
        super().__init__()
        self.bm25: BM25Okapi = None
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._responses = None
        self._language = language
    
    @property
    def language(self) -> str:
        return self._language
    
    @property
    def responses(self) -> typing.List[str]:
        return self._responses if self._responses is not None else []

    def rerank(self, query: str, responses: typing.List[str]) -> typing.List[float]:
        tokenized_index_data = [list(CoqExecutor.tokenize(response)) for response in responses]
        if len(tokenized_index_data) == 0:
            # print("WARNING: No tokenizable responses found. Returning all 0.0 scores.")
            # print("Query:", query)
            # print("Responses:", responses)
            return [0.0] * len(responses)
        bm25 = BM25Okapi(tokenized_index_data, k1=self.k1, b=self.b)
        query_tokens = list(CoqExecutor.tokenize(query))
        scores = bm25.get_scores(query_tokens)
        return [float(score) for score in scores]

    def get_scores(self, query: str) -> typing.List[float]:
        assert self._responses is not None, "Responses not set. Please call reindex(responses) first."
        assert self.bm25 is not None, "BM25 not initialized. Please call reindex(responses) first."
        query_tokens = list(CoqExecutor.tokenize(query))
        scores = self.bm25.get_scores(query_tokens)
        return [float(score) for score in scores]
    
    def reindex(self, responses: typing.List[str]) -> None:
        self._responses = copy.deepcopy(responses)
        tokenized_index_data = [list(CoqExecutor.tokenize(response)) for response in responses]
        self.bm25 = BM25Okapi(tokenized_index_data, k1=self.k1, b=self.b)

class CoqBM25TrainingDataRetriever(object):
    def __init__(self, 
        data_folder: str, 
        metadata_filename: str, 
        k1: float = 1.0, 
        b: float = 0.75, 
        epsilon: float = 0.25,
        logger: logging.Logger = None) -> None:
        assert data_folder is not None
        assert metadata_filename is not None
        assert os.path.exists(data_folder)
        assert os.path.exists(os.path.join(data_folder, metadata_filename))
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.data_folder = data_folder
        self.metadata_filename = metadata_filename
        self._loaded = False
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.training_data = TrainingData(self.data_folder, self.metadata_filename, logger=self.logger, max_parallelism=20)
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        self.training_data.load()
        # Go over all training data goals and tokenize them
        # If there are more than one goals then just pick the first one
        self.logger.info(f"Enumerating all training data...")
        self.all_training_data = [data for data in self.training_data if len(data.start_goals) > 0]
        self.logger.info(f"Found {len(self.all_training_data)} training data.")
        self.logger.info(f"Extracting all goals...")
        self.all_goals = [data.start_goals[0].goal for data in self.all_training_data]
        self.logger.info(f"Found {len(self.all_goals)} goals.")
        # Unload the training data
        self.logger.info(f"Unloading training data...")
        self.training_data.unload()
        self.logger.info(f"Training data unloaded.")
        # Tokenize all goals
        self.logger.info(f"Tokenizing {len(self.all_goals)} goals...")
        self._tokenized_goals = [list(CoqExecutor.tokenize(goal)) for goal in self.all_goals]
        self.logger.info(f"Tokenization complete.")
        self.logger.info(f"Initializing BM25 with k1={self.k1}, b={self.b}, epsilon={self.epsilon}")
        self.bm25 = BM25Okapi(self._tokenized_goals, k1=self.k1, b=self.b, epsilon=self.epsilon)
        self.logger.info(f"BM25 initialized.")
        self._loaded = True
    
    def find_relevant_training_data(self, query: str, num_results: int = 1) -> typing.List[typing.Tuple[float, TrainingDataFormat]]:
        assert self.is_loaded
        query_tokens = list(CoqExecutor.tokenize(query))
        scores = self.bm25.get_scores(query_tokens)
        # Sort the scores and return the top num_results
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:num_results]
        # Normalize the scores
        score_sum = sum([score for _, score in top_results]) + 1e-6 # Add a small epsilon to avoid division by zero
        top_results = [(index, score/score_sum) for index, score in top_results]
        return [(score, self.all_training_data[index]) for index, score in top_results]

if __name__ == "__main__":
    import time
    import argparse
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logging_dir = f".log/retriever/bm25/{current_time}"
    os.makedirs(logging_dir, exist_ok=True)
    log_file = f"{os.path.join(logging_dir, f'coq_raw_proofs.log')}"
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Bm25Retriever")
    logger.info(f"Process ID: {os.getpid()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default=".log/tools/coq_raw_proofs/data/test/coq/custom_group_theory/2023-09-06-03-32-37/theories/train", help="The directory where the raw proofs will be dumped")
    parser.add_argument("--metadata_filename", type=str, default="single.meta.json", help="The metadata filename")
    args = parser.parse_args()
    retriever = CoqBM25TrainingDataRetriever(args.data_folder, args.metadata_filename, logger=logger)
    retriever.load()
    while True:
        try:
            query = input("Query: ")
            if query == "exit":
                break
            results = retriever.find_relevant_training_data(query, num_results=5)
            print(f"Found top {len(results)} results:")
            for idx, (score, tdf) in enumerate(results):
                print(f"Result {idx+1}:")
                print("-"*50)
                print(f"Score: {score}")
                print(f"Goals: {tdf.start_goals[0].goal}")
                print(f"Proofs: {tdf.proof_steps[0][:100]}")
                print("-"*50)
        except KeyboardInterrupt:
            break