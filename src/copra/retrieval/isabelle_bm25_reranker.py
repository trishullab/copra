#!/usr/bin/env python3

import os
import logging
import typing
import copy
from rank_bm25 import BM25Okapi
from itp_interface.tools.isabelle_executor import IsabelleExecutor
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.training_data_format import TrainingDataFormat
from copra.retrieval.abstraction import ReRanker

class IsabelleBm25ReRanker(ReRanker):
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
        tokenized_index_data = [list(IsabelleExecutor.tokenize(response)) for response in responses]
        if len(tokenized_index_data) == 0:
            # print("WARNING: No tokenizable responses found. Returning all 0.0 scores.")
            # print("Query:", query)
            # print("Responses:", responses)
            return [0.0] * len(responses)
        bm25 = BM25Okapi(tokenized_index_data, k1=self.k1, b=self.b)
        query_tokens = list(IsabelleExecutor.tokenize(query))
        scores = bm25.get_scores(query_tokens)
        return [float(score) for score in scores]

    def get_scores(self, query: str) -> typing.List[float]:
        assert self._responses is not None, "Responses not set. Please call reindex(responses) first."
        assert self.bm25 is not None, "BM25 not initialized. Please call reindex(responses) first."
        query_tokens = list(IsabelleExecutor.tokenize(query))
        scores = self.bm25.get_scores(query_tokens)
        return [float(score) for score in scores]
    
    def reindex(self, responses: typing.List[str]) -> None:
        self._responses = copy.deepcopy(responses)
        tokenized_index_data = [list(IsabelleExecutor.tokenize(response)) for response in responses]
        self.bm25 = BM25Okapi(tokenized_index_data, k1=self.k1, b=self.b)

if __name__ == "__main__":
    logging.basicConfig(filename='isabelle_executor.log', filemode='w', level=logging.INFO)
    with IsabelleExecutor(use_human_readable_proof_context=True, main_file="data/benchmarks/miniF2F/isabelle/test/aime_1983_p1.thy", project_root="data/benchmarks/miniF2F") as isabelle_exec:
        isabelle_exec.run_to_finish()
        all_lemmas = [str(lemma) for lemma in isabelle_exec.search_type_matching_defns("")] # Get all lemmas
    
    isabelle_bm25_reranker = IsabelleBm25ReRanker()
    isabelle_bm25_reranker.reindex(all_lemmas)
    inp = ""
    print("Tokenized lemmas:")
    print('-' * 80)
    for lemma in all_lemmas[:20]:
        print(f"{lemma}: \n{list(IsabelleExecutor.tokenize(str(lemma)))}")
    print('-' * 80)
    while inp != "exit":
        print('=' * 80)
        inp = input("Query: ")
        scores = [idx for idx in enumerate(isabelle_bm25_reranker.get_scores(inp))]
        # Sort by score
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # Print the top 10
        for idx, score in scores[:10]:
            print(f"[{score}]: {isabelle_bm25_reranker.responses[idx]}")
