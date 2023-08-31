#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
from rank_bm25 import BM25Okapi
from src.tools.coq_executor import CoqExecutor
from src.retrieval.abstraction import ReRanker

class CoqBm25ReRanker(ReRanker):
    def __init__(self, k1: float = 1.0, b: float = 0.75, epsilon: float = 0.25) -> None:
        super().__init__()
        self.bm25: BM25Okapi = None
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

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