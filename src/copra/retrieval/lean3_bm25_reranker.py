#!/usr/bin/env python3

import typing
import copy
from rank_bm25 import BM25Okapi
from copra.lean_server.lean_utils import Lean3Utils
from itp_interface.tools.lean_cmd_executor import Lean3Executor
from copra.retrieval.abstraction import ReRanker

class Lean3Bm25ReRanker(ReRanker):
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
        tokenized_index_data = [list(Lean3Executor.tokenize(response)) for response in responses]
        if len(tokenized_index_data) == 0:
            # print("WARNING: No tokenizable responses found. Returning all 0.0 scores.")
            # print("Query:", query)
            # print("Responses:", responses)
            return [0.0] * len(responses)
        bm25 = BM25Okapi(tokenized_index_data, k1=self.k1, b=self.b)
        query_tokens = list(Lean3Executor.tokenize(query))
        scores = bm25.get_scores(query_tokens)
        return [float(score) for score in scores]
    
    def get_scores(self, query: str) -> typing.List[float]:
        assert self._responses is not None, "Responses not set. Please call reindex(responses) first."
        assert self.bm25 is not None, "BM25 not initialized. Please call reindex(responses) first."
        query_tokens = list(Lean3Executor.tokenize(query))
        scores = self.bm25.get_scores(query_tokens)
        return [float(score) for score in scores]
    
    def reindex(self, responses: typing.List[str]) -> None:
        self._responses = copy.deepcopy(responses)
        tokenized_index_data = [list(Lean3Executor.tokenize(response)) for response in responses]
        self.bm25 = BM25Okapi(tokenized_index_data, k1=self.k1, b=self.b)

if __name__ == "__main__":
    from copra.lean_server.lean3_search_tool import Lean3SearchTool
    from itp_interface.tools.lean_cmd_executor import Constants

    def take_multiline_input() -> str:
        inp = []
        x = None
        while x != "":
            if x is not None:
                inp.append(x)
            x = input()
        inp = "\n".join(inp)
        return inp

    mathlib_path = "data/benchmarks/miniF2F/_target/deps/mathlib"
    lean3_search_tool = Lean3SearchTool(mathlib_path=mathlib_path)
    lean3_search_tool.initialize()
    # for root, dirs, files in os.walk(mathlib_src_path):
    #     for file in files:
    #         if file.endswith(".lean"):
    #             namespace = root.split(mathlib_src_path)[1].strip("/")
    #             namespace = namespace.replace("/", ".")
    #             namespace = namespace + "." + file.replace(".lean", "")
    #             lean3_search_tool.add(os.path.join(root, file), namespace)
    lean3_bm25_reranker = Lean3Bm25ReRanker()
    lean3_bm25_reranker.reindex([str(lemma) for lemma in lean3_search_tool.lemmas])
    inp = ""
    print("Tokenized lemmas:")
    print('-' * 80)
    for lemma in lean3_search_tool.lemmas[:20]:
        print(f"{lemma}: \n{list(Lean3Executor.tokenize(str(lemma)))}")
    print('-' * 80)
    while inp != "exit":
        print('=' * 80)
        print("Enter query:")
        inp = take_multiline_input()
        scores = [idx for idx in enumerate(lean3_bm25_reranker.get_scores(inp))]
        # Sort by score
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # Print the top 10
        for idx, score in scores[:10]:
            print(f"[{score}]: {lean3_bm25_reranker.responses[idx]}")
