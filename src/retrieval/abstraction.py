#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
from abc import ABC, abstractmethod

class ReRanker(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def language(self) -> str:
        pass

    @property
    @abstractmethod
    def responses(self) -> typing.List[str]:
        pass

    @abstractmethod
    def rerank(self, query: str, responses: typing.List[str]) -> typing.List[float]:
        pass

    @abstractmethod
    def get_scores(self, query: str) -> typing.List[float]:
        pass

    @abstractmethod
    def reindex(self, responses: typing.List[str]) -> None:
        pass