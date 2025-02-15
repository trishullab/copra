#!/usr/bin/env python3

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