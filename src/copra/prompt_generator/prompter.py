#!/usr/bin/env python3

import typing
from abc import ABC, abstractmethod

class PolicyPrompter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run_prompt(self, requests: typing.Any) -> typing.Any:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> typing.Any:
        pass

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {}