#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

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