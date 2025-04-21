#!/usr/bin/env python3

import typing
from abc import ABC, abstractmethod

class PolicyPrompter(ABC):
    def __init__(self):
        pass

    def reset_last_action(self, action):
        """
        Resets the last action taken by the policy.
        """
        pass

    def add_delayed(self, action):
        """
        Adds a delayed action to the policy.
        """
        pass

    @abstractmethod
    def run_prompt(self, requests: typing.Any) -> typing.Any:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> typing.Any:
        pass

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {}
    
    def fix_action(self, action: typing.Any):
        return action