#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from abc import ABC, abstractmethod

class State(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def serialize(self) -> str:
        pass

class Action(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def serialize(self) -> str:
        pass

class Env(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: Action) -> typing.Tuple[State, float, bool, dict]:
        pass

    @abstractmethod
    def render(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @property
    @abstractmethod
    def history(self) -> typing.List[typing.Tuple[State, Action, float, bool, dict]]:
        pass

    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @abstractmethod
    def checkpoint(self):
        pass

    @abstractmethod
    def clone(self):
        pass

class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state: typing.Any) -> Action:
        pass

    @abstractmethod
    def update(self, state: typing.Any, action: Action, reward: float, next_state: typing.Any, done: bool):
        pass

    @abstractmethod
    def checkpoint(self):
        pass

    @abstractmethod
    def clone(self):
        pass

class Agent(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def policy(self) -> Policy:
        pass

    @abstractmethod
    def checkpoint(self):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def run(self, env: Env, episodes: int, render: bool):
        pass