#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from abc import ABC, abstractmethod

class State(object):
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

class Action(object):
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

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

class QFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state: State, action: Action) -> typing.Tuple[float, typing.Any]:
        pass

    @abstractmethod
    def update(self, state: State, action: Action, next_state: State, reward: float, done: bool, info: typing.Any):
        pass

    @abstractmethod
    def checkpoint(self):
        pass

    @abstractmethod
    def clone(self):
        pass

class Env(ABC):
    def __init__(self):
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
    def done(self) -> bool:
        pass

    @property
    @abstractmethod
    def history(self) -> typing.List[typing.Tuple[State, Action, State, float, bool, typing.Any]]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: Action) -> typing.Tuple[State, Action, State, float, bool, typing.Any]:
        pass

    @abstractmethod
    def render(self):
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
    def __call__(self, env: Env) -> Action:
        pass

    @abstractmethod
    def update(self, state: State, action: Action, next_state: State, reward: float, done: bool, info: typing.Any):
        pass

    @abstractmethod
    def checkpoint(self):
        pass

    @abstractmethod
    def clone(self):
        pass

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {}

class Agent(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def checkpoint(self):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def run(self, env: Env, episodes: int, max_steps_per_episode: int, render: bool):
        pass

    @abstractmethod
    def run_episodes_till_stop(self, env: Env, episodes: int, render: bool, 
        stop_policy: typing.Callable[[int, typing.Dict[str, typing.Any]], bool], 
        policy_info_message: typing.Callable[[int, typing.Dict[str, typing.Any]], str]):
        pass