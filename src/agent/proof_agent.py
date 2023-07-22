#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from abc import ABC, abstractmethod
from src.rl.abstraction import Env, Agent, Policy
from src.rl.proof_env import ProofEnv, ProofEnvInfo


class ProofAgent(Agent):
    def __init__(self, name: str, policy: Policy):
        self._policy = policy
        self._name = name
        pass

    @property
    def name(self) -> str:
        return self._name

    def checkpoint(self):
        pass

    def clone(self):
        pass

    def run_episode(self, env: ProofEnv, max_steps_per_episode: int, render: bool):
        env.reset()
        done = False
        steps = 0
        total_reward = 0
        while not done and steps < max_steps_per_episode:
            action = self.policy(env)
            state, action, next_state, reward, done, info = env.step(action)
            if render:
                env.render()
            self._policy.update(state, action, next_state, reward, done, info)
            state = next_state
            steps += 1
            total_reward += reward
        pass

    def run(self, env: ProofEnv, episodes: int, max_steps_per_episode: int, render: bool):
        while episodes > 0:
            self.run_episode(env, max_steps_per_episode, render)
            episodes -= 1
        pass