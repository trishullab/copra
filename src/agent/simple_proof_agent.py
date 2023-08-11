#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.rl.abstraction import Agent, Policy
from src.rl.simple_proof_env import ProofEnv


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
            action = self._policy(env.state)
            state, action, next_state, reward, done, info = env.step(action)
            if render:
                env.render()
            self._policy.update(state, action, next_state, reward, done, info)
            state = next_state
            steps += 1
            total_reward += reward
        env.dump_proof()
        self._policy.checkpoint()

    def run(self, env: ProofEnv, episodes: int, max_steps_per_episode: int, render: bool):
        while episodes > 0:
            self.run_episode(env, max_steps_per_episode, render)
            episodes -= 1
        pass