#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.rl.proof_action import ProofAction
from src.rl.abstraction import Agent, Policy
from src.rl.simple_proof_env import ProofEnv


class ProofAgent(Agent):
    def __init__(self, name: str, policy: Policy, should_checkpoint: bool = False):
        self._policy = policy
        self._name = name
        self._should_checkpoint = should_checkpoint
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
        next_state = env.state
        while not done and steps < max_steps_per_episode:
            action = self._policy(next_state)
            assert isinstance(action, ProofAction)
            if action.action_type != ProofAction.ActionType.EXIT:
                state, _, next_state, reward, done, info = env.step(action)
                if render:
                    env.render()
                if action.action_type != ProofAction.ActionType.BACKTRACK:
                    # Don't update policy for backtracking actions, this will create a 
                    # a very nasty loop in the policy.
                    self._policy.update(state, action, next_state, reward, done, info)
                steps += 1
                total_reward += reward
            else:
                break
        env.dump_proof()
        if self._should_checkpoint:
            self._policy.checkpoint()

    def run(self, env: ProofEnv, episodes: int, max_steps_per_episode: int, render: bool):
        while episodes > 0:
            self.run_episode(env, max_steps_per_episode, render)
            episodes -= 1
        pass