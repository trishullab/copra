#!/usr/bin/env python3

import logging
import typing
import copy
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.abstraction import Agent, Policy
from itp_interface.rl.simple_proof_env import ProofEnv, ProgressState, ProofState
from copra.agent.handle_have_tactic import HandleHaveTactic


class ProofAgent(Agent):
    def __init__(self, 
        name: str, 
        policy: Policy, 
        should_checkpoint: bool = False, 
        proof_dump_file_name: str = None, 
        logger: logging.Logger = None):
        self._policy = policy
        self._name = name
        self._should_checkpoint = should_checkpoint
        self._proof_dump_file_name = proof_dump_file_name
        self.logger = logger or logging.getLogger(__name__)
        pass

    @property
    def name(self) -> str:
        return self._name

    def checkpoint(self):
        pass

    def clone(self):
        pass

    def run_episode(self, env: ProofEnv, max_steps_per_episode: int, render: bool):
        def _stop_policy(steps: int, info: typing.Dict[str, typing.Any]):
            return steps >= max_steps_per_episode
        def _policy_info_message(steps: int, info: typing.Dict[str, typing.Any]):
            return f"Step {steps}/{max_steps_per_episode}"
        self._run_episode_as_per_policy(env, _stop_policy, _policy_info_message, render)

    def run(self, env: ProofEnv, episodes: int, max_steps_per_episode: int, render: bool):
        assert isinstance(env, ProofEnv)
        while episodes > 0:
            self.run_episode(env, max_steps_per_episode, render)
            episodes -= 1
        pass

    def run_episodes_till_stop(self, env: ProofEnv, episodes: int, render: bool, 
        stop_policy: typing.Callable[[int, typing.Dict[str, typing.Any]], bool], 
        policy_info_message: typing.Callable[[int, typing.Dict[str, typing.Any]], str]):
        assert isinstance(env, ProofEnv)
        while episodes > 0:
            self._run_episode_as_per_policy(env, stop_policy, policy_info_message, render)
            episodes -= 1

    def _run_episode_as_per_policy(self, 
            env: ProofEnv, 
            stop_policy: typing.Callable[[int, typing.Dict[str, typing.Any]], bool],
            policy_info_message: typing.Callable[[int, typing.Dict[str, typing.Any]], str],
            render: bool):
        env.reset()
        done = False
        steps = 0
        total_reward = 0
        next_state = env.state
        additional_info = self._policy.get_efficiency_info()
        while not done and not stop_policy(steps, additional_info):
            self.logger.info(policy_info_message(steps, additional_info))
            self.logger.info("Asking policy for next action")
            action = self._policy(next_state)
            assert isinstance(action, ProofAction)
            self.logger.info(f"Got Action:\n{action}")
            if action.action_type != ProofAction.ActionType.EXIT:
                state, modified_action, next_state, reward, done, info = env.step(action)
                if render:
                    self.logger.info("**"*20)
                    env.render()
                    self.logger.info("**"*20)
                if action.action_type != ProofAction.ActionType.BACKTRACK:
                    # Don't update policy for backtracking actions, this will create a 
                    action_was_modified = False
                    if env.language == ProofAction.Language.LEAN4 and \
                    action.action_type == ProofAction.ActionType.RUN_TACTIC and \
                    isinstance(modified_action, ProofAction) and \
                    modified_action.kwargs.get("modified", False):
                        # Specially change the last action with modified action
                        self.logger.info("Resetting last action in policy with modified action")
                        modified_action.original_message = f"[RUN TACTIC]\n{'\n'.join(modified_action.kwargs['tactics'])}\n[END]"
                        self.logger.info(f"Modified Action:\n{modified_action}")
                        self._policy.reset_last_action(modified_action)
                        action_was_modified = True
                    # a very nasty loop in the policy.
                    self.logger.info("Updating policy")
                    if action_was_modified:
                        self._policy.update(state, modified_action, next_state, reward, done, info)
                        self.logger.info("Policy updated with modified action")
                    else:
                        self._policy.update(state, action, next_state, reward, done, info)
                    self.logger.info("Policy updated")
                steps += 1
                total_reward += reward
                additional_info = self._policy.get_efficiency_info()
            else:
                self.logger.warning("Got EXIT action, exiting")
                break
        env.dump_proof(self._proof_dump_file_name, additional_info)
        if self._should_checkpoint:
            self.logger.info("Checkpointing policy")
            self._policy.checkpoint()
