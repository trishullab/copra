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
        lean_hack = HandleHaveTactic()
        done = False
        steps = 0
        total_reward = 0
        next_state = env.state
        additional_info = self._policy.get_efficiency_info()
        action_fixed = True
        while not done and not stop_policy(steps, additional_info):
            self.logger.info(policy_info_message(steps, additional_info))
            self.logger.info("Asking policy for next action")
            action = self._policy(next_state)
            assert isinstance(action, ProofAction)
            self.logger.info(f"Got Action: {action}")
            action = lean_hack.parse_have_tactic_action(action)
            self.logger.info(f"Parsed Action: {action}")
            if action.action_type != ProofAction.ActionType.EXIT:
                action_fixed, indented_action = lean_hack.fix_action(action, self.logger)
                assert action_fixed, f"Action {action} is not fixed"
                modified_action = copy.deepcopy(action)
                has_more_than_one_tactic = len(action.kwargs.get('tactics', [])) > 1
                is_have_tactic = lean_hack.is_single_line_have_tactic(action)
                if modified_action.action_type == ProofAction.ActionType.RUN_TACTIC:
                    modified_action.kwargs['tactics'] = [indented_action.kwargs['tactics'][0]]
                    if len(action.kwargs.get('tactics', [])) > 1:
                        self._policy.reset_last_action(modified_action)
                self.logger.info(f"Action to run: {modified_action}")
                state, modified_action, next_state, reward, done, info = env.step(modified_action)
                # **IMPORTANT NOTE**: Here we update the action because sometimes the proof env can optimize the action
                # and return a different action which kind of aligns with the action taken by the
                # policy but only more efficient. This is slightly different traditional RL setting
                ignore_single_line_have = (not has_more_than_one_tactic) and is_have_tactic
                lean_hack.scope_state(state, modified_action, next_state, info, self.logger, ignore_single_line_have)
                if lean_hack.is_within_have_tactic():
                    assert isinstance(next_state, ProofState)
                    if next_state.training_data_format.goal_description is None:
                        next_state.training_data_format.goal_description = ""
                    last_have_tactic = lean_hack.get_last_have_tactic()
                    assert last_have_tactic is not None, f"Last have tactic is None, {lean_hack._have_tactics}"
                    next_state.training_data_format.goal_description += \
                    f"IMPORTANT NOTE: Working on the sub-goal with have tactic: \n{last_have_tactic}."
                if render:
                    self.logger.info("**"*20)
                    env.render()
                    self.logger.info("**"*20)
                if action.action_type != ProofAction.ActionType.BACKTRACK:
                    # Don't update policy for backtracking actions, this will create a 
                    # a very nasty loop in the policy.
                    self.logger.info("Updating policy")
                    self._policy.update(state, action, next_state, reward, done, info)
                    self.logger.info("Policy updated")
                if action.action_type == ProofAction.ActionType.RUN_TACTIC and \
                    info.progress != ProgressState.FAILED and \
                    len(action.kwargs.get('tactics', [])) > 1:
                    remaining_tactics = action.kwargs.get('tactics', [])[1:]
                    remaining_tactics.reverse()
                    for tactic in remaining_tactics:
                        modified_action : ProofAction = copy.deepcopy(action)
                        modified_action.kwargs['tactics'] = [tactic]
                        modified_action.original_message["content"] = f"[RUN TACTIC]\n{tactic}\n[END]"
                        self.logger.info(f"Adding delayed action: {modified_action}")
                        self._policy.add_delayed(modified_action)
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
