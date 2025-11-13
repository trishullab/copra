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
        max_retry_count = 1
        validation_failed = True
        while max_retry_count > 0 and validation_failed:
            env.reset()
            if env.language == ProofAction.Language.LEAN4:
                env.set_max_proof_step_length(3000)
            done = False
            steps = 0
            total_reward = 0
            next_state = env.state
            additional_info = self._policy.get_efficiency_info()
            while not done and not stop_policy(steps, additional_info):
                _, _, next_state, opt_done, steps, additional_info = self._run_single_proof_step(
                    policy_info_message,
                    env, 
                    next_state, 
                    steps, 
                    total_reward, 
                    additional_info, 
                    render)
                done = opt_done
                if opt_done is None:
                    break
            proof_search_result = env.collect_proof_search_result(additional_info)
            if proof_search_result.proof_found:
                # Validate the proof
                self.logger.info("Validating the proof found")
                val_result = env.validate_proof_completion()
                is_valid_proof = val_result.get("compilation_ok", False)
                is_valid_proof = is_valid_proof and val_result.get("success", False)
                has_sorries = val_result.get("has_sorries", True)
                std_err = val_result.get("std_err", "")
                if has_sorries or not is_valid_proof:
                    proof_search_result.proof_found = False
                    if isinstance(proof_search_result.additional_info, dict):
                        proof_search_result.additional_info["validation_error"] = val_result.get("error_message", "Unknown validation error")
                        proof_search_result.additional_info["std_err"] = std_err
                        proof_search_result.additional_info["has_sorries"] = has_sorries
                        proof_search_result.additional_info["compilation_ok"] = is_valid_proof
                    self.logger.warning(
                        f"Proof found but validation failed for lemma: {env.lemma_name}. "
                        f"Compilation OK: {is_valid_proof}, Has Sorries: {has_sorries}. StdErr: {std_err}")
                    self.logger.info("Retrying proof search...")
                    validation_failed = True
            else:
                validation_failed = False
            max_retry_count -= 1
        env.dump_proof(self._proof_dump_file_name, additional_info)
        if self._should_checkpoint:
            self.logger.info("Checkpointing policy")
            self._policy.checkpoint()

    def _run_single_proof_step(self, 
        policy_info_message: typing.Callable[[int, typing.Dict[str, typing.Any]], str],
        env: ProofEnv,
        next_state: ProofState,
        steps: int,
        total_reward: float,
        additional_info: typing.Dict[str, typing.Any],
        render: bool
        ) -> tuple[ProofState| None, ProofAction | None, ProofState | None, bool|None, int, typing.Dict[str, typing.Any]]:
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
                # a very nasty loop in the policy.
                action_was_modified = False
                if env.language == ProofAction.Language.LEAN4 and \
                action.action_type == ProofAction.ActionType.RUN_TACTIC:
                    if isinstance(modified_action, ProofAction) and \
                    modified_action.kwargs.get("modified", False):
                        # Specially change the last action with modified action
                        self.logger.info("Resetting last action in policy with modified action")
                        actions_joined = "\n".join(modified_action.kwargs['tactics'])
                        modified_action.original_message = f"[RUN TACTIC]{actions_joined}[END]"
                        self.logger.info("Modified Action:\n" + f"{modified_action}")
                        self._policy.reset_last_action(modified_action)
                        action_was_modified = True
                    reduction_percentage = 0.05
                    # If the last action failed then reduce the proof step length limit
                    if info.progress == ProgressState.FAILED:
                        max_proof_len = env.max_proof_step_length()
                        if max_proof_len is not None:
                            max_proof_len = int(max_proof_len*(1 - reduction_percentage))
                            env.set_max_proof_step_length(max(775, max_proof_len))
                    else:
                        # If the last action succeeded then increase the proof step length limit
                        max_proof_len = env.max_proof_step_length()
                        if max_proof_len is not None:
                            max_proof_len = int(max_proof_len*(1 + reduction_percentage))
                            env.set_max_proof_step_length(min(3000, max_proof_len))
                    self.logger.info(f"New proof step length limit: {env.max_proof_step_length()}")
                    
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
            return state, modified_action, next_state, done, steps, additional_info
        else:
            self.logger.warning("Got EXIT action, exiting")
            return None, None, None, None, steps, additional_info