#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.rl.proof_env import ProofEnv
from src.prompt_generator.agent_grammar import CoqGptResponse, CoqGptResponseActions
from src.agent.coq_policy_prompter import CoqGptPolicyPrompter
from src.rl.abstraction import Policy, Action, State
from src.rl.proof_action import ProofAction

class BasicPolicy(Policy):
    def __init__(self, prompter: CoqGptPolicyPrompter, k: int = 7):
        assert k > 0, "k must be greater than 0"
        self.prompter = prompter
        self.k = k
        pass

    def __call__(self, env: ProofEnv) -> Action:
        if len(env._history) > 0:
            _, action, s2, _, _, _ = env._history[-1]
            if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.GLS, 
                training_data_format = s2.training_data_format)
            elif action.action_type == ProofAction.ActionType.GET_DFNS:
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.GET_DFNS_RESULT, 
                training_data_format = s2.training_data_format)
            elif action.action_type == ProofAction.ActionType.GET_THMS:
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.GET_THMS_RESULT, 
                training_data_format = s2.training_data_format)
            else:
                raise Exception(f"Invalid action type: {action.action_type}")
        else:
            state = env.state
            gpt_response = CoqGptResponse(action = CoqGptResponseActions.GLS, 
            training_data_format = state.training_data_format)
        responses = self.prompter.run_prompt(gpt_response)
        actions_tuple = self.prompter.parse_response(responses)
        action = actions_tuple[0][0]
        return action

    def update(self, state: State, action: Action, next_state: State, reward: float, done: bool, info: typing.Any):
        pass

    def checkpoint(self):
        pass

    def clone(self):
        pass