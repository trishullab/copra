#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.rl.proof_env import ProofEnv
from src.prompt_generator.agent_grammar import CoqGptResponse, CoqGptResponseActions
from src.agent.coq_policy_prompter import CoqGptPolicyPrompter, InvalidActionException
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
            tdf = s2.training_data_format
            if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.RUN_TACTIC_RESULT, 
                training_data_format = tdf)
            elif action.action_type == ProofAction.ActionType.GET_DFNS:
                for goal in tdf.start_goals:
                    goal.relevant_defns = goal.relevant_defns[:self.k]
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.GET_DFNS_RESULT, 
                training_data_format = tdf)
            elif action.action_type == ProofAction.ActionType.GET_THMS:
                for goal in tdf.start_goals:
                    goal.possible_useful_theorems_local = goal.possible_useful_theorems_local[:self.k]
                    goal.possible_useful_theorems_external = goal.possible_useful_theorems_external[:self.k]
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.GET_THMS_RESULT, 
                training_data_format = tdf)
            else:
                raise Exception(f"Invalid action type: {action.action_type}")
        else:
            state = env.state
            gpt_response = CoqGptResponse(action = CoqGptResponseActions.GLS, 
            training_data_format = state.training_data_format)
        success = False
        tries = 10
        exceptions = []
        while not success and tries > 0:
            try:
                responses = self.prompter.run_prompt(gpt_response)
                actions_tuple = self.prompter.parse_response(responses)
                chosen_message = actions_tuple[0][0]
                self.prompter.add_to_history(chosen_message)
                success = True
            except InvalidActionException as e:
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.ERROR, 
                message=e.message)
                chosen_message = responses[0]
                self.prompter.add_to_history(chosen_message)
                exceptions.append(e)
            tries -= 1
        if not success:
            raise Exception(f"Failed to get valid action after {tries} tries. Exceptions:\n {exceptions}")
        action = actions_tuple[0][1]
        return action

    def update(self, state: State, action: Action, next_state: State, reward: float, done: bool, info: typing.Any):
        pass

    def checkpoint(self):
        pass

    def clone(self):
        pass