#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.rl.abstraction import Agent, Policy
from src.rl.proof_env import ProofEnv


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
            action = self._policy(env)
            state, action, next_state, reward, done, info = env.step(action)
            if render:
                env.render()
            self._policy.update(state, action, next_state, reward, done, info)
            state = next_state
            steps += 1
            total_reward += reward
        env.dump_proof()

    def run(self, env: ProofEnv, episodes: int, max_steps_per_episode: int, render: bool):
        while episodes > 0:
            self.run_episode(env, max_steps_per_episode, render)
            episodes -= 1
        pass

if __name__ == "__main__":
    import os
    import time
    import logging
    from src.agent.basic_policy import BasicPolicy
    from src.agent.coq_policy_prompter import CoqGptPolicyPrompter

    os.chdir(root_dir)
    os.makedirs(".log", exist_ok=True)
    log_path = ".log/proof_agent-{}.log".format(time.strftime("%Y%m%d-%H%M%S"))
    proof_file = "data/test/SimpleAlgebra.v"
    theorem_name = "algb_is_abelian_group"
    # proof_file = "data/test/SimpleNaturalProofs.v"
    # theorem_name = "algb_add_comm"
    main_prompt = "data/prompts/system/coq-proof-agent-role.md"
    conv_prompt = "data/prompts/conversation/coq-proof-agent-example-long-conv.md"
    max_tokens_per_action = 25
    max_theorems_in_prompt = 3
    gpt_model_name = "gpt-4"
    policy_prompter = CoqGptPolicyPrompter(
        main_sys_prompt_path=main_prompt,
        example_conv_prompt_path=conv_prompt,
        max_tokens_per_action=max_tokens_per_action,
        gpt_model_name=gpt_model_name)
    basic_policy = BasicPolicy(policy_prompter, max_theorems_in_prompt)
    agent = ProofAgent("basic_proof_agent", basic_policy)
    proof_exec_callback = ProofExecutorCallback(
        project_folder=".",
        file_path=proof_file
    )
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("basic_proof_env")
    logger.info("Starting proof agent with " + 
                f"\nfile = {proof_file}," + 
                f"\ntheorem = {theorem_name}" +
                f"\nmain_prompt = {main_prompt}" +
                f"\nconv_prompt = {conv_prompt}" +
                f"\nmax_tokens_per_action = {max_tokens_per_action}" +
                f"\nmax_theorems_in_prompt = {max_theorems_in_prompt}" +
                f"\ngpt_model_name = {gpt_model_name}")
    with ProofEnv("basic_proof_env", proof_exec_callback, theorem_name, max_proof_depth=20, logger=logger) as env:
        agent.run(env, 1, 50, True)
    pass