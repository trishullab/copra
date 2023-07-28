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
    policy_prompter = CoqGptPolicyPrompter(
        main_sys_prompt_path="data/prompts/system/coq-proof-agent-role.md",
        example_conv_prompt_path="data/prompts/conversation/coq-proof-agent-example-long-conv.md",
        max_tokens_per_action=25)
    basic_policy = BasicPolicy(policy_prompter, 3)
    agent = ProofAgent("basic_proof_agent", basic_policy)
    proof_exec_callback = ProofExecutorCallback(
        project_folder=".",
        file_path="data/test/SimpleAlgebra.v"
    )
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("basic_proof_env")
    with ProofEnv("basic_proof_env", proof_exec_callback, 'algb_add_comm', max_proof_depth=10, logger=logger) as env:
        agent.run(env, 1, 50, True)
    pass