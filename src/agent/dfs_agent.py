#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import time
import logging
from src.agent.simple_proof_agent import ProofAgent
from src.rl.simple_proof_env import ProofEnv
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.agent.dfs_tree_search import DFSTreeSearch
from src.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from src.agent.dfs_policy_prompter import DfsCoqGptPolicyPrompter


if __name__ == "__main__":
    os.chdir(root_dir)
    os.makedirs(".log", exist_ok=True)
    log_path = ".log/proof_agent-{}.log".format(time.strftime("%Y%m%d-%H%M%S"))
    proof_file = "data/test/SimpleAlgebra.v"
    theorem_name = "algb_is_abelian_group"
    # proof_file = "data/test/SimpleNaturalProofs.v"
    # theorem_name = "algb_add_comm"
    main_prompt = "data/prompts/system/coq-proof-agent-role.md"
    conv_prompt = "data/prompts/conversation/coq-proof-agent-example-long-conv.md"
    checkpoint_dir = ".log/checkpoints/"
    max_tokens_per_action = 25
    max_theorems_in_prompt = 3
    gpt_model_name = "gpt-4"
    policy_prompter = DfsCoqGptPolicyPrompter(
        main_sys_prompt_path=main_prompt,
        example_conv_prompt_path=conv_prompt,
        max_tokens_per_action=max_tokens_per_action,
        gpt_model_name=gpt_model_name)
    dfs_tree_search = DFSTreeSearch()
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
        with GptGuidedTreeSearchPolicy(
            checkpoint_dir, 
            theorem_name, 
            policy_prompter,
            dfs_tree_search) as policy:
            agent = ProofAgent("proof_agent", policy)
            agent.run(env, 1, 50, True)
    pass