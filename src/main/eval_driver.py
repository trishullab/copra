#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import typing
from src.agent.dfs_policy_prompter import DfsCoqGptPolicyPrompter
from src.agent.dfs_tree_search_with_stack import DFSTreeSearch
from src.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from src.rl.proof_search_result import ProofSearchResult
from src.agent.simple_proof_agent import ProofAgent
from src.rl.simple_proof_env import ProofEnv
from src.tools.proof_exec_callback import ProofExecutorCallback
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class EvalSettings(object):
    project_folder: str
    file_path: str
    use_hammer: bool
    max_proof_depth: int = 50
    timeout_in_secs: int = 60
    proof_retries: int = 1
    main_prompt: str = "data/prompts/system/coq-proof-agent-with-dfs.md"
    conv_prompt: str = "data/prompts/conversation/coq-proof-agent-example-long-conv-dfs.md"
    max_tokens_per_action: int = 25
    max_theorems_in_prompt: int = 3
    gpt_model_name: str = "gpt-3.5-turbo"
    max_number_of_episodes: int = 1
    max_steps_per_episode: int = 50
    render: bool = False
    checkpoint_dir: str = ".log/checkpoints"
    should_checkpoint: bool = False
    temperature: float = 0.0
    max_history_messages: int = 0

def get_all_lemmas(coq_proof_exec_callback: ProofExecutorCallback):
    lemmas_to_prove = []
    with coq_proof_exec_callback.get_proof_executor() as main_executor:
        while not main_executor.execution_complete:
            assert not main_executor.is_in_proof_mode(), "main_executor must not be in proof mode"
            _ = list(main_executor.run_till_next_lemma_return_exec_stmt())
            if main_executor.execution_complete:
                break
            lemma_name = main_executor.get_lemma_name_if_running()
            if lemma_name is None:
                _ = list(main_executor.run_to_finish_lemma_return_exec())
                if main_executor.execution_complete:
                    break
            else:
                lemmas_to_prove.append(lemma_name)
                main_executor.run_to_finish_lemma()
    return lemmas_to_prove

# Code to create the evaluation driver based on theorems to be proven and search strategies
def eval_project(
        eval_settings: EvalSettings,
        logger: logging.Logger = None):
    coq_proof_exec_callback = ProofExecutorCallback(
        project_folder=eval_settings.project_folder,
        file_path=eval_settings.file_path,
        use_hammer=eval_settings.use_hammer,
        timeout_in_secs=eval_settings.timeout_in_secs,
        use_human_readable_proof_context=True,
        suppress_error_log=True,
        logger=logger)
    lemmas_to_prove = get_all_lemmas(coq_proof_exec_callback)
    logger.info(f"Discovered {len(lemmas_to_prove)} lemmas to prove in {eval_settings.file_path}")
    logger.info(f"Lemmas to prove: {lemmas_to_prove}")
    proof_results : typing.Dict[str, ProofSearchResult] = {}
    success_count = 0
    for lemma_name in lemmas_to_prove:
        logger.info(f"Attempting to prove lemma: {lemma_name}")
        policy_prompter = DfsCoqGptPolicyPrompter(
            main_sys_prompt_path=eval_settings.main_prompt,
            example_conv_prompt_path=eval_settings.conv_prompt,
            max_tokens_per_action=eval_settings.max_tokens_per_action,
            gpt_model_name=eval_settings.gpt_model_name,
            temperature=eval_settings.temperature,
            max_history_messages=eval_settings.max_history_messages,
            k=eval_settings.max_theorems_in_prompt) # k is the number of theorems to consider at each step
        dfs_tree_search = DFSTreeSearch()
        with ProofEnv(f"basic_proof_env_{lemma_name}", coq_proof_exec_callback, lemma_name, max_proof_depth=eval_settings.max_proof_depth, logger=logger) as env:
            with GptGuidedTreeSearchPolicy(
                eval_settings.checkpoint_dir, 
                lemma_name, 
                policy_prompter,
                dfs_tree_search,
                checkpoint_on_exit=eval_settings.should_checkpoint) as policy:
                agent = ProofAgent(f"proof_agent_{lemma_name}", policy)
                agent.run(env, episodes=eval_settings.max_number_of_episodes, max_steps_per_episode=eval_settings.max_steps_per_episode, render=eval_settings.render)
            proof_results[lemma_name] = env.proof_search_res
        logger.info(f"Finished the attempt for proving lemma: {lemma_name}")
    
    for lemma_name, proof_res in proof_results.items():
        if proof_res.proof_found:
            success_count += 1
            logger.info(f"Proof found for lemma: {lemma_name}")
        else:
            logger.info(f"Proof not found for lemma: {lemma_name}")
        logger.info(f"Proof/Incomplete proof: \n{proof_res}")
    logger.info(f"Success rate: {success_count}/{len(lemmas_to_prove)}")

if __name__ == "__main__":
    import os
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_folder", type=str, default=".", help="Project folder")
    parser.add_argument("--file_path", type=str, default="data/test/SimpleAlgebra.v", help="File path")
    parser.add_argument("--use_hammer", type=bool, default=False, help="Use hammer")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="GPT model name")
    parser.add_argument("--max_theorems_in_prompt", type=int, default=5, help="Max theorems in prompt")
    parser.add_argument("--max_steps_per_episode", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--render", type=bool, default=False, help="Render")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max_history_messages", type=int, default=4, help="Max history messages")
    args = parser.parse_args()
    os.chdir(root_dir)
    os.makedirs(".log", exist_ok=True)
    os.makedirs(".log/evals", exist_ok=True)
    log_path = ".log/evals/{}.log".format(time.strftime("%Y%m%d-%H%M%S"))
    eval_settings = EvalSettings(
        project_folder=args.project_folder,
        file_path=args.file_path,
        use_hammer=args.use_hammer,
        gpt_model_name=args.model_name,
        max_theorems_in_prompt=args.max_theorems_in_prompt,
        max_steps_per_episode=args.max_steps_per_episode,
        temperature=args.temperature,
        render=args.render,
        max_history_messages=args.max_history_messages
    )
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("eval_driver")
    logger.info(f"eval_settings: {eval_settings.to_json()}")
    eval_project(eval_settings, logger=logger)
    pass