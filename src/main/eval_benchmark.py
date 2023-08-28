#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import hydra
import logging
import os
import typing
from src.agent.dfs_policy_prompter import DfsCoqGptPolicyPrompter
from src.agent.dfs_tree_search_with_stack import DFSTreeSearch
from src.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from src.agent.simple_proof_agent import ProofAgent
from src.baselines.gpt4.few_shot_policy import FewShotGptPolicy
from src.baselines.gpt4.few_shot_policy_prompter import FewShotGptPolicyPrompter
from src.main.config import EvalBenchmark, EvalDataset, EvalSettings, PolicyName, parse_config
from src.prompt_generator.prompter import PolicyPrompter
from src.rl.abstraction import Policy
from src.rl.simple_proof_env import ProofEnv
from src.rl.proof_search_result import ProofSearchResult
from src.tools.proof_exec_callback import ProofExecutorCallback

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

def eval_dataset(dataset: EvalDataset, eval_settings: EvalSettings, proof_results : typing.Dict[str, ProofSearchResult], logger: logging.Logger = None):
    logger = logger or logging.getLogger(__name__)
    for file in dataset.files:
        path = os.path.join(dataset.project, file.path)
        proof_dump_file_name = f"{eval_settings.proof_dump_file_prefix}{path.replace('/', '_')}.txt"
        coq_proof_exec_callback = ProofExecutorCallback(
            project_folder=dataset.project,
            file_path=path,
            use_hammer=eval_settings.use_hammer,
            timeout_in_secs=eval_settings.timeout_in_secs,
            use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
            suppress_error_log=True,
            logger=logger)
        lemmas_to_prove = set(get_all_lemmas(coq_proof_exec_callback))
        if isinstance(file.theorems, str) and file.theorems == "*":
            file.theorems = lemmas_to_prove
        elif isinstance(file.theorems, list):
            file.theorems = list(set(file.theorems).intersection(lemmas_to_prove))
        else:
            raise ValueError(f"Invalid theorems: {file.theorems}")
        logger.info(f"Discovered {len(lemmas_to_prove)} lemmas to prove in {path}")
        logger.info(f"Lemmas to prove in file {path}: \n{lemmas_to_prove}")
        for lemma_name in lemmas_to_prove:
            logger.info(f"Attempting to prove lemma: {lemma_name}")
            search_guidance_policy : Policy = None
            policy_prompter : PolicyPrompter = None
            if eval_settings.policy_name == PolicyName.Dfs:
                policy_prompter = DfsCoqGptPolicyPrompter(
                    main_sys_prompt_path=eval_settings.main_prompt,
                    example_conv_prompt_path=eval_settings.conv_prompt,
                    max_tokens_per_action=eval_settings.max_tokens_per_action,
                    gpt_model_name=eval_settings.gpt_model_name,
                    temperature=eval_settings.temperature,
                    max_history_messages=eval_settings.max_history_messages,
                    k=eval_settings.max_theorems_in_prompt) # k is the number of theorems to consider at each step
                dfs_tree_search = DFSTreeSearch()
                search_guidance_policy = GptGuidedTreeSearchPolicy(
                    eval_settings.checkpoint_dir, 
                    lemma_name, 
                    policy_prompter,
                    dfs_tree_search,
                    checkpoint_on_exit=eval_settings.should_checkpoint)
            elif eval_settings.policy_name == PolicyName.FewShot:
                policy_prompter = FewShotGptPolicyPrompter(
                    main_sys_prompt_path=eval_settings.main_prompt,
                    example_conv_prompt_path=eval_settings.conv_prompt,
                    temperature=eval_settings.temperature,
                    max_tokens_per_action=eval_settings.max_tokens_per_action,
                    max_history_messages=eval_settings.max_history_messages,
                    gpt_model_name=eval_settings.gpt_model_name,
                    k=eval_settings.max_theorems_in_prompt,
                    logger=logger)
                search_guidance_policy = FewShotGptPolicy(
                    eval_settings.checkpoint_dir,
                    lemma_name,
                    policy_prompter,
                    checkpoint_on_exit=eval_settings.should_checkpoint,
                    logger=logger)
            else:
                raise Exception(f"Unknown policy name: {eval_settings.policy_name}")
            with ProofEnv(f"basic_proof_env_{lemma_name}", coq_proof_exec_callback, lemma_name, max_proof_depth=eval_settings.max_proof_depth, logger=logger) as env:
                with search_guidance_policy:
                    agent = ProofAgent(f"proof_agent_{lemma_name}", search_guidance_policy, eval_settings.should_checkpoint, proof_dump_file_name, logger=logger)
                    agent.run(env, episodes=eval_settings.max_number_of_episodes, max_steps_per_episode=eval_settings.max_steps_per_episode, render=eval_settings.render)
                proof_results[(path, lemma_name)] = env.proof_search_res
            logger.info(f"Finished the attempt for proving lemma: {lemma_name} in file {path}")
    pass

def measure_success(proof_results : typing.Dict[str, ProofSearchResult], logger: logging.Logger = None):
    success_count = 0
    for (path, lemma_name), proof_res in proof_results.items():
        if proof_res.proof_found:
            success_count += 1
            logger.info(f"Proof found for lemma: {lemma_name} in file {path}")
        else:
            logger.info(f"Proof not found for lemma: {lemma_name} in file {path}")
        logger.info(f"Proof/Incomplete proof: \n{proof_res}")
    logger.info(f"Success rate: {success_count}/{len(proof_results)}")

def eval_benchmark(benchmark: EvalBenchmark, eval_settings: EvalSettings, logger: logging.Logger = None):
    logger = logger or logging.getLogger(__name__)
    proof_results : typing.Dict[str, ProofSearchResult] = {}
    for dataset in benchmark.datasets:
        eval_dataset(dataset, eval_settings, proof_results, logger=logger)
    measure_success(proof_results, logger=logger)

@hydra.main(config_path="config", config_name="experiments", version_base="1.2")
def main(cfg):
    experiment = parse_config(cfg)
    eval_benchmark(experiment.benchmark, experiment.eval_settings)
    pass

if __name__ == "__main__":
    main()