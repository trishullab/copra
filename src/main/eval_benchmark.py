#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import hydra
import logging
import os
import random
import time
import math
import typing
import multiprocessing
from src.tools.log_utils import setup_logger
from src.gpts.llama_access import LlamaAccess, ServiceDownError
from src.agent.dfs_policy_prompter import DfsCoqGptPolicyPrompter
from src.agent.dfs_tree_search_with_stack import DFSTreeSearch
from src.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from src.agent.simple_proof_agent import ProofAgent
from src.baselines.gpt4.few_shot_policy import FewShotGptPolicy
from src.baselines.gpt4.few_shot_policy_prompter import FewShotGptPolicyPrompter
from src.main.config import EnvSettings, EvalBenchmark, EvalDataset, EvalProofResults, EvalSettings, Experiments, PolicyName, EvalRunCheckpointInfo, PromptSettings, parse_config
from src.prompt_generator.prompter import PolicyPrompter
from src.rl.abstraction import Policy
from src.rl.proof_tree import ProofSearchResult
from src.rl.simple_proof_env import ProofEnv
from src.rl.proof_action import ProofAction
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.tools.ray_utils import RayUtils
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from src.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor

def check_query_limit_reached(max_query_limit: int) -> typing.Callable[[int, typing.Dict[str, typing.Any]], bool]:
    def _check_query_limit_reached(steps: int, info: typing.Dict[str, typing.Any]):
        return info["queries"] >= max_query_limit
    return _check_query_limit_reached

def query_limit_info_message(max_query_limit: int) -> typing.Callable[[int, typing.Dict[str, typing.Any]], str]:
    def _query_limit_info_message(steps: int, info: typing.Dict[str, typing.Any]):
        return f"Step {info['queries']}/{max_query_limit} (Actual steps: {steps})"
    return _query_limit_info_message

def get_all_lemmas(coq_proof_exec_callback: ProofExecutorCallback, logger: logging.Logger):
    lemmas_to_prove = []
    with coq_proof_exec_callback.get_proof_executor() as main_executor:
        if isinstance(main_executor, DynamicLeanProofExecutor):
            main_executor.run_all_without_exec()
            lemmas_to_prove = main_executor.find_all_theorems_names()
        elif isinstance(main_executor, DynamicCoqProofExecutor):
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
                    logger.info(f"Discovered lemma: {lemma_name}")
                    lemmas_to_prove.append(lemma_name)
                    main_executor.run_to_finish_lemma()
    logger.info(f"Discovered {len(lemmas_to_prove)} lemmas")
    return lemmas_to_prove

def eval_dataset(env_settings: EnvSettings, eval_benchmark: EvalBenchmark, prompt_settings: PromptSettings, dataset: EvalDataset, eval_settings: EvalSettings, eval_checkpoint_info: EvalRunCheckpointInfo, eval_proof_results: EvalProofResults, logger: logging.Logger = None):
    logger = logger or logging.getLogger(__name__)
    if not eval_settings.gpt_model_name.startswith("gpt"):
        llama_logger = setup_logger(__name__ + "_llama", os.path.join(eval_checkpoint_info.logging_dirs[-1], "llama.log"), logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # This is a llama model
        LlamaAccess.class_init(eval_settings.gpt_model_name, eval_settings.temperature, debug=False, logger=llama_logger)
    for file in dataset.files:
        path = os.path.join(dataset.project, file.path)
        proof_dump_file_name = os.path.join(eval_settings.proof_dump_dir, f"{path.replace('/', '_')}.txt")
        if not os.path.exists(proof_dump_file_name):
            with open(proof_dump_file_name, "w") as f:
                f.write(f"File: {path}\n")
                f.write(f"Dataset:\n {dataset.to_json(indent=4)}\n")
                f.write(f"Evaluation Settings:\n {eval_settings.to_json(indent=4)}\n")
        eval_checkpoint_info.add_path_to_maps(path)
        eval_proof_results.add_path_to_maps(path)
        proof_exec_callback = ProofExecutorCallback(
            project_folder=dataset.project,
            file_path=path,
            language=eval_benchmark.language,
            use_hammer=eval_settings.use_hammer,
            timeout_in_secs=eval_settings.timeout_in_secs,
            use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
            suppress_error_log=True,
            always_use_retrieval=eval_settings.always_use_useful_theorem_retrieval,
            logger=logger)
        def _get_all_lemmas(ret_dict, logger):
            try:
                ret_dict["lemmas"] = get_all_lemmas(proof_exec_callback, logger)
            except:
                logger.exception(f"Exception occurred while getting all lemmas in file: {path}")
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        file_time_out = eval_settings.timeout_in_secs * eval_settings.max_proof_depth * 50
        logger.info(f"Getting all lemmas in file: {path} with timeout: {file_time_out} seconds")
        p = multiprocessing.Process(target=_get_all_lemmas, args=(return_dict, logger))
        p.start()
        p.join(file_time_out)
        if p.is_alive():
            p.kill()
            p.join()
        p.close()
        if "lemmas" not in return_dict:
            logger.info(f"Failed to get all lemmas in file: {path}, moving on to the next file.")
            continue
        lemmas_to_prove = return_dict["lemmas"]
        if isinstance(file.theorems, str) and file.theorems == "*":
            file.theorems = list(lemmas_to_prove)
        elif isinstance(file.theorems, list):
            file.theorems = list(set(file.theorems).intersection(lemmas_to_prove))
        else:
            raise ValueError(f"Invalid theorems: {file.theorems}")
        logger.info(f"Discovered {len(file.theorems)} lemmas to prove in {path}")
        logger.info(f"Lemmas to prove in file {path}: \n{file.theorems}")
        if eval_settings.sample < 1.0:
            sample_size = math.ceil(len(file.theorems) * eval_settings.sample)
            logger.info(f"Sampling {sample_size} lemmas from {len(file.theorems)} lemmas in file {path}")
            random.seed(eval_settings.sample_seed)
            file.theorems = list(random.sample(file.theorems, sample_size))
            logger.info(f"Sampled lemmas to prove in file {path}: \n{file.theorems}")
        file.theorems.sort() # sort to ensure reproducibility
        for lemma_name in file.theorems:
            no_proof_res = ProofSearchResult(
                None, 
                False, 
                lemma_name, 
                [], 
                -1, 
                -1, 
                possible_failed_paths=-1, 
                num_of_backtracks=-1, 
                is_timeout=False, 
                is_inference_exhausted=False, 
                longest_success_path=-1,
                additional_info={})
            logger.info(f"Attempting to prove lemma: {lemma_name}")
            search_guidance_policy : Policy = None
            policy_prompter : PolicyPrompter = None

            if eval_settings.policy_name == PolicyName.Dfs:
                policy_prompter = DfsCoqGptPolicyPrompter(
                    main_sys_prompt_path=prompt_settings.main_prompt,
                    example_conv_prompt_path=prompt_settings.conv_prompt,
                    max_tokens_per_action=eval_settings.max_tokens_per_action,
                    gpt_model_name=eval_settings.gpt_model_name,
                    temperature=eval_settings.temperature,
                    max_history_messages=eval_settings.max_history_messages,
                    k=eval_settings.max_theorems_in_prompt,  # k is the number of theorems to consider at each step
                    retrieve_prompt_examples=eval_settings.use_example_retrieval,
                    num_goal_per_prompt=eval_settings.num_goal_per_prompt,
                    training_data_path=eval_benchmark.dfs_data_path_for_retrieval,
                    metadata_filename=eval_benchmark.dfs_metadata_filename_for_retrieval,
                    language=eval_benchmark.language,
                    logger=logger)
                dfs_tree_search = DFSTreeSearch(language=eval_benchmark.language)
                search_guidance_policy = GptGuidedTreeSearchPolicy(
                    eval_settings.checkpoint_dir, 
                    lemma_name, 
                    policy_prompter,
                    dfs_tree_search,
                    checkpoint_on_exit=eval_settings.should_checkpoint,
                    language=eval_benchmark.language)
            elif eval_settings.policy_name == PolicyName.FewShot:
                policy_prompter = FewShotGptPolicyPrompter(
                    main_sys_prompt_path=prompt_settings.main_prompt,
                    example_conv_prompt_path=prompt_settings.conv_prompt,
                    temperature=eval_settings.temperature,
                    max_tokens_per_action=eval_settings.max_tokens_per_action,
                    max_history_messages=eval_settings.max_history_messages,
                    gpt_model_name=eval_settings.gpt_model_name,
                    k=eval_settings.max_theorems_in_prompt,
                    retrieve_prompt_examples=eval_settings.use_example_retrieval,
                    training_data_path=eval_benchmark.few_shot_data_path_for_retrieval,
                    metadata_filename=eval_benchmark.few_shot_metadata_filename_for_retrieval,
                    language=eval_benchmark.language,
                    logger=logger)
                search_guidance_policy = FewShotGptPolicy(
                    eval_settings.checkpoint_dir,
                    lemma_name,
                    policy_prompter,
                    checkpoint_on_exit=eval_settings.should_checkpoint,
                    language=eval_benchmark.language,
                    logger=logger)
            else:
                raise Exception(f"Unknown policy name: {eval_settings.policy_name}")

            if lemma_name not in eval_checkpoint_info.theorem_maps[path]:
                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                def _run_prover(ret_dict):
                    try:
                        with ProofEnv(f"basic_proof_env_{lemma_name}", proof_exec_callback, lemma_name, retrieval_strategy=env_settings.retrieval_strategy, max_proof_depth=eval_settings.max_proof_depth, always_retrieve_thms=eval_settings.always_use_useful_theorem_retrieval, logger=logger) as env:
                            with search_guidance_policy:
                                agent = ProofAgent(f"proof_agent_{lemma_name}", search_guidance_policy, eval_settings.should_checkpoint, proof_dump_file_name, logger=logger)
                                agent.run_episodes_till_stop(
                                    env,
                                    episodes=eval_settings.max_number_of_episodes,
                                    render=eval_settings.render,
                                    stop_policy=check_query_limit_reached(eval_settings.max_steps_per_episode),
                                    policy_info_message=query_limit_info_message(eval_settings.max_steps_per_episode)
                                )
                            proof_res = env.proof_search_res
                            ret_dict["proof_res"] = proof_res
                            ret_dict["attempted_success"] = True
                            ret_dict["service_down"] = False
                    except ServiceDownError:
                        logger.exception(f"ServiceDownError occurred while proving lemma: {lemma_name} in file {path}")
                        ret_dict["attempted_success"] = False
                        ret_dict["service_down"] = True
                    except:
                        logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")
                        ret_dict["attempted_success"] = False
                        ret_dict["service_down"] = False

                should_retry = True
                max_retry = 4 # This retry is only when for some mysterious reason the llama service goes down
                while should_retry and max_retry > 0:
                    # Run the prover with a timeout
                    timeout = min(eval_settings.timeout_in_secs * eval_settings.max_proof_depth * 1.25, 60 * 12) # max 12 minutes
                    logger.info(f"Running the prover agent for lemma: {lemma_name} with timeout: {timeout} seconds")
                    p = multiprocessing.Process(target=_run_prover, args=(return_dict,))
                    p.start()
                    p.join(timeout)
                    if p.is_alive():
                        p.kill()
                        p.join()
                    p.close()
                    if "attempted_success" not in return_dict:
                        logger.info(f"Prover Agent for lemma: {lemma_name} in file {path} got killed as it timed out.")
                        eval_proof_results.add_theorem_to_maps(path, lemma_name, no_proof_res)
                        eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
                        should_retry = False
                    elif not return_dict["attempted_success"]:
                        if not return_dict["service_down"] or eval_settings.gpt_model_name.startswith("gpt") or max_retry <= 1:
                            logger.info(f"Failed to prove lemma: {lemma_name} in file {path}")
                            eval_proof_results.add_theorem_to_maps(path, lemma_name, no_proof_res)
                            eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
                            should_retry = False
                        elif return_dict["service_down"]:
                            # Kill the llama process if it is a llama model
                            should_retry = True
                            logger.info("Killing the llama process")
                            LlamaAccess.class_kill()
                            logger.info("Killed the llama process")
                            logger.info("Restarting the llama process")
                            LlamaAccess.class_init(eval_settings.gpt_model_name, eval_settings.temperature, debug=False, logger=llama_logger)
                            logger.info("Restarted the llama process")                            
                    else:
                        logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
                        eval_proof_results.add_theorem_to_maps(path, lemma_name, return_dict["proof_res"])
                        eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, True)
                        should_retry = False
                    return_dict.clear()
                    max_retry -= 1
            else:
                logger.info(f"Skipping the attempt for proving lemma: {lemma_name} in file {path} as it was already attempted before.")
    if not eval_settings.gpt_model_name.startswith("gpt"):
        # This is a llama model
        LlamaAccess.class_kill()
    pass

def measure_success(benchmark : EvalBenchmark, eval_settings : EvalSettings, eval_proof_results: EvalProofResults, logger: logging.Logger = None):
    success_count = 0
    proofs_dump_file = os.path.join(eval_settings.proof_dump_dir, "benchmark_proof_results.txt")
    proof_dump_file_exists = os.path.exists(proofs_dump_file)
    open_mode = "a" if proof_dump_file_exists else "w"
    with open(proofs_dump_file, open_mode) as f:
        if not proof_dump_file_exists:
            f.write(f"Settings: \n{eval_settings.to_json(indent=4)}\n")
            f.write(f"Benchmark: \n{benchmark.to_json(indent=4)}\n")
        for path, proofs in eval_proof_results.theorem_map.items():
            for lemma_name, proof_res in proofs.items():
                if proof_res.proof_found:
                    success_count += 1
                    logger.info(f"Proof found for lemma: {lemma_name} in file {path}")
                else:
                    logger.info(f"Proof not found for lemma: {lemma_name} in file {path}")
                f.write(f"Lemma: {lemma_name}\n")
                f.write(f"File: {path}\n")
                f.write(f"Proof/Incomplete proof: \n{proof_res}\n")
        total_attempted = sum([len(x) for _, x in eval_proof_results.theorem_map.items()])
        logger.info(f"Success rate: {success_count}/{total_attempted} = {success_count/total_attempted} for benchmark: {benchmark.name}")
        f.write(f"Success rate: {success_count}/{total_attempted} = {success_count/total_attempted} for benchmark: {benchmark.name}\n")

def eval_benchmark(experiment: Experiments, log_dir: str, logger: logging.Logger = None):
    trial_cnt = 1
    env_settings = experiment.env_settings
    eval_settings = experiment.eval_settings
    benchmark = experiment.benchmark
    checkpoint_dir = experiment.eval_settings.checkpoint_dir
    prompt_settings = experiment.prompt_settings
    eval_settings.checkpoint_dir = os.path.join(checkpoint_dir, benchmark.name, eval_settings.name, prompt_settings.name)
    os.makedirs(eval_settings.checkpoint_dir, exist_ok=True)
    # Load the checkpoint file if it exists
    checkpoint_file = os.path.join(eval_settings.checkpoint_dir, "checkpoint_info.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_info: EvalRunCheckpointInfo = EvalRunCheckpointInfo.from_json(f.read())
        eval_settings.proof_dump_dir = checkpoint_info.proof_dump_dir
        checkpoint_info.logging_dirs.append(log_dir)
    else:
        time_now = time.strftime("%Y%m%d-%H%M%S")
        eval_settings.proof_dump_dir = os.path.join(eval_settings.proof_dump_dir, benchmark.name, time_now)
        os.makedirs(eval_settings.proof_dump_dir, exist_ok=True)
        checkpoint_info = EvalRunCheckpointInfo(
            checkpoint_file=checkpoint_file,
            proof_dump_dir=eval_settings.proof_dump_dir, 
            logging_dirs=[log_dir], 
            theorem_maps={})
    eval_proof_file = os.path.join(eval_settings.proof_dump_dir, "proof_results.json")
    if os.path.exists(eval_proof_file):
        with open(eval_proof_file, "r") as f:
            eval_proof_results: EvalProofResults = EvalProofResults.from_json(f.read())
    else:
        eval_proof_results = EvalProofResults(
            path=os.path.join(eval_settings.proof_dump_dir, "proof_results.json"),
            theorem_map={})
    while trial_cnt > 0:
        try:
            logger = logger or logging.getLogger(__name__)
            for dataset in benchmark.datasets:
                eval_dataset(env_settings, benchmark, prompt_settings, dataset, eval_settings, checkpoint_info, eval_proof_results, logger=logger)
            measure_success(benchmark, eval_settings, eval_proof_results, logger=logger)
            trial_cnt = 0
        except:
            trial_cnt -= 1
            logger.exception(f"Exception occurred. Retrying {trial_cnt} more times.")
            time.sleep(10)
    logger.info(f"Finished running experiment: \n{experiment.to_json(indent=4)}")

@hydra.main(config_path="config", config_name="experiments", version_base="1.2")
def main(cfg):
    experiment = parse_config(cfg)
    os.chdir(root_dir)
    log_dir = ".log/evals/benchmark/{}/{}".format(experiment.benchmark.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(__name__, log_path, logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
    eval_benchmark(experiment, log_dir, logger=logger)
    pass

if __name__ == "__main__":
    # RayUtils.init_ray(num_of_cpus=20, object_store_memory_in_gb=50, memory_in_gb=1)
    main()