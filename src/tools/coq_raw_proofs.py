#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import os
import typing
import ray
from src.tools.ray_utils import RayUtils
from src.tools.coq_parse_utils import CoqLineByLineReader
from src.tools.training_data_format import Goal, TrainingDataFormat, TrainingDataMetadataFormat
from src.tools.training_data import TrainingData
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.tools.coq_build_tool import CoqRepoBuilder

def extract_raw_proof(file_content: str, lemma_name: str) -> TrainingDataFormat:
    # Find the first occurence of the lemma_name and then find the first occurence of "Qed."
    # The proof is the text between the two
    start_index = file_content.find(lemma_name)
    assert start_index != -1, f"Lemma {lemma_name} not found in file"
    end_index = file_content.find("Qed.", start_index)
    assert end_index != -1, f"Lemma {lemma_name} does not have a proof"
    lemma_proof = file_content[start_index:end_index + len("Qed.")]
    line_reader = CoqLineByLineReader(file_content=lemma_proof)
    # Just read the first line
    statement = next(line_reader.instruction_step_generator())
    # Find the next occurence of '.' after the statement which is not in a comment or quotation
    next_dot_index = lemma_proof.find(".", len(statement) - 1) # This may not always work but it is good enough for now
    assert next_dot_index != -1, f"Lemma {lemma_name} does not have a statement"
    proof = lemma_proof[next_dot_index + 1:]
    goal = Goal(goal=statement)
    training_data = TrainingDataFormat(start_goals=[goal], end_goals=[], proof_steps=[proof])
    return training_data

@ray.remote
def get_proofs_in_file(coq_proof_exec_callback: ProofExecutorCallback, file_content: str, file_path: str) -> typing.List[TrainingDataFormat]:
    training_data_collection : typing.List[TrainingDataFormat] = []
    failures = []
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
                try:
                    tdf = extract_raw_proof(file_content, lemma_name)
                    training_data_collection.append(tdf)
                    main_executor.run_to_finish_lemma()
                except Exception:
                    print(f"Failed to extract proof for lemma {lemma_name} in {file_path}")
                    _ = list(main_executor.run_to_finish_lemma_return_exec())
                    failures.append(lemma_name)
                    continue
    return file_path, training_data_collection, failures

def dump_raw_proofs(repo_builder: CoqRepoBuilder, output_dir: str, logger: logging.Logger):
    assert os.path.exists(output_dir), f"output_dir {output_dir} does not exist"
    RayUtils.init_ray(num_of_cpus=20, object_store_memory_in_gb=100, memory_in_gb=1)
    projects = repo_builder.compilable_projects

    def _log_proof_cnt(file_type: str, proof_cnt_in_file: int, file_path: str, failures: typing.List[str]):
        logger.info(f"==============================>[{file_type}] Added proofs {proof_cnt_in_file} in {file_path} <==============================")
        if len(failures) > 0:
            logger.info(f"==============================>[{file_type}] Failed to extract proofs for {len(failures)} lemmas in {file_path} <==============================")
            for lemma_name in failures:
                logger.info(f"==============================>[{file_type}] Failed to extract proof for lemma {lemma_name} in {file_path} <==============================")
    
    def _get_next_files_callback(files):
        _idx = 0
        sorted_files = sorted(files)
        def _get_files(cnt: int):
            nonlocal _idx
            file_input = []
            for file_path in sorted_files[_idx:_idx + cnt]:
                if file_path.startswith(project_path):
                    with open(file_path, 'r') as fd:
                        file_content = fd.read()
                    coq_proof_exec_callback = ProofExecutorCallback(
                        project_folder=project_path,
                        file_path=file_path,
                        use_hammer=False,
                        timeout_in_secs=60,
                        use_human_readable_proof_context=True,
                        suppress_error_log=True,
                        logger=logger)
                    file_input.append((coq_proof_exec_callback, file_content, file_path))
            _idx += cnt
            return file_input
        return _get_files
    
    def _get_file_op_remotes_callback(file_inputs):
        return [get_proofs_in_file.remote(coq_proof_exec_callback, file_content, file_path) for coq_proof_exec_callback, file_content, file_path in file_inputs]
    
    def _transform_outputs_callback(training_data: TrainingData, file_type: str):
        def _transform_outputs(results: typing.List[typing.List[TrainingDataFormat]]):
            for file_path, training_data_collection, failures in results:
                for tdf in training_data_collection:
                    training_data.merge(tdf)
                _log_proof_cnt(file_type, len(training_data_collection), file_path, failures)
        return _transform_outputs

    for project in sorted(projects):
        project_proof_cnt = 0
        # Create temporary directory for each project
        logger.info(f"==============================> Discovering files in project {project}<==============================")
        project_path = os.path.join(repo_builder.root, project)
        assert os.path.exists(project_path), f"project_path {project_path} does not exist"
        files_type_map = {
            "train": repo_builder.train_compilable_files,
            "test": repo_builder.test_compilable_files,
        }
        for file_type, files in files_type_map.items():
            output_path = os.path.join(output_dir, project, file_type)
            os.makedirs(output_path, exist_ok=True)
            training_metadata = TrainingDataMetadataFormat(training_data_buffer_size=2500, data_filename_prefix="single_data_", lemma_ref_filename_prefix="single_lemma_ref_")
            training_data = TrainingData(output_path, "single.meta.json", training_metadata)
            RayUtils.ray_run_within_parallel_limits(
                10,
                len(files),
                _transform_outputs_callback(training_data, file_type),
                _get_next_files_callback(files),
                _get_file_op_remotes_callback,
                logger=logger,
                turn_off_logging=True)
            logger.info(f"[{file_type}] Saving training data to {output_path}")
            save_folder = training_data.save()
            logger.info(f"[{file_type}] Saved training data to {save_folder}")
            project_proof_cnt += len(training_data)
            logger.info(f"==============================>[{file_type}] Added all  {len(training_data)} proofs in project {project}<==============================")
        logger.info(f"==============================> Added {project_proof_cnt} proofs in project {project}<==============================")
    logger.info(f"==============================> Added proofs in all projects<==============================")

def count_proofs(repo_builder: CoqRepoBuilder, logger: logging.Logger):
    proof_cnt = 0
    projects = repo_builder.compilable_projects
    for project in sorted(projects):
        project_proof_cnt = 0
        # Create temporary directory for each project
        logger.info(f"==============================> Discovering files in project {project}<==============================")
        project_path = os.path.join(repo_builder.root, project)
        assert os.path.exists(project_path), f"project_path {project_path} does not exist"
        files_type_map = {
            "train": repo_builder.train_compilable_files,
            "test": repo_builder.test_compilable_files,
        }
        for file_type, files in files_type_map.items():
            file_type_cnt = 0
            for file_path in sorted(files):
                current_proof_cnt = 0
                if file_path.startswith(project_path):
                    with open(file_path, 'r') as fd:
                        file_content = fd.read()
                    current_proof_cnt = file_content.count("Qed.")
                logger.info(f"==============================>[{file_type}] Found {current_proof_cnt} proofs in file {file_path}<==============================")
                proof_cnt += current_proof_cnt
                file_type_cnt += current_proof_cnt
                project_proof_cnt += current_proof_cnt
            logger.info(f"==============================>[{file_type}] Found {file_type_cnt} proofs in project {project}<==============================")
        logger.info(f"==============================> Found {project_proof_cnt} proofs in project {project}<==============================")
    logger.info(f"==============================> Found {proof_cnt} proofs in all projects<==============================")

if __name__ == "__main__":
    import time
    import argparse
    os.chdir(root_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logging_dir = f".log/tools/coq_raw_proofs/{current_time}"
    os.makedirs(logging_dir, exist_ok=True)
    log_file = f"{os.path.join(logging_dir, f'coq_raw_proofs.log')}"
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("ProofCounter")
    logger.info(f"Process ID: {os.getpid()}")
    repo_builder = CoqRepoBuilder(root_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--info_file", type=str, default="data/test/coq/custom_group_theory/compilable_projects_info.log.json", help="The file which will have the info about the compilable projects")
    parser.add_argument("--output_dir", type=str, default=".log/tools/coq_raw_proofs/data/test/coq/custom_group_theory", help="The directory where the raw proofs will be dumped")
    args = parser.parse_args()
    info_file = args.info_file
    repo_builder : CoqRepoBuilder = CoqRepoBuilder.load_from_file(info_file)
    output_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    # count_proofs(repo_builder, logger)
    dump_raw_proofs(repo_builder, output_dir, logger)
