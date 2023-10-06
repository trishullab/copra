#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import os
import torch
import typing
import random
from transformers import set_seed
from datetime import datetime
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor
from src.tools.coq_build_tool import CoqRepoBuilder
from src.baselines.t5_prover.proof_search.common.generic_proof_search_engine import Prover, GenericTacticGenerationEngine, ProofSearchResult

@dataclass_json
@dataclass
class ProverConfig(object):
    config_file_path: str
    name: str
    dumping_folder: str
    logging_folder : str
    k : int = 5
    proof_depth: int = 10
    proof_timeout_in_secs: int = 60
    only_test: bool = True
    seed: int = 0xf00
    disable_backtracking: bool = False
    max_inferences_allowed: typing.Optional[int] = None

class DatasetProver(Prover):
    def __init__(self, config: ProverConfig, tactic_engine: GenericTacticGenerationEngine, context_type: DynamicProofExecutor.ContextType = DynamicProofExecutor.ContextType.NoContext, logger: logging.Logger = None, max_inferences_allowed: typing.Optional[int] = None):
        assert isinstance(config, ProverConfig)
        assert isinstance(tactic_engine, GenericTacticGenerationEngine)
        self.config = config
        assert os.path.exists(config.config_file_path)
        super().__init__(config.name, config.k, config.proof_depth, config.proof_timeout_in_secs, tactic_engine, context_type, config.disable_backtracking, logger, max_inferences_allowed)

    def run_prover(self):
        # Create a CoqRepoBuilder
        coq_repo_builder : CoqRepoBuilder = CoqRepoBuilder.load_from_file(self.config.config_file_path)
        coq_repo_builder.set_logger(self.logger)
        projects = coq_repo_builder.compilable_projects
        if self.config.only_test:
            files = coq_repo_builder.test_compilable_files
        else:
            files = coq_repo_builder.test_compilable_files + coq_repo_builder.train_compilable_files
        os.makedirs(self.config.dumping_folder, exist_ok=True)
        # Run the prover on each project over each file
        for project in sorted(projects):
            project_path = os.path.join(coq_repo_builder.root, project)
            assert os.path.exists(project_path), f"project_path {project_path} does not exist"
            some_files_processed = False
            proj_dump_folder = os.path.join(self.config.dumping_folder, project)
            os.makedirs(proj_dump_folder, exist_ok=True)
            for file_path in sorted(files):
                if file_path.startswith(project_path):
                    some_files_processed = True
                    rel_file_path = os.path.relpath(file_path, project_path)
                    rel_file_path = rel_file_path.replace("/", ".").replace(".v", "")
                    proofs = self.try_proving_theorems_in_file(file_path, project_path)
                    # dump the proofs
                    dump_file_path = os.path.join(proj_dump_folder, f"{rel_file_path}.json")
                    proofs_ser = ProofSearchResult.schema().dumps(proofs, many=True)
                    with open(dump_file_path, "w") as f:
                        f.write(proofs_ser)

        if not some_files_processed:
            self.logger.info(f"==============================>No files processed for project {project}<==============================")
        else:
            self.logger.info(f"==============================>Finished proving over project {project}<==============================")

class ProverExperiment(object):
    def __init__(self, prover_config: ProverConfig, tactic_engine: GenericTacticGenerationEngine, enabled: bool = True, context_type: DynamicProofExecutor.ContextType = DynamicProofExecutor.ContextType.NoContext):
        assert isinstance(prover_config, ProverConfig)
        assert isinstance(tactic_engine, GenericTacticGenerationEngine)
        self.prover_config = prover_config
        self.tactic_engine = tactic_engine
        self.enabled = enabled
        self.context_type = context_type

    def run(self):
        if not self.enabled:
            print(f"Experiment {self.prover_config.name} is disabled")
            return
        torch.manual_seed(self.prover_config.seed)
        random.seed(self.prover_config.seed)
        set_seed(self.prover_config.seed)
        assert self.prover_config.logging_folder is not None
        experiment_name = self.prover_config.name
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dumping_folder = self.prover_config.dumping_folder + f"_{current_time}"
        logging_folder = self.prover_config.logging_folder
        max_inferences_allowed = self.prover_config.max_inferences_allowed
        self.prover_config.dumping_folder = dumping_folder
        os.makedirs(logging_folder, exist_ok=True)
        os.makedirs(dumping_folder, exist_ok=True)
        log_file = f"{os.path.join(logging_folder, f'experiment_run_{current_time}.log')}"
        with open(log_file, "w") as f:
            f.write("") # Clear the file
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(experiment_name)
        logger.setLevel(logging.INFO)
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(f"Starting experiment {experiment_name}: {self.prover_config.to_json()}")
        try:
            self.tactic_engine.set_logger(logging.getLogger(f"{experiment_name}_tactic_engine"))
            dataset_prover = DatasetProver(self.prover_config, self.tactic_engine, self.context_type, logger, max_inferences_allowed)
            dataset_prover.run_prover()
        except Exception as e:
            logger.exception(f"Exception while running experiment {experiment_name}")
            raise e
        pass