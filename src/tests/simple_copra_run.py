import os
import time
import logging
import unittest
import json
from unittest import TestCase
from copra.main.config import parse_config
from copra.main.eval_benchmark import eval_benchmark
from itp_interface.tools.log_utils import setup_logger
from hydra import compose, initialize

class TestSimpleCopraRun(TestCase):
    def test_simple_copra_run(self):
        # Parse hydra configs from src/copra/main/config
        # Initialize Hydra and compose the config
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.dirname(parent_dir)
        full_path = os.path.join(root_dir, "src/copra/main/config")
        full_path = os.path.abspath(full_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.relpath(full_path, current_dir)
        with initialize(config_path=relative_path, version_base="1.2"):
            cfg = compose(config_name="experiments.yaml")
        experiment = parse_config(cfg)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = ".log/evals/benchmark/{}/{}".format(experiment.benchmark.name, timestr)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "eval.log")
        logger = setup_logger(__name__, log_path, logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info(f"Pid: {os.getpid()}")
        logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
        eval_benchmark(experiment, log_dir, logger=logger, timestr=timestr)
        proof_results = os.path.join(".log/proofs/eval_driver/dfs/simple_benchmark_lean4", timestr, "proof_results.json")
        self.assertTrue(os.path.exists(proof_results))
        with open(proof_results, "r") as f:
            proof_results = f.read()
        proof_results = json.loads(proof_results)
        theorem_map = proof_results["theorem_map"]
        for file, theorems in theorem_map.items():
            for theorem, result in theorems.items():
                self.assertTrue(result["proof_found"] == True)
                self.assertTrue(len(result["proof_steps"]) > 0)
                print(f"File: {file}, \nTheorem: \n{result['lemma_name']}")
                proof = '\n\t'.join([_['proof_steps'][0] for _ in result["proof_steps"]])
                print(f" by \n\t{proof}")

if __name__ == "__main__":
    unittest.main()