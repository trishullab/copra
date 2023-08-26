#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import json
import typing
from comet_ml import Experiment

class CometHelper(object):
    def __init__(self, secret_file: str, secret_folder: str = ".secrets"):
        assert secret_file is not None, "Secret file cannot be None"
        secret_file_path = os.path.join(secret_folder, secret_file)
        assert os.path.exists(secret_file_path), f"Secret file {secret_file} does not exist in the folder {secret_folder}"
        secret = None
        with open(secret_file_path, "r") as f:
            secret = f.read().strip()
            secret = json.loads(secret)
        self.project_name = secret["project_name"]
        self.api_key = secret["api_key"]
        self.workspace = secret["workspace"]
    
    def get_experiment(self, experiment_name: str, tags: typing.List[str] = None) -> Experiment:
        assert experiment_name is not None, "Experiment name cannot be None"
        exp = Experiment(project_name=self.project_name,
                         workspace=self.workspace,
                         api_key=self.api_key)
        exp.set_name(experiment_name)
        if tags is not None:
            exp.add_tags(tags)
        return exp
    

if __name__ == "__main__":
    helper = CometHelper(secret_file="test_experiments.json")
    exp = helper.get_experiment(experiment_name="test", tags=["test"])
    exp.log_metric("test", 1)
