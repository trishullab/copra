#!/usr/bin/env python3

import sys

from dataclasses_json import dataclass_json

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from dataclasses import dataclass
from enum import Enum

class SettingType(Enum):
    Agent = "Agent"
    GptF = "GptF"

    def __str__(self):
        return self.value

class PolicyName(Enum):
    # WARN: Don't make enums dataclasses because deserialization has some weird bug which matches the deserialized enum to all the enum values
    Dfs = "Dfs"
    FewShot = "FewShot"

    def __str__(self):
        return self.value

@dataclass_json
@dataclass
class EvalSettings(object):
    # project_folder: str
    # file_path: str
    use_hammer: bool
    setting_type: SettingType = SettingType.Agent
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
    policy_name: PolicyName = PolicyName.Dfs
    proof_dump_dir: str = ".log/proofs/proof-dump-"
    use_human_readable_proof_context: bool = True

@dataclass_json
@dataclass
class EvalFile(object):
    path: str
    theorems: typing.Union[str, typing.List[str]]

@dataclass_json
@dataclass
class EvalDataset(object):
    project: str
    files: typing.List[EvalFile]

@dataclass_json
@dataclass
class EvalBenchmark(object):
    name: str
    num_files: int
    datasets: typing.List[EvalDataset]

@dataclass_json
@dataclass
class Experiments(object):
    eval_settings: EvalSettings
    benchmark: EvalBenchmark

def parse_config(cfg):
    eval_settings_cfg = cfg["eval_settings"]
    eval_settings = EvalSettings(
        use_hammer=eval_settings_cfg["use_hammer"],
        setting_type=SettingType(eval_settings_cfg["setting_type"]),
        max_proof_depth=eval_settings_cfg["max_proof_depth"],
        timeout_in_secs=eval_settings_cfg["timeout_in_secs"],
        proof_retries=eval_settings_cfg["proof_retries"],
        main_prompt=eval_settings_cfg["main_prompt"],
        conv_prompt=eval_settings_cfg["conv_prompt"],
        max_tokens_per_action=eval_settings_cfg["max_tokens_per_action"],
        max_theorems_in_prompt=eval_settings_cfg["max_theorems_in_prompt"],
        gpt_model_name=eval_settings_cfg["gpt_model_name"],
        max_number_of_episodes=eval_settings_cfg["max_number_of_episodes"],
        max_steps_per_episode=eval_settings_cfg["max_steps_per_episode"],
        render=eval_settings_cfg["render"],
        checkpoint_dir=eval_settings_cfg["checkpoint_dir"],
        should_checkpoint=eval_settings_cfg["should_checkpoint"],
        temperature=eval_settings_cfg["temperature"],
        max_history_messages=eval_settings_cfg["max_history_messages"],
        policy_name=PolicyName(eval_settings_cfg["policy_name"]),
        proof_dump_dir=eval_settings_cfg["proof_dump_dir"],
        use_human_readable_proof_context=eval_settings_cfg["use_human_readable_proof_context"])
    benchmark_cfg = cfg["benchmark"]
    datasets_cfg = benchmark_cfg["datasets"]
    eval_datasets = []
    for dataset_cfg in datasets_cfg:
        files_cfg = list(dataset_cfg["files"])
        eval_files = []
        for file_cfg in files_cfg:
            theorems = None
            if type(file_cfg["theorems"]) == str:
                theorems = file_cfg["theorems"]
            else:
                theorems = list(file_cfg["theorems"])
            eval_files.append(EvalFile(
                path=file_cfg["path"],
                theorems=theorems))
        eval_datasets.append(EvalDataset(
            project=dataset_cfg["project"],
            files=eval_files))
    benchmark = EvalBenchmark(
        name=benchmark_cfg["name"],
        num_files=benchmark_cfg["num_files"],
        datasets=eval_datasets)
    return Experiments(eval_settings=eval_settings, benchmark=benchmark)