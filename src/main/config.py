#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum
from src.rl.proof_tree import ProofSearchResult
from src.rl.proof_action import ProofAction
from src.rl.simple_proof_env import ProofEnvReRankStrategy

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
class EnvSettings(object):
    name: str
    retrieval_strategy: ProofEnvReRankStrategy

@dataclass_json
@dataclass
class PromptSettings(object):
    name: str
    main_prompt: str
    conv_prompt: str

@dataclass_json
@dataclass
class EvalSettings(object):
    name: str
    use_hammer: bool
    setting_type: SettingType = SettingType.Agent
    max_proof_depth: int = 50
    timeout_in_secs: int = 60
    proof_retries: int = 1
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
    sample: float = 1.0
    sample_seed: int = 0xf00
    use_example_retrieval: bool = False
    always_use_useful_theorem_retrieval: bool = False
    num_goal_per_prompt: typing.Optional[int] = None

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
    language: ProofAction.Language
    datasets: typing.List[EvalDataset]
    few_shot_data_path_for_retrieval: str = None
    few_shot_metadata_filename_for_retrieval: str = None
    dfs_data_path_for_retrieval: str = None
    dfs_metadata_filename_for_retrieval: str = None

@dataclass_json
@dataclass
class Experiments(object):
    env_settings: EnvSettings
    prompt_settings: PromptSettings
    eval_settings: EvalSettings
    benchmark: EvalBenchmark

@dataclass_json
@dataclass
class EvalRunCheckpointInfo(object):
    checkpoint_file: str
    logging_dirs: typing.List[str]
    proof_dump_dir: str
    theorem_maps: typing.Dict[str, typing.Dict[str, bool]]

    def add_path_to_maps(self, path: str):
        if path not in self.theorem_maps:
            self.theorem_maps[path] = {}

    def add_theorem_to_maps(self, path: str, theorem: str, success: bool):
        self.theorem_maps[path][theorem] = success
        with open(self.checkpoint_file, "w") as f:
            f.write(self.to_json(indent=4))
    
@dataclass_json
@dataclass
class EvalProofResults(object):
    path: str
    theorem_map: typing.Dict[str, typing.Dict[str, ProofSearchResult]]

    def add_path_to_maps(self, path: str):
        if path not in self.theorem_map:
            self.theorem_map[path] = {}
    
    def add_theorem_to_maps(self, path: str, theorem: str, proof_result: ProofSearchResult):
        self.theorem_map[path][theorem] = proof_result
        with open(self.path, "w") as f:
            f.write(self.to_json(indent=4))


def parse_config(cfg):
    env_settings_cfg = cfg["env_settings"]
    env_settings = EnvSettings(
        name=env_settings_cfg["name"],
        retrieval_strategy=ProofEnvReRankStrategy(env_settings_cfg["retrieval_strategy"]))
    prompt_settings_cfg = cfg["prompt_settings"]
    prompt_settings = PromptSettings(
        name=prompt_settings_cfg["name"],
        main_prompt=prompt_settings_cfg["main_prompt"],
        conv_prompt=prompt_settings_cfg["conv_prompt"])
    eval_settings_cfg = cfg["eval_settings"]
    eval_settings = EvalSettings(
        name=eval_settings_cfg["name"],
        use_hammer=eval_settings_cfg["use_hammer"],
        setting_type=SettingType(eval_settings_cfg["setting_type"]),
        max_proof_depth=eval_settings_cfg["max_proof_depth"],
        timeout_in_secs=eval_settings_cfg["timeout_in_secs"],
        proof_retries=eval_settings_cfg["proof_retries"],
        # main_prompt=eval_settings_cfg["main_prompt"],
        # conv_prompt=eval_settings_cfg["conv_prompt"],
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
        use_human_readable_proof_context=eval_settings_cfg["use_human_readable_proof_context"],
        sample=eval_settings_cfg["sample"],
        sample_seed=eval_settings_cfg["sample_seed"],
        use_example_retrieval=eval_settings_cfg["use_example_retrieval"],
        always_use_useful_theorem_retrieval=eval_settings_cfg["always_use_useful_theorem_retrieval"],
        num_goal_per_prompt=eval_settings_cfg["num_goal_per_prompt"])
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
    language = ProofAction.Language(benchmark_cfg["language"])
    benchmark = EvalBenchmark(
        name=benchmark_cfg["name"],
        num_files=benchmark_cfg["num_files"],
        language=language,
        datasets=eval_datasets,
        few_shot_data_path_for_retrieval=benchmark_cfg["few_shot_data_path_for_retrieval"],
        few_shot_metadata_filename_for_retrieval=benchmark_cfg["few_shot_metadata_filename_for_retrieval"],
        dfs_data_path_for_retrieval=benchmark_cfg["dfs_data_path_for_retrieval"],
        dfs_metadata_filename_for_retrieval=benchmark_cfg["dfs_metadata_filename_for_retrieval"])
    return Experiments(env_settings=env_settings, eval_settings=eval_settings, benchmark=benchmark, prompt_settings=prompt_settings)