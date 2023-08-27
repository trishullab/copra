#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import math
import hydra
import os
from datetime import datetime
from omegaconf import DictConfig
from src.baselines.t5_prover.llm_helpers.base_model import BeamSearchParameters, NucleusSamplingParameters
from src.baselines.t5_prover.llm_helpers.data_format_layout import DataFormatLayoutTypes
from src.baselines.t5_prover.proof_search.common.dataset_prover import ProverConfig, ProverExperiment
from src.baselines.t5_prover.proof_search.tatic_search.provers.local_context_prover import LocalTacticGenerationEngine
from src.tools.dynamic_proof_exec import DynamicProofExecutor

global_store = {}

def get_data_format_type(data_format_type: str):
    if data_format_type.startswith('DataFormatLayoutTypes'):
        data_format_type = data_format_type.split('.')[-1]
    return DataFormatLayoutTypes[data_format_type]


def get_decoding_strategy(decoding_strategy_name: str, cfg: DictConfig):
    if decoding_strategy_name == 'BeamSearchParameters':
        return BeamSearchParameters(
            num_samples=cfg.model.decoding.num_samples,
            generation_max_length=cfg.model.decoding.generation_max_length,
            padding=cfg.model.decoding.padding,
            beam_width=cfg.model.decoding.beam_width
        )
    elif decoding_strategy_name == 'NucleusSamplingParameters':
        return NucleusSamplingParameters(
            num_samples=cfg.model.decoding.num_samples,
            generation_max_length=cfg.model.decoding.generation_max_length,
            padding=cfg.model.decoding.padding,
            top_k=cfg.model.decoding.top_k,
            top_p=cfg.model.decoding.top_p,
            temperature=cfg.model.decoding.temperature
        )
    else:
        raise ValueError(f"Unknown decoding strategy {decoding_strategy_name}")

def get_tactic_engine(tactic_engine_name: str, cfg: DictConfig):
    global global_store
    if tactic_engine_name == 'LocalTacticGenerationEngine':
        return LocalTacticGenerationEngine(
            model_name_or_path=cfg.model.tactic_engine.model_name_or_path,
            max_model_length=cfg.model.tactic_engine.max_model_length,
            data_format_layout=get_data_format_type(cfg.model.tactic_engine.data_format_layout),
            decoding_strategy=get_decoding_strategy(cfg.model.decoding.name, cfg),
            layout_split=[(length, end_trimmed) for length, end_trimmed in zip(cfg.model.tactic_engine.layout_split1, cfg.model.tactic_engine.layout_split2)]
        )
    else:
        raise ValueError(f"Unknown tactic engine {tactic_engine_name}")

def get_context_type(context_type: str):
    if context_type.startswith('DynamicProofExecutor.ContextType'):
        context_type = context_type.split('.')[-1]
    return DynamicProofExecutor.ContextType[context_type]

def set_cuda_visible_devices(cuda_visible_devices: list):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cuda_visible_devices])

@hydra.main(config_path="config", config_name="experiments", version_base="1.2")
def main(cfg):
    if cfg.cuda_visible_devices:
        set_cuda_visible_devices(cfg.cuda_visible_devices)
    from torch.multiprocessing import Pool, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    _experiment_type = ProverExperiment
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging_folder = os.path.join(cfg.model.output_settings.logging_folder_prefix, cfg.model.output_settings.name_suffix, time)
    dumping_folder = os.path.join(cfg.model.output_settings.dumping_folder_prefix, cfg.model.output_settings.name_suffix, time)
    name = os.path.join(cfg.model.output_settings.name_prefix, cfg.model.output_settings.name_suffix, time)
    os.makedirs(logging_folder, exist_ok=True)
    os.makedirs(dumping_folder, exist_ok=True)
    experiments =[
        _experiment_type(
            enabled=cfg.model.enabled,
            context_type=get_context_type(cfg.model.context_type),
            prover_config=ProverConfig(
                k=cfg.model.prover_config.k,
                proof_depth=cfg.model.prover_config.proof_depth,
                proof_timeout_in_secs= math.inf if cfg.model.prover_config.proof_timeout_in_secs is None or cfg.model.prover_config.proof_timeout_in_secs == 'inf' else cfg.model.prover_config.proof_timeout_in_secs,
                max_inferences_allowed=cfg.model.prover_config.max_inferences_allowed,
                config_file_path=cfg.model.prover_config.config_file_path,
                name=name,
                dumping_folder=dumping_folder,
                logging_folder=logging_folder,
                only_test=cfg.model.prover_config.only_test,
                seed=cfg.model.prover_config.seed,
                disable_backtracking=cfg.model.prover_config.disable_backtracking
            ),
            tactic_engine=get_tactic_engine(cfg.model.tactic_engine.name, cfg)
        )]
    for experiment in experiments:
        experiment.run()
    pass

if __name__ == "__main__":
    main()
