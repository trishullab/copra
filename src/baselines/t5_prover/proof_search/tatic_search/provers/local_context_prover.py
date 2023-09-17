#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import copy
import os
import logging
import math
from src.tools.training_data_format import TrainingDataFormat, TrainingDataCollection
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor
from src.baselines.t5_prover.llm_helpers.cuda_context import CudaContext
from src.baselines.t5_prover.llm_helpers.code_t5_helper import CodeT5Helper
from src.baselines.t5_prover.llm_helpers.base_model import DecodingParameters, BeamSearchParameters, NucleusSamplingParameters
from src.baselines.t5_prover.llm_helpers.data_format_layout import DataFormatLayoutTypes, DataLayout
from src.baselines.t5_prover.proof_search.common.generic_proof_search_engine import GenericTacticGenerationEngine, Prover, TacticGenerationEngineParams
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LocalTacticGenerationEngineParams(TacticGenerationEngineParams):
    model_name_or_path: str
    max_model_length: str
    data_format_layout: DataFormatLayoutTypes
    decoding_strategy: DecodingParameters
    layout_split: typing.List[typing.Tuple[int, bool]]
    cuda_device_id: int = 0

    def get_tactic_generation_engine(self, logger: logging.Logger = None):
        cuda_context = CudaContext(turn_off_cuda=self.cuda_device_id < 0, device_id=self.cuda_device_id)
        return LocalTacticGenerationEngine(self.model_name_or_path, self.max_model_length, self.data_format_layout, self.decoding_strategy, self.layout_split, cuda_context, logger)

class LocalTacticGenerationEngine(GenericTacticGenerationEngine):
    def __init__(self, 
            model_name_or_path: str, 
            max_model_length: int, 
            data_format_layout: DataFormatLayoutTypes,
            decoding_strategy: DecodingParameters,
            layout_split: typing.List[typing.Tuple[int, bool]] = [], 
            cuda_context: CudaContext = None,
            logger: logging.Logger = None):
        assert model_name_or_path is not None
        assert isinstance(model_name_or_path, str)
        assert cuda_context is None or isinstance(cuda_context, CudaContext)
        assert isinstance(max_model_length, int)
        assert max_model_length > 0
        assert isinstance(data_format_layout, DataFormatLayoutTypes)
        assert isinstance(layout_split, list)
        assert isinstance(decoding_strategy, DecodingParameters)
        self.decoding_strategy = decoding_strategy
        self.data_format_layout = data_format_layout
        self.data_layout = DataLayout(data_format_layout, with_label=False)
        self.generation_parser = self.data_layout.get_format_parser()
        self.input_formatter = self.data_layout.get_layout_formatter()
        self.split = layout_split
        self.model_name = model_name_or_path
        self.max_model_length = max_model_length
        self._cuda_context = cuda_context
        self._is_loaded = False
        self._load_model()
        super().__init__()
        self.logger = logger
    
    def _assert_loaded(self):
        assert self._is_loaded, "Model is not loaded"

    def _load_model(self):
        self._model = CodeT5Helper(self._cuda_context)
        self._model.load_model(self.model_name, self.max_model_length)
        self._is_loaded = True
    
    def get_next_tactic_step(self, partial_data: TrainingDataFormat, k: int = 1) -> typing.List[TrainingDataFormat]:
        self._assert_loaded()
        assert isinstance(partial_data, TrainingDataFormat)
        assert isinstance(k, int)
        assert k > 0
        decoding_params = copy.deepcopy(self.decoding_strategy)
        decoding_params.num_samples = k
        if isinstance(decoding_params, BeamSearchParameters):
            if decoding_params.beam_width < k:
                decoding_params.beam_width = k
        formated_text = self.input_formatter(partial_data, self.split)
        generated_res = self._model.generate(formated_text, decoding_params)
        unique_res = set()
        final_res = []
        # Parse the results
        for i in range(len(generated_res.output)):
            r = generated_res.output[i].strip()
            try:
                res = self.generation_parser(formated_text, r)
            except Exception as e:
                # ignore any parsing errors because LLMs are not perfect
                # self.logger.warning(f"Error parsing the result:\nINPUT: {formated_text}\nOUTPUT: {r}")
                # self.logger.exception(e)
                continue
            proof_steps = res.proof_steps
            if len(proof_steps) == 0:
                # Ignore empty results
                continue
            if not proof_steps[-1].endswith("."):
                # Ignore results which do not end with a dot
                continue
            proof_step_set = '\n'.join([p.strip() for p in proof_steps])
            if proof_step_set in unique_res:
                # Remove duplicates
                continue
            # self.logger.info(f"Generated proof step:\n{proof_step_set}")
            unique_res.add(proof_step_set)
            final_res.append(res)
        return final_res

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging_dir = f".log/local_context_prover/test/{current_time}"
    proof_file = "data/test/SimpleAlgebra.v"
    project_dir = "."
    # proof_file = "data/custom_group_theory/SimpleProofs.v"
    # proof_file = "data/custom_group_theory/Homework1.v"
    # project_dir = None
    k = 20
    depth = 20
    timeout = math.inf
    max_inferences_allowed = 500
    # decoding_params = BeamSearchParameters(k, 350//4, padding=True, beam_width=k+3)
    decoding_params = NucleusSamplingParameters(k, 150//4, padding=True, top_k=k+10, top_p=0.95, temperature=0.8)
    model_name = ".log/generator_experiments/codet5/test/code_t5_small_Local_test_x750_y150_2023-08-27_20-22-11"
    model_max_length = 1050
    format = DataFormatLayoutTypes.Start_Local_MProof__T5Style
    layout_split = []
    os.makedirs(logging_dir, exist_ok=True)
    log_file = os.path.join(logging_dir, f'local_context_prover.log')
    with open(log_file, "w") as f:
        f.write("") # Clear the file
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger("Local Context Prover")
    logger.setLevel(logging.INFO)
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Starting experiment Local Context Prove at K={k}, Depth={depth}, timeout={timeout} secs, decoding params = {decoding_params}, max_inferences_allowed={max_inferences_allowed}")
    tactic_engine = LocalTacticGenerationEngine(model_name, model_max_length, format, decoding_params, layout_split, logger=logger)
    tactic_engine.logger = logging.getLogger("Tactic Engine")
    try:
        prover = Prover("LocalContextProverT5", k, depth, timeout, tactic_engine, DynamicProofExecutor.ContextType.LocalContext, disable_backtracking=False, logger=logger, max_inferences_allowed=max_inferences_allowed)
        proofs = prover.try_proving_theorems_in_file(proof_file, project_dir)
        num_success = len([p for p in proofs if p.proof_found])
        logger.info(f"Proved {num_success} out of {len(proofs)} theorems.")
        logger.info(f"Succes rate: {num_success / len(proofs)}.")
    except Exception as e:
        logger.exception(e)