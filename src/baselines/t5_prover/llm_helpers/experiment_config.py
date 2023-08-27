import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import typing
import logging
from datetime import datetime
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from src.baselines.t5_prover.llm_helpers.data_format_layout import DataLayout, DataFormatLayoutTypes
from src.baselines.t5_prover.llm_helpers.trainable_model import TrainingDataArguments, TheoremProvingTrainableModelWrapper
from src.baselines.t5_prover.llm_helpers.comet_helper import CometHelper
from src.baselines.t5_prover.llm_helpers.theorem_proving_trainer import TheoremProvingTrainer
from src.baselines.t5_prover.llm_helpers.code_t5_trainable import CodeT5Trainable
from src.baselines.t5_prover.llm_helpers.theorem_proving_dataset import DatasetFactory
from transformers import TrainingArguments

@dataclass_json
@dataclass
class ExperimentConfig(object):
    experiment_name: str
    data_format_type: DataFormatLayoutTypes
    training_args: TrainingArguments
    training_data_args: TrainingDataArguments
    training_folder_name: str
    commet_secret_name: str = "test_experiments.json"
    test_folder_name: typing.Optional[str] = None
    last_checkpoint_number: typing.Optional[int] = None
    top_k: int = 10
    num_gen_examples_dumps: int = 10
    max_sub_part_lens: typing.List[typing.Tuple[int, bool]] = field(default_factory=list)
    test_sample_percentage: float = 1.0
    max_len_x: int = 256
    max_len_y: int = 256
    use_data_parallel: bool = False,
    tokenize_during_loss_computation: bool = False
    train_percentage: float = 1.0
    model_name: typing.Optional[str] = None
    model_param_top_k: int = 20
    model_param_top_p: float = 0.95
    model_param_temperature: float = 1.0
    cuda_visible_devices: typing.List[int] = field(default_factory=list)

    @property
    def block_size(self):
        return self.max_len_x + self.max_len_y

class TheoremProvingExperiment(object):
    def __init__(self, experiment_config: ExperimentConfig, trainable_model_factory: typing.Callable[[ExperimentConfig], TheoremProvingTrainableModelWrapper], enabled: bool = True, dataset_factory: DatasetFactory = None, test_dataset_factory: DatasetFactory = None):
        assert experiment_config is not None, "Experiment config cannot be None"
        assert isinstance(experiment_config, ExperimentConfig), "Experiment config must be an ExperimentConfig"
        assert trainable_model_factory is not None, "Trainable model cannot be None"
        assert callable(trainable_model_factory), "Trainable model must be a callable"
        assert isinstance(enabled, bool), "Enabled must be a boolean"
        self.experiment_config = experiment_config
        self.trainable_model_factory = trainable_model_factory
        self.enabled = enabled
        self.dataset_factory = dataset_factory if dataset_factory is not None else DatasetFactory(
            experiment_config.data_format_type)
        self.test_dataset_factory = test_dataset_factory if test_dataset_factory is not None else DatasetFactory(
            experiment_config.data_format_type)
    
    def run(self):
        if not self.enabled:
            print(f"Experiment {self.experiment_config.experiment_name} is disabled")
            return
        experiment_name = self.experiment_config.experiment_name
        training_folder =  self.experiment_config.training_folder_name
        test_folder = self.experiment_config.test_folder_name
        comet_secret_name = self.experiment_config.commet_secret_name
        last_checkpoint_number = self.experiment_config.last_checkpoint_number
        top_k = self.experiment_config.top_k
        num_gen_examples_dumps = self.experiment_config.num_gen_examples_dumps
        test_sample_percentage = self.experiment_config.test_sample_percentage
        tokenize_during_loss_computation = self.experiment_config.tokenize_during_loss_computation
        use_data_parallel = self.experiment_config.use_data_parallel
        train_percentage = self.experiment_config.train_percentage
        os.makedirs(self.experiment_config.training_args.logging_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"{os.path.join(self.experiment_config.training_args.logging_dir, f'experiment_run_{current_time}.log')}"
        with open(log_file, "w") as f:
            f.write("") # Clear the file
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(experiment_name)
        logger.setLevel(logging.INFO)
        logger.info(f"Process ID: {os.getpid()}")
        logger.info(f"Starting experiment {self.experiment_config.to_json()}")
        try:
            comet_helper = CometHelper(secret_file=comet_secret_name)
            trainable = self.trainable_model_factory(self.experiment_config)
            experiment = comet_helper.get_experiment(experiment_name)
            trainer = TheoremProvingTrainer(
                training_name=experiment_name,
                trainable_model=trainable,
                training_folder=training_folder,
                experiment = experiment,
                dev_folder=test_folder,
                test_folder=test_folder,
                last_checkpoint_number=last_checkpoint_number,
                logger=logger,
                top_k=top_k,
                num_gen_example_dumps=num_gen_examples_dumps,
                test_sample_percentage=test_sample_percentage,
                tokenize_during_loss_computation=tokenize_during_loss_computation,
                use_data_parallel=use_data_parallel,
                train_percentage=train_percentage,
                dataset_factory=self.dataset_factory,
                test_dataset_factory=self.test_dataset_factory)
            with trainer:
                trainer.start_training()
        except Exception as e:
            logger.error(f"Error while running experiment {experiment_name}: {e}")
            logger.exception(e)
            raise e

def code_t5_small_trainable_factory(experiment_config: ExperimentConfig) -> TheoremProvingTrainableModelWrapper:
    data_layout = DataLayout(experiment_config.data_format_type, with_label=True)
    temp_training_data_formatter = data_layout.get_layout_formatter()
    if len(experiment_config.max_sub_part_lens) > 0:
        training_data_formatter = lambda x: temp_training_data_formatter(x, experiment_config.max_sub_part_lens)
    else:
        training_data_formatter = temp_training_data_formatter
    training_data_parser = data_layout.get_format_parser()
    default_model_name = "Salesforce/codet5-small" if experiment_config.model_name is None else experiment_config.model_name
    return CodeT5Trainable(training_data_formatter, training_data_parser, experiment_config.training_args, experiment_config.training_data_args, block_size=experiment_config.block_size, model_name=default_model_name, top_k=experiment_config.model_param_top_k, top_p=experiment_config.model_param_top_p, temperature=experiment_config.model_param_temperature)

def code_t5_base_trainable_factory(experiment_config: ExperimentConfig) -> TheoremProvingTrainableModelWrapper:
    data_layout = DataLayout(experiment_config.data_format_type, with_label=True)
    temp_training_data_formatter = data_layout.get_layout_formatter()
    if len(experiment_config.max_sub_part_lens) > 0:
        training_data_formatter = lambda x: temp_training_data_formatter(x, experiment_config.max_sub_part_lens)
    else:
        training_data_formatter = temp_training_data_formatter
    training_data_parser = data_layout.get_format_parser()
    default_model_name = "Salesforce/codet5-base" if experiment_config.model_name is None else experiment_config.model_name
    return CodeT5Trainable(training_data_formatter, training_data_parser, experiment_config.training_args, experiment_config.training_data_args, block_size=experiment_config.block_size, model_name=default_model_name, top_k=experiment_config.model_param_top_k, top_p=experiment_config.model_param_top_p, temperature=experiment_config.model_param_temperature)

def code_t5_large_trainable_factory(experiment_config: ExperimentConfig) -> TheoremProvingTrainableModelWrapper:
    data_layout = DataLayout(experiment_config.data_format_type, with_label=True)
    temp_training_data_formatter = data_layout.get_layout_formatter()
    if len(experiment_config.max_sub_part_lens) > 0:
        training_data_formatter = lambda x: temp_training_data_formatter(x, experiment_config.max_sub_part_lens)
    else:
        training_data_formatter = temp_training_data_formatter
    training_data_parser = data_layout.get_format_parser()
    default_model_name = "Salesforce/codet5-large" if experiment_config.model_name is None else experiment_config.model_name
    return CodeT5Trainable(training_data_formatter, training_data_parser, experiment_config.training_args, experiment_config.training_data_args, block_size=experiment_config.block_size, model_name=default_model_name, top_k=experiment_config.model_param_top_k, top_p=experiment_config.model_param_top_p, temperature=experiment_config.model_param_temperature)