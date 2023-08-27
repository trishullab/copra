#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import logging
import torch
import typing
import math
import numpy as np
from src.baselines.t5_prover.llm_helpers.trainable_model import TheoremProvingTrainableModelWrapper, ObjectiveTypes
from src.baselines.t5_prover.llm_helpers.theorem_proving_dataset import TheoremProvingDataLoaderFactory, CustomBatch, DatasetFactory
from transformers import Trainer, set_seed, SchedulerType
from torch.utils.data import DataLoader, Dataset
from comet_ml import Experiment


class TheoremProvingTrainer(Trainer):
    max_block_size_limit = 1024
    def __init__(self, 
                 training_name: str, 
                 trainable_model: TheoremProvingTrainableModelWrapper, 
                 training_folder: str,
                 meta_filename: str = "local.meta.json",
                 objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive,
                 dev_folder: str = None,
                 test_folder: str = None, 
                 experiment: Experiment = None, 
                 logger: logging.Logger = None,
                 last_checkpoint_number: typing.Optional[int] = None,
                 top_k: int = 5,
                 num_gen_example_dumps: int = 5,
                 test_sample_percentage: float = 1.0,
                 dev_sample_percentage: float = 1.0,
                 dataset_factory: DatasetFactory = None,
                 use_data_parallel: bool = False,
                 tokenize_during_loss_computation: bool = False,
                 train_percentage: float = 1.0,
                 test_dataset_factory: DatasetFactory = None):
        assert training_name is not None, "Training name cannot be None"
        assert isinstance(training_name, str), "Training name must be a string"
        assert trainable_model is not None, "Model cannot be None"
        assert isinstance(trainable_model, TheoremProvingTrainableModelWrapper), "Model must be a TrainableModel"
        assert training_folder is not None and os.path.exists(training_folder), "Training files must exist"
        assert dev_folder is None or os.path.exists(dev_folder), "Dev files must exist"
        assert test_folder is None or os.path.exists(test_folder), "Test files must exist"
        assert objective_type is not None, "Objective type cannot be None"
        assert isinstance(objective_type, ObjectiveTypes), "Objective type must be an ObjectiveTypes"
        assert isinstance(top_k, int), "Top k must be an integer"
        assert top_k > 0, "Top k must be greater than 0"
        assert isinstance(num_gen_example_dumps, int), "Number of generated example dumps must be an integer"
        assert num_gen_example_dumps >= 0, "Number of generated example dumps must be greater than or equal to 0"
        self.trainable_model = trainable_model
        self.model = trainable_model.get_pretrained_model()
        self.tokenizer = trainable_model.get_tokenizer()
        self.metrics = {}
        self.experiment = experiment
        self.training_folder = training_folder
        self.dev_folder = dev_folder
        self.test_folder = test_folder
        self.training_name = training_name
        self.training_args = self.trainable_model.training_args()
        self.training_data_format_args = self.trainable_model.get_training_data_arguments()
        self.logger = logger if logger is not None else logging.getLogger(training_name)
        self.meta_filename = meta_filename
        block_size = self.trainable_model.get_block_size()
        assert self.training_data_format_args.max_length <= block_size, f"Block size must be greater than or equal to {self.training_data_format_args.max_length}"
        self.top_k = top_k
        self.num_gen_example_dumps = num_gen_example_dumps
        self.use_data_parallelism = use_data_parallel
        self.train_percentage = train_percentage
        self.training_data_loader_factory = TheoremProvingDataLoaderFactory(
            self.trainable_model.tokenize,
            self.training_data_format_args, 
            self.logger, 
            return_custom_batch=tokenize_during_loss_computation)
        self.objective_type = objective_type
        self.test_sample_percentage = test_sample_percentage
        self.dev_sample_percentage = dev_sample_percentage
        self.dataset_factory = dataset_factory if dataset_factory is not None else DatasetFactory()
        self.test_dataset_factory = test_dataset_factory if test_dataset_factory is not None else DatasetFactory()
        if block_size is None:
            self.block_size = self.tokenizer.model_max_length

        super().__init__(
            model=self.model, 
            args=self.training_args,
            tokenizer=self.tokenizer)
        set_seed(self.training_args.seed)
        self.last_checkpoint_number = last_checkpoint_number
        self._is_initialized = False
        self.tokens_seen = 0
    
    def _exit_dataset(self, dataset: Dataset, exc_type = None, exc_value = None, traceback = None):
        if dataset is not None:
            try:
                dataset.__exit__(exc_type, exc_value, traceback)
            except:
                pass

    def _get_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return self.training_data_loader_factory.get_data_loader(dataset, batch_size)

    def _set_dev_dataset(self, dev_dataset: Dataset):
        if self._is_initialized:
            self._exit_dataset(self.dev_dataset)
        self.dev_dataset = dev_dataset
        try:
            self.dev_dataset.__enter__()
        except:
            pass
        self.dev_dataloader = self._get_dataloader(self.dev_dataset, self.training_args.eval_batch_size)

    def _set_train_dataset(self, training_dataset: Dataset):
        if self._is_initialized:
            self._exit_dataset(self.training_dataset)
        self.training_dataset = training_dataset
        try:
            self.training_dataset.__enter__()
        except:
            pass
        self.train_dataloader = self._get_dataloader(self.training_dataset, self.training_args.train_batch_size)

    def _set_test_dataset(self, test_dataset: Dataset):
        if self._is_initialized:
            self._exit_dataset(self.test_dataset)
        self.test_dataset = test_dataset
        try:
            self.test_dataset.__enter__()
        except:
            pass
        self.test_dataloader = self._get_dataloader(self.test_dataset, self.training_args.eval_batch_size)

    def __enter__(self):
        if self.experiment is not None:
            # log parameters to comet.ml
            self.experiment.log_parameters(self.training_args.__dict__)
        if self.training_args.do_train:
            self._set_train_dataset(self.dataset_factory.get_dataset(
                self.training_folder, 
                self.meta_filename, 
                self.training_data_format_args, 
                self.logger, 
                sample_percent=self.train_percentage, 
                shuffle_seed=self.training_args.seed))
            self.logger.info(f"Loaded training dataset from {self.training_folder}, with {len(self.training_dataset)} examples.")
        else:
            self.logger.info("Training disabled, no training dataset loaded.")
        if self.dev_folder is not None and self.training_args.do_train:
            self._set_dev_dataset(self.dataset_factory.get_dataset(
                self.dev_folder,
                self.meta_filename, 
                self.training_data_format_args, 
                self.logger, 
                sample_percent=self.dev_sample_percentage, 
                shuffle_seed=self.training_args.seed))
        elif self.training_args.do_train:
            self._set_dev_dataset(self.dataset_factory.get_dataset(
                self.training_folder, 
                self.meta_filename,
                self.training_data_format_args, 
                self.logger, 
                sample_percent=self.dev_sample_percentage, 
                shuffle_seed=self.training_args.seed))
        if self.training_args.do_train:
            self.logger.info(f"Loaded dev dataset from {self.dev_folder}, with {len(self.dev_dataset)} examples.")
        else:
            self.logger.info("Training disabled, no dev dataset loaded.")
        if self.test_folder is not None:
            self._set_test_dataset(self.test_dataset_factory.get_dataset(
                self.test_folder, 
                self.meta_filename,
                self.training_data_format_args, 
                self.logger, 
                sample_percent=self.test_sample_percentage, 
                shuffle_seed=self.training_args.seed))
        else:
            self._set_test_dataset(self.test_dataset_factory.get_dataset(
                self.training_folder,
                self.meta_filename,
                self.training_data_format_args,
                self.logger, 
                sample_percent=self.test_sample_percentage, 
                shuffle_seed=self.training_args.seed))
        self.logger.info(f"Loaded test dataset from {self.test_folder}, with {len(self.test_dataset)} examples.")
        self._is_initialized = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if not self._is_initialized:
            return
        self._exit_dataset(self.training_dataset, exc_type, exc_value, traceback)
        self._exit_dataset(self.dev_dataset, exc_type, exc_value, traceback)
        self._exit_dataset(self.test_dataset, exc_type, exc_value, traceback)

    def get_train_dataloader(self) -> DataLoader:
        assert self._is_initialized, "The trainer must be initialized before starting training"
        return self.train_dataloader
    
    def get_eval_dataloader(self, eval_dataset: typing.Optional[torch.utils.data.dataset.Dataset] = None) -> DataLoader:
        assert self._is_initialized, "The trainer must be initialized before starting training"
        dev_dataloader = self.dev_dataloader
        if eval_dataset is not None:
            dev_dataloader = self.training_data_loader_factory.get_data_loader(eval_dataset, self.training_args.eval_batch_size)
        return dev_dataloader
    
    def get_test_dataloader(self) -> DataLoader:
        assert self._is_initialized, "The trainer must be initialized before starting training"
        return self.test_dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        assert self._is_initialized, "The trainer must be initialized before starting training"
        if isinstance(inputs, CustomBatch):
            x = inputs.x
            if inputs.should_tokenize_x:
                x = self.trainable_model.tokenize(inputs.x, inputs.truncation, inputs.padding, inputs.max_length_x, model)
            y = inputs.y
            if inputs.should_tokenize_y:
                y = self.trainable_model.tokenize(inputs.y, inputs.truncation, inputs.padding, inputs.max_length_y, model)
        else:
            x, y = inputs
        model = model if self.use_data_parallelism else None
        try:
            output, loss = self.trainable_model.get_output_with_loss(inputs=x, labels=y, objective_type=self.objective_type, model=model)
        except:
            if self.trainable_model.is_generative_model():
                self.logger.error(f"Error while computing loss: {x['input_ids'].shape}, {y['input_ids'].shape}")
            else:
                self.logger.error(f"Error while computing loss: {x['input_ids'].shape}, {len(y)}")
            raise
        if self.experiment:
            loss_sum = np.average(loss.cpu().detach().numpy())
            loss_val = float(loss_sum)
            token_count = self.trainable_model.get_number_of_tokens(x, y)
            self.tokens_seen += token_count
            self.experiment.log_metric("tokens_seen", self.tokens_seen, epoch=self.state.epoch, step=self.state.global_step)
            self.experiment.log_metric("train_batch_loss", loss_val, epoch=self.state.epoch, step=self.state.global_step)
            if self.trainable_model.is_generative_model():
                try:
                    perp = math.exp(loss_val)
                except:
                    perp = float("inf")
                self.experiment.log_metric("train_batch_perplexity", perp, epoch=self.state.epoch, step=self.state.global_step)
        if return_outputs:
            return loss, output
        return loss
    
    def _format_generated_example(self, generated_tuple, idx):
        input: str = generated_tuple.input
        label: str = generated_tuple.label
        generated: typing.List[str] = generated_tuple.generated
        generation_sep = "\n$$$$$$$$$$$$$$$$$$$$\n"
        full_generated = generation_sep.join(generated)
        formatted_log = f"\n<Step: {self.state.global_step}, Example: {idx}> <Input> \n {input} \n <Label> \n {label} \n <Generated> {generation_sep} {full_generated}"
        return formatted_log
    
    def evaluate(self, eval_dataset: typing.Optional[Dataset] = None, ignore_keys: typing.Optional[typing.List[str]] = None, metric_key_prefix: str = "eval") -> typing.Dict[str, float]:
        metrics : typing.Dict[str, float] = {}
        new_eval_dataset_to_use =  eval_dataset is not None
        test_dataset = None
        try:
            if self.trainable_model.is_generative_model():
                good_examples: list = []
                bad_examples: list = []
                test_dataset = self.test_dataset if not new_eval_dataset_to_use else eval_dataset
                self.logger.info(f"Evaluating on {len(test_dataset)} examples from {test_dataset}.")
                test_dataloader = self._get_dataloader(test_dataset, self.training_args.eval_batch_size)
                # Go over a few examples and log the prediction texts
                metrics : typing.Dict[str, float] = {}
                examples_evaluated = 0
                for idx, examples in enumerate(test_dataloader):
                    if isinstance(examples, CustomBatch):
                        x = examples.x
                        if examples.should_tokenize_x:
                            x = self.trainable_model.tokenize(examples.x, examples.truncation, examples.padding, examples.max_length_x, self.trainable_model.model)
                        y = examples.y
                        if examples.should_tokenize_y:
                            y = self.trainable_model.tokenize(examples.y, examples.truncation, examples.padding, examples.max_length_y, self.trainable_model.model)
                    else:
                        x, y = examples
                    # Accumulate metrics over all batches
                    generated_list = self.trainable_model.generate_and_compare(x, y, top_k=self.top_k, max_new_tokens=self.training_data_format_args.max_len_y, objective_type=self.objective_type, metrics=metrics, good_examples=good_examples, bad_examples=bad_examples)
                    self.logger.info(f"Evaluating batch. Current metrics: {metrics}")
                    examples_evaluated += len(generated_list)
                self.logger.info(f"Finished evaluation. Evaluated {examples_evaluated} examples. Metrics: {metrics}")
                # Log the generated examples
                idx = 0
                for generated_tuple in good_examples[:self.num_gen_example_dumps]:
                    formatted_log = self._format_generated_example(generated_tuple, idx)
                    self.logger.info("<Good>\n"+formatted_log)
                    idx += 1
                    if self.experiment:
                        self.experiment.log_text(formatted_log, step=self.state.global_step, metadata={"example_type": "good"})
                idx = 0
                for generated_tuple in bad_examples[:self.num_gen_example_dumps]:
                    formatted_log = self._format_generated_example(generated_tuple, idx)
                    self.logger.info("<Bad>\n"+formatted_log)
                    idx += 1
                    if self.experiment:
                        self.experiment.log_text(formatted_log, step=self.state.global_step, metadata={"example_type": "bad"})
                
                # Add eval prefix to all metrics
                for metric_name, metric_value in list(metrics.items()):
                    metrics[f"{metric_key_prefix}_{metric_name}"] = metric_value
                    del metrics[metric_name]
            else:
                test_dataset = self.test_dataset if not new_eval_dataset_to_use else eval_dataset
                test_dataloader = self._get_dataloader(test_dataset, self.training_args.eval_batch_size)
                self.logger.info(f"Evaluating on {len(test_dataset)} examples from {test_dataset}.")
                other_metrics = self.trainable_model.get_metrics(test_dataloader, metric_key_prefix, self.objective_type)
                for metric_name, metric_value in other_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_value
                    else:
                        metrics[f"{metric_name}_other"] = metric_value
            if self.experiment:
                for metric_name, metric_value in metrics.items():
                    self.experiment.log_metric(metric_name, metric_value, epoch=self.state.epoch, step=self.state.global_step)
                    self.logger.info(f"Logged metric {metric_name} with value {metric_value} at step {self.state.global_step} and epoch {self.state.epoch}")
        finally:
            if new_eval_dataset_to_use and test_dataset is not None:
                self._exit_dataset(test_dataset)
        return metrics
    
    def predict(self, test_dataset: Dataset, ignore_keys: typing.Optional[typing.List[str]] = None, metric_key_prefix: str = "test") -> typing.Any:
        test_dataloader = self._get_dataloader(test_dataset, self.training_args.eval_batch_size)
        named_tuple = self.trainable_model.get_predictions(test_dataloader, metric_key_prefix, self.objective_type)
        return named_tuple
    
    def start_training(self):
        assert self._is_initialized, "The trainer must be initialized before starting training"
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint_number is not None:
                output_dir = self.training_args.output_dir
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{self.last_checkpoint_number}")
                if os.path.exists(checkpoint_dir):
                    checkpoint = checkpoint_dir
            self.logger.info(f"Starting training from checkpoint {checkpoint}")
            train_result = self.train(resume_from_checkpoint=checkpoint)
            self.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics

            self.log_metrics("train", metrics)
            self.save_metrics("train", metrics)
            self.save_state()

        # Evaluation
        if self.training_args.do_eval:
            self.logger.info("*** Evaluate ***")

            metrics = self.evaluate()
            self.log_metrics("eval", metrics)
            self.save_metrics("eval", metrics)

        # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        # if data_args.dataset_name is not None:
        #     kwargs["dataset_tags"] = data_args.dataset_name
        #     if data_args.dataset_config_name is not None:
        #         kwargs["dataset_args"] = data_args.dataset_config_name
        #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        #     else:
        #         kwargs["dataset"] = data_args.dataset_name

        # if training_args.push_to_hub:
        #     trainer.push_to_hub(**kwargs)
        # else:
        #     trainer.create_model_card(**kwargs)
        pass

if __name__ == "__main__":
    from src.baselines.t5_prover.llm_helpers.comet_helper import CometHelper
    from src.baselines.t5_prover.llm_helpers.code_t5_trainable import CodeT5Trainable
    from src.baselines.t5_prover.llm_helpers.trainable_model import PaddingPolicy, TrainingDataArguments
    from src.baselines.t5_prover.llm_helpers.data_format_layout import DataFormatLayoutTypes, DataLayout
    from src.baselines.t5_prover.llm_helpers.theorem_proving_dataset import TheoremProvingDataset
    from src.baselines.t5_prover.llm_helpers.cuda_context import CudaContext
    from transformers import TrainingArguments
    from datetime import datetime
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    cuda_context = CudaContext()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_folder = f".log/run_data_generation_transforms/data/test/custom_group_theory/train"
    test_folder = f".log/run_data_generation_transforms/data/test/custom_group_theory/train"
    model_output_dir = f".log/models/codet5/simple_benchmark/{time}"
    model_log_dir = f".log/modesl/logs/codet5/simple_benchmark/{time}"
    log_folder = f".log/theorem_proving_trainer/test/{time}"
    experiment_name = f"codet5_test_1-{time}"
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(model_log_dir, exist_ok=True)
    log_file = f"{os.path.join(log_folder, f'experiment_run_{time}.log')}"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    comet_helper = CometHelper(secret_file="test_experiments.json")
    data_layout = DataLayout(DataFormatLayoutTypes.Start_Local_MProof__GPTStyle, with_label=True)
    training_data_formatter = data_layout.get_layout_formatter()
    training_data_parser = data_layout.get_format_parser()
    epochs = 41
    per_device_batch_size = 3
    seed = 0xf00
    training_args = TrainingArguments(
        output_dir=model_output_dir, 
        do_train=True, 
        do_eval=True,
        per_device_train_batch_size=per_device_batch_size, 
        per_device_eval_batch_size=per_device_batch_size,
        seed=seed, 
        overwrite_output_dir=True, 
        logging_dir=model_log_dir, 
        num_train_epochs=epochs,
        eval_steps=50,
        save_steps=100,
        logging_steps=50,
        save_total_limit=1,
        evaluation_strategy='steps', 
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_ratio=0.01,
        weight_decay=0.01,
        dataloader_num_workers=10,
        lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
        load_best_model_at_end=True,
        metric_for_best_model='acc_at_2',
        greater_is_better=True)
    data_args = TrainingDataArguments(padding=True, truncation=True, max_len_x=500, max_len_y=50, padding_policy=PaddingPolicy.MAX_LENGTH, ignore_input_longer_than_block_size=True, shuffle=True)
    code_t5_trainable = CodeT5Trainable(training_data_formatter, training_data_parser, training_args, data_args, block_size=900, cuda_context=cuda_context)
    dataset_factory = DatasetFactory(
        data_format_layout=data_layout.data_format_layout,
        with_label=True,
        dataset_class=TheoremProvingDataset
    )
    experiment = comet_helper.get_experiment(experiment_name)
    codet5_trainer = TheoremProvingTrainer(
        training_name=experiment_name,
        trainable_model=code_t5_trainable,
        training_folder=training_folder,
        experiment = experiment,
        test_folder=test_folder,
        dataset_factory=dataset_factory,
        last_checkpoint_number=None,
        top_k=2,
        use_data_parallel=False,
        tokenize_during_loss_computation=False
    )
    with codet5_trainer:
        codet5_trainer.start_training()