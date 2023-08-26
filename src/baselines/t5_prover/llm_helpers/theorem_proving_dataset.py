#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import logging
import random
import typing
from torch.utils.data import DataLoader, Dataset
from src.baselines.t5_prover.llm_helpers.data_format_layout import DataFormatLayoutTypes, DataLayout
from src.tools.training_data import TrainingData
from src.tools.training_data_format import TrainingDataFormat
from src.baselines.t5_prover.llm_helpers.trainable_model import TrainingDataArguments, PaddingPolicy


class TheoremProvingDataset(Dataset):
    def __init__(self, 
        data_folder: str, 
        meta_filename: str, 
        formatter: typing.Callable[[TrainingDataFormat, typing.List[typing.Tuple[int, bool]]], typing.Union[typing.Tuple[str, str], str]], 
        data_args: TrainingDataArguments, 
        logger: logging.Logger = None, 
        sample_percent: float = 1.0, 
        shuffle_seed: int = None, 
        data_sampler: typing.Callable[[Dataset, typing.Tuple[TrainingDataFormat]], bool] = None):
        assert data_folder is not None, "Data folder cannot be None"
        assert os.path.exists(data_folder), f"Data folder {data_folder} does not exist"
        self.data_folder = data_folder
        self.logger = logger if logger is not None else logging.getLogger()
        self.training_data : TrainingData = TrainingData(data_folder, meta_filename, logger=self.logger)
        self.training_data.load_meta()
        self.formatter = formatter
        self.data_args : TrainingDataArguments = data_args
        self.sample_percentage = sample_percent
        self.ignored_inputs = 0
        self.data_sampler = data_sampler
        self.seed = shuffle_seed
        self.shuffle_idx = None
        if self.data_args.shuffle:
            self.shuffle_idx = list(range(len(self)))
            if shuffle_seed is not None:
                current_random_state = random.getstate()
                random.seed(shuffle_seed)
            random.shuffle(self.shuffle_idx)
            if shuffle_seed is not None:
                random.setstate(current_random_state)

    def __enter__(self):
        self.training_data.load()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        del self.training_data
    
    def __len__(self):
        assert self.training_data._meta_loaded, "Cannot get length of dataset before loading!"
        return len(self.training_data)
    
    def __str__(self) -> str:
        return self.data_folder
        
    def __getitem__(self, index):
        assert self.training_data._is_loaded, "Cannot get item before loading!"
        assert index >= 0 and index < len(self), f"Index {index} out of range [0, {len(self)})"
        if self.data_args.shuffle:
            shuffled_index = self.shuffle_idx[index]
        else:
            shuffled_index = index
        return self.formatter(self.training_data[shuffled_index])

class CustomBatch(object):
    def __init__(self, x, y, should_tokenize_x, should_tokenize_y, max_length_x, max_length_y, padding, truncation):
        self.x = x
        self.y = y
        self.should_tokenize_x = should_tokenize_x
        self.should_tokenize_y = should_tokenize_y
        self.max_length_x = max_length_x
        self.max_length_y = max_length_y
        self.padding = padding
        self.truncation = truncation
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return CustomBatch(self.x[idx], self.y[idx], self.should_tokenize_x, self.should_tokenize_y, self.max_length_x, self.max_length_y, self.padding, self.truncation)

class DatasetFactory(object):
    def __init__(self,
        data_format_layout: DataFormatLayoutTypes = DataFormatLayoutTypes.Start_Local_MProof__GPTStyle,
        with_label: bool = True, 
        dataset_class = TheoremProvingDataset, 
        data_sampler: typing.Any = None):
        data_layout = DataLayout(data_format_layout, with_label)
        self.training_data_formatter = data_layout.get_layout_formatter()
        self.dataset_class = dataset_class
        self.data_sampler = data_sampler
        pass

    def get_dataset(self, 
        data_folder: str, 
        meta_filename: str, 
        data_args: TrainingDataArguments, 
        logger: logging.Logger = None, 
        sample_percent: float = 1.0, 
        shuffle_seed: int = None, 
        data_sampler: typing.Callable[[Dataset, typing.Tuple[TrainingDataFormat]], bool] = None) -> Dataset:
        assert data_folder is not None, "Data folder cannot be None"
        assert os.path.exists(data_folder), f"Data folder {data_folder} does not exist"
        assert isinstance(sample_percent, float) and sample_percent > 0.0 and sample_percent <= 1.0, "Sample percent must be a float between 0 and 1"        
        assert shuffle_seed is None or isinstance(shuffle_seed, int), "Seed must be greater than or equal to 0"
        self.logger = logger if logger is not None else logging.getLogger()
        dataset = TheoremProvingDataset(
            data_folder=data_folder,
            meta_filename=meta_filename,
            formatter=self.training_data_formatter,
            data_args=data_args,
            logger=self.logger,
            sample_percent=sample_percent,
            shuffle_seed=shuffle_seed,
            data_sampler=data_sampler
        )
        return dataset

class TheoremProvingDataLoaderFactory:
    def __init__(self,  
            tokenizer: typing.Callable[[typing.Union[str, typing.List[str]], bool, bool, int], typing.Dict[str, typing.Any]], 
            data_args: TrainingDataArguments, 
            logger: logging.Logger = None, 
            return_custom_batch: bool = False):
        self.data_args = data_args
        self.logger = logger if logger is not None else logging.getLogger()
        self.return_custom_batch = return_custom_batch
        self.tokenizer = tokenizer

    def tokenize(self, texts: typing.Union[str, typing.List[str]], truncation: bool = True, padding: bool = False, max_length: int = None) -> typing.Dict[str, typing.Any]:
        return self.tokenizer(texts, truncation, padding, max_length)

    def collate_fn(self, batch):
        """
        Collate function for the data loader
        batch: list of tuples (data_x, data_y)"""
        assert batch is not None, "Batch cannot be None"
        assert len(batch) > 0, "Batch cannot be empty"
        # 1. First find the maximum length of the data_x
        x = [_x for _x, _ in batch]
        y = [_y for _, _y in batch]
        if self.data_args.padding_policy == PaddingPolicy.MAX_LENGTH:
            max_len_x = self.data_args.max_len_x
            max_len_y = self.data_args.max_len_y
        else:
            if isinstance(x[0], str):
                max_len_x = max([len(_) for _ in x])
            if isinstance(y[0], str):
                max_len_y = max([len(_) for _ in y])

        tokenized_x = x
        if isinstance(x[0], str) and not self.return_custom_batch:
            # 2. Tokenize x
            tokenized_x = self.tokenize(x, 
                                        truncation=self.data_args.truncation, 
                                        padding=self.data_args.padding,
                                        max_length=max_len_x)
        tokenized_y = y
        if isinstance(y[0], str) and not self.return_custom_batch:
            # 3. Tokenize y
            tokenized_y = self.tokenize(y,
                                        truncation=self.data_args.truncation,
                                        padding=self.data_args.padding,
                                        max_length=max_len_y)
        if not self.return_custom_batch:
            return tokenized_x, tokenized_y
        else:
            return CustomBatch(x, y, isinstance(x[0], str), isinstance(y[0], str), max_len_x, max_len_y, self.data_args.padding, self.data_args.truncation)

    def get_data_loader(self, dataset: Dataset, batch_size: int = 5):
        assert dataset is not None, "Dataset cannot be None"
        assert batch_size is not None, "Batch size cannot be None"
        assert batch_size > 0, "Batch size must be greater than 0"
        return DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn)

if __name__ == "__main__":
    from datetime import datetime
    log_dir = ".log/theorem_proving_dataset/test/custom_group_theory/train"
    data_folder = ".log/run_data_generation_transforms/data/test/custom_group_theory/train"
    trainable_model_output_dir = ".log/models/test/codet5/"
    trainable_model_log_dir = ".log/models/test/logs/codet5"
    assert os.path.exists(data_folder), f"Data folder {data_folder} does not exist"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trainable_model_output_dir, exist_ok=True)
    os.makedirs(trainable_model_log_dir, exist_ok=True)
    time=datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"test-{time}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Loading training data")
    random.seed(0xf00)
    from transformers import TrainingArguments
    from src.baselines.t5_prover.llm_helpers.code_t5_trainable import CodeT5Trainable
    data_layout = DataLayout(DataFormatLayoutTypes.Start_Local_MProof__GPTStyle, with_label=True)
    training_data_formatter = data_layout.get_layout_formatter()
    max_length = 1024
    training_data_args = TrainingDataArguments(
        padding = True, 
        truncation = True, 
        max_length = max_length,
        max_len_x = int(max_length * 0.75),
        max_len_y = int(max_length * 0.25),
        padding_policy=PaddingPolicy.MAX_BATCH_LENGTH, 
        ignore_input_longer_than_block_size=True, 
        shuffle=True)
    training_data_parser = data_layout.get_format_parser()
    sample_percent = 1.0
    training_args = TrainingArguments(
        output_dir=trainable_model_output_dir, 
        do_train=True, 
        do_eval=True, 
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=2,
        seed=0xf00, 
        overwrite_output_dir=True, 
        logging_dir=trainable_model_log_dir,
        num_train_epochs=20,
        eval_steps=10,
        save_steps=10,
        logging_steps=10,
        save_total_limit=3,
        evaluation_strategy='steps', 
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_steps=5,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        load_best_model_at_end=True,
        metric_for_best_model='eval_perplexity',
        greater_is_better=False)
    codet5_trainable = CodeT5Trainable(
        training_data_formatter,
        training_data_parser,
        training_args,
        training_data_args,
        block_size=max_length)
    tokenizer = codet5_trainable.tokenize
    with TheoremProvingDataset(
        data_folder=data_folder, 
        meta_filename="local.meta.json",
        formatter=training_data_formatter, 
        data_args=training_data_args,
        logger=logger, 
        sample_percent=sample_percent, 
        shuffle_seed=0xf00) as dataset:
        logger.info(f"Loaded {len(dataset)} training examples")
        logger.info("Loading data loader")
        data_loader_factory = TheoremProvingDataLoaderFactory(
            tokenizer, 
            training_data_args,
            logger=logger)
        data_loader = data_loader_factory.get_data_loader(dataset=dataset, batch_size=5)
        logger.info("Loading data loader")
        num_batches = 0
        number_of_data_points = 0
        for batch in data_loader:
            x, y = batch
            if num_batches < 50:
                logger.info(f"x['text'][0] = \n{x['text'][0]}")
                logger.info(f"Batch input_ids: x-shape = {x['input_ids'].shape}, y-shape = {y['input_ids'].shape}")
                logger.info(f"Batch attention: x-shape = {x['attention_mask'].shape}, y-shape = {y['attention_mask'].shape}")
            num_batches += 1
            number_of_data_points += x['input_ids'].shape[0]
        len_dataset = int(len(dataset) * sample_percent)
        # excepted_numbatches = (len_dataset - dataset.ignored_inputs) // 5
        assert len_dataset == number_of_data_points, f"Number of batches {num_batches} is not equal to expected number of batches {excepted_numbatches}"
        print(f"Ignored {dataset.ignored_inputs} inputs")