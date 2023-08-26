#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import torch
import math
from torch.utils.data import DataLoader
from collections import namedtuple
from src.baselines.t5_prover.llm_helpers.trainable_model import TheoremProvingTrainableModelWrapper, ObjectiveTypes, TrainingDataArguments, PaddingPolicy
from src.baselines.t5_prover.llm_helpers.code_t5_helper import CodeT5Helper
from src.baselines.t5_prover.llm_helpers.cuda_context import CudaContext
from src.tools.training_data_format import TrainingDataFormat
from transformers import TrainingArguments

class CodeT5Trainable(TheoremProvingTrainableModelWrapper):
    def __init__(self, 
                 training_data_formatter: typing.Callable[[TrainingDataFormat], typing.Tuple[str, str]],
                 training_data_parser: typing.Callable[[str, str], TrainingDataFormat],
                 training_arguments: TrainingArguments,
                 data_arguments: TrainingDataArguments,
                 block_size: int = 512, 
                 model_name: str = "Salesforce/codet5-small", 
                 cuda_context: CudaContext = None,
                 top_k: int = 20,
                 top_p: float = 0.95,
                 temperature: float = 1.0):
        assert model_name is not None, "Model name cannot be None"
        assert isinstance(model_name, str), "Model name must be a string"
        assert training_data_formatter is not None, "Training data formatter cannot be None"
        assert callable(training_data_formatter), "Training data formatter must be a function"
        assert training_data_parser is not None, "Training data parser cannot be None"
        assert callable(training_data_parser), "Training data parser must be a function"
        assert training_arguments is not None, "Training arguments cannot be None"
        assert isinstance(training_arguments, TrainingArguments), "Training arguments must be a TrainingArguments"
        assert data_arguments is not None, "Data arguments cannot be None"
        assert isinstance(data_arguments, TrainingDataArguments), "Data arguments must be a TrainingDataArguments"
        assert block_size is not None, "Block size cannot be None"
        assert isinstance(block_size, int), "Block size must be an integer"
        assert block_size > 0, "Block size must be greater than 0"
        assert top_k is not None, "Top k cannot be None"
        assert isinstance(top_k, int), "Top k must be an integer"
        assert top_k > 0, "Top k must be greater than 0"
        assert top_p is not None, "Top p cannot be None"
        assert isinstance(top_p, float), "Top p must be a float"
        assert top_p >= 0, "Top p must be greater than 0"
        assert top_p <= 1, "Top p must be less than or equal to 1"
        self.model_name = model_name
        self.training_data_formatter = training_data_formatter
        self.training_data_parser = training_data_parser
        self.code_t5_helper = CodeT5Helper(cuda_context)
        self.training_arguments = training_arguments
        self.data_arguments = data_arguments
        self.code_t5_helper.load_model(self.model_name, self.data_arguments.max_length)
        self.prediction_named_tuple = namedtuple("Prediction", ["predictions", "label_ids", "metrics"])
        self.generation_named_tuple = namedtuple("Generation", ["input", "label", "generated"])
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        super().__init__(self.code_t5_helper.model, self.code_t5_helper.tokenizer, self.code_t5_helper.model.config, block_size)
        self.special_token = set(self.tokenizer.all_special_tokens)
        for token in self.tokenizer.all_special_tokens:
            if token.startswith("<extra_id"):
                self.special_token.remove(token)
    
    def get_max_length(self):
        return self.data_arguments.max_length

    def get_training_data_formatter(self) -> typing.Callable[[TrainingDataFormat], typing.Tuple[str, str]]:
        return self.training_data_formatter
    
    def get_training_data_parser(self) -> typing.Callable[[str, str], TrainingDataFormat]:
        return self.training_data_parser

    def training_args(self) -> TrainingArguments:
        return self.training_arguments
    
    def get_training_data_arguments(self) -> TrainingDataArguments:
        return self.data_arguments
    
    def is_generative_model(self) -> bool:
        return True

    def tokenize(self, texts: typing.Union[str, typing.List[str]], truncation: bool = True, padding: bool = False, max_length: int = None, model: torch.nn.Module = None) -> typing.Dict[str, typing.Any]:
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        max_length = max_length if max_length is not None else self.get_max_length()
        encoding = self.tokenizer(
            texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt",
        )
        return {'input_ids': self.code_t5_helper.cuda_context.try_get_gpu(encoding.input_ids), 
                'attention_mask': self.code_t5_helper.cuda_context.try_get_gpu(encoding.attention_mask),
                'text': texts}
    
    def get_number_of_tokens(self, inputs: typing.Dict[str, typing.Any], labels: typing.Any) -> int:
        # Compute the number of tokens in the input and the label
        token_count = inputs['attention_mask'].sum().detach().cpu().numpy()
        token_count = int(token_count)
        label_token_count = labels['attention_mask'].sum().detach().cpu().numpy()
        label_token_count = int(label_token_count)
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        return token_count + label_token_count

    def get_output_with_loss(self, inputs: typing.Dict[str, typing.Any], labels: typing.Any, objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive, model: torch.nn.Module = None):
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target = labels['input_ids']
        model = model if model is not None else self.model
        assert len(input_ids.shape) == 2, "Input ids must be a 2D tensor"
        assert len(attention_mask.shape) == 2, "Attention mask must be a 2D tensor"
        assert len(target.shape) == 2, "Target must be a 2D tensor"
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        target[target == self.tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
        return outputs, outputs.loss

    def get_metrics(self, dataloader: DataLoader, metric_key_prefix: str = "train", objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive):
        total_loss = 0.0
        num_batches = 0
        for x, y in dataloader:
            _, loss = self.get_output_with_loss(inputs=x, labels=y, objective_type=objective_type)
            total_loss += float(loss.cpu().detach().numpy())
            num_batches += 1
        avg_loss = total_loss / num_batches
        metrics = {f"{metric_key_prefix}_loss": avg_loss, f"{metric_key_prefix}_perplexity": math.exp(avg_loss)}
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        return metrics
    
    def get_predictions(self, dataloader: DataLoader, metric_key_prefix: str = "test", objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive) -> typing.Any:
        total_loss = 0.0
        num_batches = 0
        outputs = []
        for x, y in dataloader:
            output, loss = self.get_output_with_loss(inputs=x, labels=y, objective_type=objective_type)
            total_loss += float(loss.cpu().detach().numpy())
            outputs.append(output)
            num_batches += 1
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        avg_loss = total_loss / num_batches
        metrics = {f"{metric_key_prefix}_loss": avg_loss, f"{metric_key_prefix}_perplexity": math.exp(avg_loss)}
        logits_collection = [output.logits for output in outputs]
        logits = torch.cat(logits_collection, dim=0)
        logits_np = logits.cpu().detach().numpy()
        named_tuple = self.prediction_named_tuple(logits_np, None, metrics)
        return named_tuple
    
    def generate_and_compare(self, inputs: typing.Dict[str, typing.Any], labels: typing.Any, top_k: int = 5, objective_type: ObjectiveTypes = ObjectiveTypes.AutoRegressive, metrics: typing.Dict[str, float] = {}, good_examples: list = [], bad_examples: list = []) -> list:
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']
        texts = inputs['text']
        label_texts = labels['text']
        assert len(input_ids.shape) == 2, "Input ids must be a 2D tensor"
        assert len(attention_masks.shape) == 2, "Attention mask must be a 2D tensor"
        named_tuples = []
        generated = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, 
                                        max_new_tokens=self.get_block_size()//4, 
                                        do_sample=True, 
                                        top_k=self.top_k, 
                                        top_p=self.top_p, 
                                        temperature=self.temperature, 
                                        num_return_sequences=top_k, 
                                        return_dict_in_generate=True)
        decoded_str = self.tokenizer.batch_decode(generated["sequences"], skip_special_tokens=False)
        decoded_format = []
        for i in range(len(decoded_str)):
            tokenized_string = self.tokenizer.tokenize(decoded_str[i])
            # Filter out special tokens which are not <extra_id_0>
            tokenized_string = [token for token in tokenized_string if token not in self.special_token]
            # Combine tokens into a string
            decoded_str[i] = self.tokenizer.convert_tokens_to_string(tokenized_string)
        idx = 0
        acc_cnt_at_k = 0 if f"acc_cnt_at_{top_k}" not in metrics else metrics[f"acc_cnt_at_{top_k}"]
        training_ex_cnt = 0 if f"training_ex_cnt" not in metrics else metrics["training_ex_cnt"]
        for text, label in zip(texts, label_texts):
            generated = decoded_str[idx:idx+top_k]
            actual_format = self.training_data_parser(text, label)
            is_good_example = False
            for i in range(len(generated)):
                try:
                    decoded_format = self.training_data_parser(text, generated[i])
                    if actual_format.have_same_proof_steps(decoded_format):
                        acc_cnt_at_k += 1
                        is_good_example = True
                except:
                    pass
                if is_good_example:
                    break
            named_tuple = self.generation_named_tuple(text, label, generated)
            named_tuples.append(named_tuple)
            if is_good_example:
                good_examples.append(named_tuple)
            else:
                bad_examples.append(named_tuple)
            training_ex_cnt += 1
            idx += top_k
        metrics[f"acc_cnt_at_{top_k}"] = acc_cnt_at_k
        metrics[f"training_ex_cnt"] = training_ex_cnt
        metrics[f"acc_at_{top_k}"] = acc_cnt_at_k / training_ex_cnt
        self.code_t5_helper.cuda_context.free_cuda_cache_if_possible()
        return named_tuples

if __name__ == "__main__":
    from src.baselines.t5_prover.proof_search.common.data_format_layout import DataLayout, DataFormatLayoutTypes
    data_layout = DataLayout(DataFormatLayoutTypes.Start_Proof_End__GPTStyle, with_label=True)
    training_data_formatter = data_layout.get_layout_formatter()
    training_data_parser = data_layout.get_format_parser()
    training_args = TrainingArguments(
        output_dir=".temp_cache/code_t5_small_test", 
        do_train=True, 
        do_eval=True, 
        per_gpu_eval_batch_size=2, 
        per_gpu_train_batch_size=2, 
        seed=0xf00, 
        overwrite_output_dir=True, 
        logging_dir=".log/code_t5_small_test", 
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
        metric_for_best_model='eval_accuracy',
        greater_is_better=True)
    data_args = TrainingDataArguments(padding=True, truncation=True, max_length=512, padding_policy=PaddingPolicy.MAX_BATCH_LENGTH, shuffle=True)
    code_t5_trainable = CodeT5Trainable(training_data_formatter, training_data_parser, training_args, data_args)
    tokens1 = code_t5_trainable.tokenize("print('hello world')")
    tokens2 = code_t5_trainable.tokenize(["print('hello world')", "n*factorial(n-1)"])
    target = {"input_ids": tokens2['input_ids'][1, :].unsqueeze(0), "attention_mask": tokens2['attention_mask'][1, :].unsqueeze(0), "text": [tokens2['text']]}
    output, loss = code_t5_trainable.get_output_with_loss(tokens1, target)
    print(output)
    print(loss)
    generated_vals = code_t5_trainable.generate_and_compare(tokens2, tokens2, top_k=5)
    for generated_val in generated_vals:
        print(generated_val.input)
        print(generated_val.generated)