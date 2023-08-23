#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from src.baselines.t5_prover.llm_helpers.cuda_context import CudaContext
from src.baselines.t5_prover.llm_helpers.base_model import BaseLM, DecodingParameters, BeamSearchParameters, NucleusSamplingParameters, GenerationResult
import torch
import enum
import typing
import random
import math

class CodeT5Helper(BaseLM):
    class EmbeddingType(enum.Enum):
        AVG_LAST_HIDDEN_STATE = 1
        SUM_LAST_HIDDEN_STATE = 2
        CLS_TOKEN_LAST_HIDDEN_STATE = 3

    """Class to run inference on a T5 model.
    """

    def __init__(self, cuda_context: CudaContext = None):
        self.cuda_context: CudaContext = cuda_context if cuda_context is not None else CudaContext.get_default_context()
        self.tokenizer: RobertaTokenizer = None
        self.model: T5ForConditionalGeneration = None
    
    def _assert_model_loaded(self):
        if self.model is None:
            raise Exception("Model not loaded. Call load_model() first.")
    
    def load_model(self, model_name="Salesforce/codet5-small", model_max_length=512):
        self.model = self.cuda_context.try_get_gpu(T5ForConditionalGeneration.from_pretrained(model_name))
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, model_max_length=model_max_length)
        self.special_token = set(self.tokenizer.all_special_tokens)
        for token in self.tokenizer.all_special_tokens:
            if token.startswith("<extra_id"):
                self.special_token.remove(token)
        self.cuda_context.free_cuda_cache_if_possible()

    def generate(self, inputs: typing.Union[typing.List[str], str], decoding_params: DecodingParameters) -> typing.Union[GenerationResult, typing.List[GenerationResult]]:
        self._assert_model_loaded()
        self.cuda_context.free_cuda_cache_if_possible()
        is_beams_search = isinstance(decoding_params, BeamSearchParameters)
        is_nucleus_sampling = isinstance(decoding_params, NucleusSamplingParameters)
        assert is_beams_search or is_nucleus_sampling and (not (is_beams_search and is_nucleus_sampling)), "Decoding parameters must be either BeamSearchParameters or NucleusSamplingParameters"
        input_is_str = False
        if isinstance(inputs, str):
            inputs = [inputs]
            input_is_str = True
        encoding = self.tokenizer(
            inputs,
            padding=decoding_params.padding,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = self.cuda_context.try_get_gpu(encoding["input_ids"])
        attention_masks = self.cuda_context.try_get_gpu(encoding["attention_mask"])
        assert len(input_ids.shape) == 2, "Input ids must be a 2D tensor"
        assert len(attention_masks.shape) == 2, "Attention mask must be a 2D tensor"
        if is_nucleus_sampling:
            generated = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, 
                                            max_new_tokens=decoding_params.generation_max_length, 
                                            do_sample=True, 
                                            top_k=decoding_params.top_k, 
                                            top_p=decoding_params.top_p, # Nucleus sampling parameters
                                            temperature=decoding_params.temperature,
                                            num_return_sequences=decoding_params.num_samples, 
                                            return_dict_in_generate=True)
        elif is_beams_search:
            generated = self.model.generate(input_ids=input_ids, attention_mask=attention_masks,
                                            max_new_tokens=decoding_params.generation_max_length,
                                            num_beams=decoding_params.beam_width, # Beam search parameters
                                            num_return_sequences=decoding_params.num_samples,
                                            return_dict_in_generate=True)
        else:
            raise Exception("Invalid decoding parameters")
        target = generated["sequences"]
        decoded_str = self.tokenizer.batch_decode(target, skip_special_tokens=False)
        for i in range(len(decoded_str)):
            tokenized_string = self.tokenizer.tokenize(decoded_str[i])
            # Filter out special tokens which are not <extra_id_0>
            tokenized_string = [token for token in tokenized_string if token not in self.special_token]
            # Combine tokens into a string
            decoded_str[i] = self.tokenizer.convert_tokens_to_string(tokenized_string)

        target[target == self.tokenizer.pad_token_id] = -100
        neg_log_likelihoods = []
        # no gradient computation
        with torch.no_grad():
            for i in range(len(inputs)):
                inp_i = input_ids[i, :].unsqueeze(0)
                attention_i = attention_masks[i, :].unsqueeze(0)
                for j in range(decoding_params.num_samples):
                    target_j = target[i*decoding_params.num_samples+j].unsqueeze(0)
                    out_j = self.model(input_ids=inp_i, attention_mask=attention_i, labels=target_j)
                    neg_log_likelihoods.append(float(out_j.loss.detach().cpu().numpy()))
        idx = 0
        generation_res = []
        for text in inputs:
            generated_text = decoded_str[idx:idx+decoding_params.num_samples]
            probs = neg_log_likelihoods[idx:idx+decoding_params.num_samples]
            probs = [math.exp(-p) for p in probs] # Compute the probability of each generated sequence
            # sort by probability
            probs, generated_text = zip(*sorted(zip(probs, generated_text), reverse=True))
            res = GenerationResult(text, generated_text, probs)
            generation_res.append(res)
            idx += decoding_params.num_samples
        if input_is_str:
            return generation_res[0]
        else:
            return generation_res
    

if __name__ == "__main__":
    torch.manual_seed(0xf00)
    random.seed(0xf00)
    CudaContext.set_default_context(turn_off_cuda=True)
    t5_helper = CodeT5Helper()
    t5_helper.load_model("Salesforce/codet5-large")
    test_lm_input = \
"""def factorial(n):
    if n == <extra_id_0>:
        return <extra_id_1>
    else:
        return n * factorial(<extra_id_2>)
"""
    decoding_params = BeamSearchParameters(num_samples=5, beam_width=5, generation_max_length=50)
    new_gen = t5_helper.generate(test_lm_input, decoding_params=decoding_params)
    print(t5_helper.cuda_context.get_gpu_memory_map())
    print(f"Input sentence to LM:\n{test_lm_input}")
    for gen, prob in zip(new_gen.output, new_gen.probabilities):
        print(f"[{prob}]\t{gen}")
        print("-"*100)