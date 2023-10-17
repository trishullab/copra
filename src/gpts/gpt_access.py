#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import json
import openai
import typing
import tiktoken

class GptAccess(object):
    gpt_model_info ={
        "gpt-3.5-turbo": {
            "token_limit_per_min": 45000, 
            "request_limit_per_min" : 3400, 
            "max_token_per_prompt" : int(3.75*2**10) # less than 4k because additional tokens are added at times
        },
        "gpt-4": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75*2**10) # less than 8k because additional tokens are added at times
        }
    }
    def __init__(self, 
        secret_filepath: str = ".secrets/openai_key.json",
        model_name: typing.Optional[str] = None) -> None:
        assert secret_filepath.endswith(".json"), "Secret filepath must be a .json file"
        assert os.path.exists(secret_filepath), "Secret filepath does not exist"
        self.secret_filepath = secret_filepath
        self._load_secret()
        self.models_supported = openai.Model.list().data
        self.models_supported_name = [model.id for model in self.models_supported]
        if model_name is not None:
            assert model_name in self.models_supported_name, f"Model name {model_name} not supported"
            self.model_name = model_name
        self.supports_usage_api = True
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        pass

    def get_models(self) -> list:
        return self.models_supported
    
    def complete_prompt(self, 
        prompt: str, 
        model: typing.Optional[str] = None,
        n: int = 1, 
        max_tokens: int = 5, 
        temperature: float = 0.25, 
        top_p: float = 1.0, 
        frequency_penalty: float = 0.0, 
        presence_penalty: float = 0.0, 
        stop: list = ["\n"],
        logprobs: int = 0) -> typing.List[typing.Tuple[str, float]]:
        model = self.model_name if model is None else model
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            logprobs=logprobs
            # best_of=n
        )
        if self.supports_usage_api:
            usagae = response.usage
            self.usage["prompt_tokens"] += usagae.prompt_tokens
            self.usage["completion_tokens"] += usagae.completion_tokens
            self.usage["total_tokens"] += usagae.total_tokens        
        resp = [(obj.text, sum(obj.logprobs.token_logprobs)) for obj in response.choices]
        resp.sort(key=lambda x: x[1], reverse=True)
        return resp

    def complete_chat(self,
            messages: typing.List[str],
            model: typing.Optional[str] = None,
            n: int = 1,
            max_tokens: int = 5,
            temperature: float = 0.25,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            stop: list = ["\n"]) -> typing.Tuple[list, dict]:
        model = self.model_name if model is None else model
        if self.supports_usage_api:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                n=n
            )
            usage = response.usage
            self.usage["prompt_tokens"] += usage.prompt_tokens
            self.usage["completion_tokens"] += usage.completion_tokens
            self.usage["total_tokens"] += usage.total_tokens
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )            
        return_responses = [{"role": choice.message.role, "content": choice.message.content} for choice in response.choices]
        for i in range(len(return_responses) - 1):
            return_responses[i]["finish_reason"] = "stop"
        if len(response.choices) > 0:
            return_responses[-1]["finish_reason"] = response.choices[-1].finish_reason
        if self.supports_usage_api:
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "reason": response.choices[-1].finish_reason if len(response.choices) > 0 else "stop"
            }
        else:
            usage_dict = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reason": response.choices[-1].finish_reason if len(response.choices) > 0 else "stop"
            }
        return return_responses, usage_dict
    
    def num_tokens_from_messages(self, messages, model=None):
        # Model name is like "gpt-3.5-turbo-0613"
        model = model if model is not None else self.model_name
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _load_secret(self) -> None:
        with open(self.secret_filepath, "r") as f:
            secret = json.load(f)
            # openai.organization = secret["organization"]
            openai.api_key = secret["api_key"]
        pass

    def get_usage(self) -> dict:
        return self.usage

if __name__ == "__main__":
    os.chdir(root_dir)
    # openai_access = GptAccess(model_name="gpt-3.5-turbo")
    openai_access = GptAccess(model_name="gpt-4")
    # openai_access = GptAccess(model_name="davinci")
    # print(openai_access.get_models())
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
        {
            "role": "user",
            "content": "Our idea seems to be scooped, don't know how to change direction now."
        }
    ]
    print(openai_access.complete_chat(messages, max_tokens=15, n=2, temperature=0.8))
    pass