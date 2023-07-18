#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import json
import openai
import typing

class GptAccess(object):
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
            stop: list = ["\n"]) -> str:
        model = self.model_name if model is None else model
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
        usagae = response.usage
        self.usage["prompt_tokens"] += usagae.prompt_tokens
        self.usage["completion_tokens"] += usagae.completion_tokens
        self.usage["total_tokens"] += usagae.total_tokens
        return [{"role": choice.message.role, "content": choice.message.content} for choice in response.choices]

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
    openai_access = GptAccess(model_name="gpt-3.5-turbo")
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
        }
    ]
    print(openai_access.complete_chat(messages, max_tokens=15, n=5, temperature=0.8))
    pass