#!/usr/bin/env python3
import os
import json
import typing
import unittest
import tiktoken
import copy
from openai import OpenAI

class GptAccess:
    # Static dictionary of model information.
    gpt_model_info = {
        "gpt-3.5-turbo": {
            "token_limit_per_min": 45000,
            "request_limit_per_min": 3400,
            "max_token_per_prompt": int(3.75 * 2**10)
        },
        "gpt-4": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75 * 2**10)
        },
        "gpt-4-0314": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75 * 2**10)
        },
        "gpt-4-0613": {
            "token_limit_per_min": 20000,
            "request_limit_per_min": 100,
            "max_token_per_prompt": int(7.75 * 2**10)
        },
        "gpt-4-1106-preview": {
            "token_limit_per_min": 150000,
            "request_limit_per_min": 20,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "gpt-4o": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "gpt-4o-mini": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "o1-mini": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "o1": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "o3": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "o3-mini": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        }
    }

    def __init__(self,
                 secret_filepath: str = ".secrets/openai_key.json",
                 model_name: typing.Optional[str] = None) -> None:
        assert secret_filepath.endswith(".json"), "Secret filepath must be a .json file"
        assert os.path.exists(secret_filepath), "Secret filepath does not exist"
        self.secret_filepath = secret_filepath
        self._load_secret()
        self.is_open_ai_model = True
        # Use our static dictionary keys as the supported model list.
        self.models_supported_name = list(self.gpt_model_info.keys())
        if model_name is not None:
            assert model_name in self.models_supported_name, (
                f"Model {model_name} not supported. Supported models: {self.models_supported_name}"
            )
            self.model_name = model_name
        else:
            self.model_name = self.models_supported_name[0]
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # Create the OpenAI client instance.
        self.client = OpenAI(api_key=self.api_key)

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
        response = self.client.completions.create(
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
        )
        usage = response.usage
        self.usage["prompt_tokens"] += usage.prompt_tokens
        self.usage["completion_tokens"] += usage.completion_tokens
        self.usage["total_tokens"] += usage.total_tokens
        results = [(choice.text, sum(choice.logprobs.token_logprobs)) for choice in response.choices]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def complete_chat(self,
            messages: typing.List[typing.Dict[str, str]],
            model: typing.Optional[str] = None,
            n: int = 1,
            max_tokens: int = 5,
            temperature: float = 0.25,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            stop: list = [],
            reasoning_effort: str = "low", # low, medium, high
            reasoning_token_count: int = 350) -> typing.Tuple[list, dict]:
        assert isinstance(messages, list), "Messages must be a list"
        assert len(messages) > 0, "Messages list cannot be empty"
        assert reasoning_effort in ["low", "medium", "high"], "Reasoning effort must be one of: low, medium, high"
        model = self.model_name if model is None else model
        stopping_reasons = "stop"
        if self.is_open_ai_model:
            if self.model_name.startswith("o1") or \
            self.model_name.startswith("o3"): 
                messages = copy.deepcopy(messages)
                for message in messages:
                    if message["role"] == "system" and self.model_name.startswith("o1"):
                        message["role"] = "user" # No system role in o1
                    name = message.get("name")
                    if name is not None:
                        message["content"] = f"```\n{name}:\n{message['content']}```\n"
                        message.pop("name")
                # Now merge all same role messages occurring together into one message
                merged_messages = []
                for message in messages:
                    if len(merged_messages) == 0 or merged_messages[-1]["role"] != message["role"]:
                        merged_messages.append(message)
                    else:
                        merged_messages[-1]["content"] += message["content"]
                messages = merged_messages
                for message in messages:
                    for key in list(message.keys()):
                        if key not in ["role", "content"]:
                            message.pop(key)
                if self.model_name == "o1-mini":
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=max_tokens + reasoning_token_count,
                        stop=stop
                    )
                    usage = response.usage
                    self.usage["prompt_tokens"] += usage.prompt_tokens
                    self.usage["completion_tokens"] += usage.completion_tokens
                    self.usage["total_tokens"] += usage.total_tokens
                    return_responses = [{"role": choice.message.role, "content": choice.message.content} for choice in response.choices]
                    for i in range(len(return_responses) - 1):
                        return_responses[i]["finish_reason"] = "stop"
                    if len(response.choices) > 0:
                        return_responses[-1]["finish_reason"] = response.choices[-1].finish_reason
                    stopping_reasons = response.choices[-1].finish_reason if len(response.choices) > 0 else "stop"
                    usage = {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    }                    
                else:
                    response = self.client.responses.create(
                        model=model,
                        reasoning={"effort": reasoning_effort},
                        input=messages,
                        max_output_tokens=max_tokens + reasoning_token_count
                    )
                    usage = response.usage
                    self.usage["prompt_tokens"] += usage.input_tokens
                    self.usage["completion_tokens"] += usage.output_tokens
                    self.usage["total_tokens"] += usage.total_tokens
                    return_responses = [{"role": "assistant", "content": response.output_text}]
                    return_responses[-1]["finish_reason"] = "stop" if response.status == "completed" else "length"
                    stopping_reasons = "stop" if response.status == "completed" else "length"
                    usage = {
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens
                    }
            else:
                response = self.client.chat.completions.create(
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
                return_responses = [{"role": choice.message.role, "content": choice.message.content} for choice in response.choices]
                for i in range(len(return_responses) - 1):
                    return_responses[i]["finish_reason"] = "stop"
                if len(response.choices) > 0:
                    return_responses[-1]["finish_reason"] = response.choices[-1].finish_reason
                stopping_reasons = response.choices[-1].finish_reason if len(response.choices) > 0 else "stop"
                usage = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                n=n
            )
            usage = response.usage
            self.usage["prompt_tokens"] += usage.prompt_tokens
            self.usage["completion_tokens"] += usage.completion_tokens
            self.usage["total_tokens"] += usage.total_tokens
            return_responses = [{"role": choice.message.role, "content": choice.message.content} for choice in response.choices]
            for i in range(len(return_responses) - 1):
                return_responses[i]["finish_reason"] = "stop"
            if len(response.choices) > 0:
                return_responses[-1]["finish_reason"] = response.choices[-1].finish_reason
            stopping_reasons = response.choices[-1].finish_reason if len(response.choices) > 0 else "stop"
        usage_dict = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "reason": stopping_reasons
        }
        return return_responses, usage_dict

    def num_tokens_from_messages(self, messages, model=None):
        model = model if model is not None else self.model_name
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 4
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
        num_tokens += 2
        return num_tokens

    def _load_secret(self) -> None:
        with open(self.secret_filepath, "r") as f:
            secret = json.load(f)
            self.api_key = secret["api_key"]

    def get_usage(self) -> dict:
        return self.usage

# Integration Test Suite
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.secret_filepath = ".secrets/openai_key.json"
        if not os.path.exists(self.secret_filepath):
            self.skipTest("Secret file not found. Skipping integration tests.")

    def _run_chat_test(self, model_name: str, token_count: int):
        try:
            access = GptAccess(secret_filepath=self.secret_filepath, model_name=model_name)
        except AssertionError as e:
            self.skipTest(f"Model {model_name} not supported: {e}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant that translates corporate jargon into plain English."},
            {"role": "system", "name": "example_user", "content": "New synergies will help drive top-line growth."},
            {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
            {"role": "system", "name": "example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
            {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
            {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
            {"role": "user", "content": "Our idea seems to be scooped, don't know how to change direction now."}
        ]
        responses, usage = access.complete_chat(messages, max_tokens=token_count, n=1, temperature=0.7)
        self.assertIsInstance(responses, list)
        self.assertGreater(len(responses), 0)
        self.assertIn("content", responses[0])
        self.assertTrue(len(responses[0]["content"]) > 0, f"No content returned for model {model_name}")
        print(f"Model: {model_name}, Response: {responses[0]['content']}")

    def test_o3_mini_model(self):
        self._run_chat_test("o3-mini", token_count=100)

    def test_gpt4o_model(self):
        self._run_chat_test("gpt-4o", token_count=100)
    
    def test_o1_mini_model(self):
        self._run_chat_test("o1-mini", token_count=300)
    
    def test_o1_model(self):
        self._run_chat_test("o1", token_count=300)

if __name__ == "__main__":
    unittest.main()