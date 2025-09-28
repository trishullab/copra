#!/usr/bin/env python3
import os
import json
import typing
import unittest
import tiktoken
import copy
import boto3
from openai import OpenAI
from copra.tools.misc import is_open_ai_model, is_anthropic_model, is_bedrock_model

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
        },
        "o4-mini": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "gpt-5-mini": {
            "token_limit_per_min": 8000000,
            "request_limit_per_min": 8000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "claude-3-7-sonnet-20250219": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-7-sonnet-20250219-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-5-haiku-20241022-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-opus-20240229-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        },
        "deepseek.r1-v1:0": {
            "token_limit_per_min": 2000000,
            "request_limit_per_min": 1000,
            "max_token_per_prompt": int(1.2 * 10**5)
        }
    }

    secret_filepath_map = {
        "gpt-3.5-turbo": ".secrets/openai_key.json",
        "gpt-4": ".secrets/openai_key.json",
        "gpt-4-0314": ".secrets/openai_key.json",
        "gpt-4-0613": ".secrets/openai_key.json",
        "gpt-4-1106-preview": ".secrets/openai_key.json",
        "gpt-4o": ".secrets/openai_key.json",
        "gpt-4o-mini": ".secrets/openai_key.json",
        "o1-mini": ".secrets/openai_key.json",
        "o1": ".secrets/openai_key.json",
        "o3": ".secrets/openai_key.json",
        "o3-mini": ".secrets/openai_key.json",
        "o4-mini": ".secrets/openai_key.json",
        "gpt-5-mini": ".secrets/openai_key.json",
        "claude-3-7-sonnet-20250219": ".secrets/anthropic_key.json",
        "anthropic.claude-3-7-sonnet-20250219-v1:0": ".secrets/bedrock_key.json",
        "anthropic.claude-3-5-haiku-20241022-v1:0": ".secrets/bedrock_key.json",
        "anthropic.claude-3-5-sonnet-20241022-v2:0": ".secrets/bedrock_key.json",
        "anthropic.claude-3-5-sonnet-20240620-v1:0": ".secrets/bedrock_key.json",
        "anthropic.claude-3-opus-20240229-v1:0": ".secrets/bedrock_key.json",
        "anthropic.claude-3-haiku-20240307-v1:0": ".secrets/bedrock_key.json",
        "anthropic.claude-3-sonnet-20240229-v1:0": ".secrets/bedrock_key.json",
        "deepseek.r1-v1:0": ".secrets/bedrock_key.json",
    }

    base_url_map = {
        "claude-3-7-sonnet-20250219": "https://api.anthropic.com/v1"
    }

    def __init__(self,
                 secret_filepath: str = None,
                 model_name: typing.Optional[str] = None) -> None:
        assert secret_filepath is None or secret_filepath.endswith(".json"), "Secret filepath must be a .json file"
        if secret_filepath is None:
            # Use the default secret filepath based on the model name.
            assert model_name in self.secret_filepath_map, (
                f"Model {model_name} not supported. Supported models: {list(self.secret_filepath_map.keys())}"
            )
            self.secret_filepath = self.secret_filepath_map[model_name]
        else:
            self.secret_filepath = secret_filepath
        assert os.path.exists(self.secret_filepath), "Secret filepath does not exist"
        self.is_open_ai_model = is_open_ai_model(model_name)
        self.is_anthropic_model = is_anthropic_model(model_name)
        self.is_bedrock_model = is_bedrock_model(model_name)
        self._load_secret()
        assert sum([self.is_open_ai_model, self.is_anthropic_model, self.is_bedrock_model]) == 1, \
            "Model must be either OpenAI or Anthropic, not both."
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
        if self.is_open_ai_model or self.is_anthropic_model:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url_map.get(model_name, None))
        elif self.is_bedrock_model:
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
        else:
            raise Exception("Something went wrong with model name initialization")

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
        if self.is_bedrock_model:
            return self.complete_chat_bedrock(
                messages, 
                model,
                n, 
                max_tokens, 
                temperature, 
                top_p, 
                frequency_penalty, 
                presence_penalty, 
                stop, 
                reasoning_effort, 
                reasoning_token_count)
        stopping_reasons = "stop"
        if self.is_open_ai_model or self.is_anthropic_model:
            if self.model_name.startswith("o1") or \
            self.model_name.startswith("o3") or \
            self.model_name.startswith("o4") or \
            self.model_name.startswith("gpt-5") or \
            self.is_anthropic_model:
                messages = self.handle_thinking_messages(messages)
                return_responses, usage, stopping_reasons = \
                self.get_thinking_response(model, messages, max_tokens, stop, reasoning_token_count, reasoning_effort)
            else:
                # GPT-4o
                messages = self.handle_thinking_messages(messages)
                return_responses, usage, stopping_reasons = \
                self.get_gpt_4_o_response(model, messages, max_tokens, stop, temperature, top_p, frequency_penalty, presence_penalty, n)
        else:
            return_responses, usage, stopping_reasons = \
            self.get_response_generic(model, messages, max_tokens, stop, temperature, n)
        usage_dict = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "reason": stopping_reasons
        }
        return return_responses, usage_dict
    
    def handle_thinking_messages(self, messages: typing.List[typing.Dict[str, str]]) -> typing.List[typing.Dict[str, str]]:
        messages = copy.deepcopy(messages)
        for message in messages:
            if message["role"] == "system" and \
                self.model_name.startswith("o1"):
                message["role"] = "user" # No system role in o1
            name = message.get("name")
            if name is not None:
                message["content"] = f"\n{name}:```\n{message['content']}```\n"
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
        return messages

    def get_thinking_response(self, model, messages, max_tokens, stop : typing.List[str], reasoning_token_count, reasoning_effort) -> typing.Tuple[list, dict, str]:
        response = None
        if self.is_open_ai_model and model in {"o1-mini", "o4-mini", "gpt-5-mini"}:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens + reasoning_token_count
            )
        elif self.is_open_ai_model or self.is_anthropic_model:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens + reasoning_token_count,
                reasoning_effort=reasoning_effort,
                stop=stop
            )
        else:
            raise Exception("Something went wrong with model name initialization")
        assert response is not None, "No model found for the given model name"
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
        return return_responses, usage, stopping_reasons

    def get_gpt_4_o_response(self, model, messages, max_tokens, stop, temperature, top_p, frequency_penalty, presence_penalty, n) -> typing.Tuple[list, dict, str]:
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
        return return_responses, usage, stopping_reasons
    
    def complete_chat_bedrock(self,
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
        assert self.is_bedrock_model, "Model must be a Bedrock model"
        model = self.model_name if model is None else model
        model_id = f"us.{model}"
        if model.startswith("anthropic"):
            body_kwargs = {
                "max_tokens": max_tokens + reasoning_token_count,
                "temperature": temperature,
                "top_p": top_p,
                "stop_sequences": stop,
                "anthropic_version": "bedrock-2023-05-31"
            }
        else:
            # DeepSeek
            body_kwargs = {
                "max_tokens": max_tokens + reasoning_token_count,
                "temperature": temperature,
                "top_p": top_p,
                # "stop": stop # This can kill it in the reasoning phase
            }
        messages = self.handle_thinking_messages_bedrock(messages)
        if messages[0]["role"] == "system" and model.startswith("anthropic"):
            system_message = messages.pop(0)["content"]
            body_kwargs["system"] = system_message
        else:
            system_message = None
        body_kwargs["messages"] = messages
        response = self.bedrock_client.invoke_model(
            body=json.dumps(body_kwargs).encode('utf-8'),
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response['body'].read())
        if model.startswith("anthropic"):
            usage_response = result['usage']
            usage = {
                "prompt_tokens": usage_response['input_tokens'],
                "completion_tokens": usage_response['output_tokens'],
                "total_tokens": usage_response['input_tokens'] + usage_response['output_tokens']
            }
            contents = [content["text"] for content in result["content"] if content["type"] == "text"]
            stopping_reasons = result["stop_reason"]
            if stopping_reasons == "end_turn" or stopping_reasons == "stop_sequence":
                stopping_reasons = "stop"
        else:
            http_headers = response['ResponseMetadata']['HTTPHeaders']
            usage = {
                "prompt_tokens": int(http_headers['x-amzn-bedrock-input-token-count']),
                "completion_tokens": int(http_headers['x-amzn-bedrock-output-token-count']),
                "total_tokens": int(http_headers['x-amzn-bedrock-input-token-count']) + \
                int(http_headers['x-amzn-bedrock-output-token-count'])
            }
            contents = [content["message"]["content"] for content in result["choices"]]
            stopping_reasons = result["choices"][-1]["stop_reason"]
            if stopping_reasons != "length":
                stopping_reasons = "stop"
        return_responses = [{"role": "assistant", "content": content} for content in contents]
        for i in range(len(return_responses) - 1):
            return_responses[i]["finish_reason"] = "stop"
        if len(return_responses) > 0:
            return_responses[-1]["finish_reason"] = stopping_reasons
        usage = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "reason": stopping_reasons
        }
        return return_responses, usage

    def handle_thinking_messages_bedrock(self, messages: typing.List[typing.Dict[str, str]]) -> typing.List[typing.Dict[str, str]]:
        messages = copy.deepcopy(messages)
        for message in messages:
            # if message["role"] == "system": #and \
            #     # self.model_name.startswith("deepseek"):
            #     message["role"] = "user"
            name = message.get("name")
            if name is not None:
                message["content"] = f"\n{name}:```\n{message['content']}```\n"
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
        return messages


    def get_response_generic(self, model, messages, max_tokens, stop, temperature, n) -> typing.Tuple[list, dict, str]:
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
        return return_responses, usage, stopping_reasons

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
        if self.is_open_ai_model or self.is_anthropic_model:
            with open(self.secret_filepath, "r") as f:
                secret = json.load(f)
                self.api_key = secret["api_key"]
        elif self.is_bedrock_model:
            with open(self.secret_filepath, "r") as f:
                secret = json.load(f)
                self.region_name = secret["region_name"]
                self.aws_access_key_id = secret["aws_access_key_id"]
                self.aws_secret_access_key = secret["aws_secret_access_key"]
        else:
            raise Exception("Something went wrong with model name initialization")

    def get_usage(self) -> dict:
        return self.usage

# Integration Test Suite
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.secret_filepath = None

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
    
    def test_o4_mini_model(self):
        self._run_chat_test("o4-mini", token_count=100)

    def test_gpt4o_model(self):
        self._run_chat_test("gpt-4o", token_count=100)
    
    def test_o1_mini_model(self):
        self._run_chat_test("o1-mini", token_count=600)
    
    def test_o1_model(self):
        self._run_chat_test("o1", token_count=300)
    
    def test_claude_model(self):
        self._run_chat_test("claude-3-7-sonnet-20250219", token_count=300)

    def test_bedrock_claude_model(self):
        self._run_chat_test("anthropic.claude-3-7-sonnet-20250219-v1:0", token_count=300)
    
    def test_bedrock_deepseek_model(self):
        self._run_chat_test("deepseek.r1-v1:0", token_count=300)

    def test_gpt5_mini_model(self):
        self._run_chat_test("gpt-5-mini", token_count=100)

if __name__ == "__main__":
    unittest.main()