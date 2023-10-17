#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import time
import os
import openai
from subprocess import Popen, PIPE, STDOUT
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from src.gpts.gpt_access import GptAccess

class LlamaAccess(GptAccess):
    """
    This is not thread safe"""
    def __init__(self, model_name: str | None = None) -> None:
        self.secret_filepath = None
        self.models_supported_name = ['codellama/CodeLlama-7b-Instruct-hf']
        if model_name is not None:
            assert model_name in self.models_supported_name, f"Model name {model_name} not supported"
            self.model_name = model_name
        else:
            self.model_name = self.models_supported_name[0]
        self.debug = False
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.supports_usage_api = False

    def __enter__(self):
        self._start_service()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.kill()
    
    def _start_service(self) -> None:
        # Change the openai.api_key to the llama api key
        # Start the docker container for llama TGI
        cmd = f'sh src/gpts/start_llama.sh {self.model_name}'
        self.process = Popen(
            cmd, 
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = root_dir, 
            bufsize = 1, 
            universal_newlines = True)
        exit_wait = False
        while not exit_wait:
            line = self.process.stdout.readline().strip()
            if line:
                if self.debug:
                    print(f'Received {line}')
                if line.endswith('Connected'):
                    time.sleep(1)
                    exit_wait = True
            else:
                # sleep for a bit to avoid busy waiting
                time.sleep(0.02)
        self.proxy_process = Popen(
            f'litellm --model huggingface/{self.model_name} --api_base http://localhost:8080',
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = root_dir, 
            bufsize = 1, 
            universal_newlines = True
        )
        time.sleep(5)
        test_process = Popen(
            'litellm --test',
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = root_dir, 
            bufsize = 1, 
            universal_newlines = True
        )
        test_process.wait()
        if test_process.returncode != 0:
            if self.debug:
                print(test_process.stdout.read())
            test_process.kill()
            raise Exception('litellm test failed')
        else:
            if self.debug:
                print(test_process.stdout.read())
            test_process.kill()
        openai.api_key = "xyz"
        openai.api_base = "http://0.0.0.0:8000"
        self.model_name = f"huggingface/{self.model_name}"
        pass

    def kill(self):
        # Ensure that the process is killed
        os.system('docker stop $(docker ps -q)')
        # kill the litellm process
        os.system('killall litellm')
        self.process.kill()
        self.proxy_process.kill()
        try:
            self.process.stdin.close()
            self.proxy_process.stdin.close()
        except:
            pass

if __name__ == '__main__':
    with LlamaAccess() as llama:
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
        print(llama.complete_chat(messages, max_tokens=15, n=2, temperature=0.8))
        pass