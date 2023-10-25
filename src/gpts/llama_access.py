#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import time
import typing
import os
import openai
import random
import logging
import threading
from litellm import token_counter
from subprocess import Popen, PIPE, STDOUT
from src.gpts.gpt_access import GptAccess

class ServiceDownError(Exception):
    pass

class LlamaAccess(GptAccess):
    """
    This is not thread safe"""
    process = None
    proxy_process = None
    model_name = None
    debug = False
    random_suffix = random.randint(0, 10**16)
    models_supported_name = ['codellama/CodeLlama-7b-Instruct-hf', 'EleutherAI/llemma_7b']
    logger : logging.Logger = None
    docker_exit_signal = False
    litellm_exit_signal = False
    docker_logging_thread = None
    litellm_logging_thread = None
    def __init__(self, model_name: str | None = None, temperature = 0.0) -> None:
        assert model_name == LlamaAccess.model_name or model_name is None, "Model name must be initialized before use"
        assert LlamaAccess.process is not None, "LlamaAccess class must be initialized before use"
        assert LlamaAccess.proxy_process is not None, "LlamaAccess class must be initialized before use"
        self.secret_filepath = None
        self.model_name = model_name if model_name is not None else LlamaAccess.models_supported_name[0]
        self.temperature = temperature
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.is_open_ai_model = False

    def __enter__(self):
        self.model_name = f"huggingface/{self.model_name}"
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # self.kill() # only kill the service if this is the last object
        pass
    
    def _get_docker_container_name(model_name: str) -> str:
        return model_name.replace("/","-") + f"-{LlamaAccess.random_suffix}"

    def _get_docker_container_id(model_name: str) -> str:
        return os.popen(f"docker ps -q --filter='NAME={LlamaAccess._get_docker_container_name(model_name)}'").read().strip()

    def _check_if_docker_running() -> bool:
        try:
            return LlamaAccess._get_docker_container_id(LlamaAccess.model_name) != ""
        except:
            return False

    def class_init(model_name: str = None, temperature = 0.0, debug = False, logger: logging.Logger = None):            
        if model_name is None:
            model_name = LlamaAccess.models_supported_name[0]
        elif model_name is not None:
            assert model_name in LlamaAccess.models_supported_name, f"Model name {model_name} not supported"
        # Check if docker is running
        if LlamaAccess.model_name is None:
            LlamaAccess.model_name = model_name
            LlamaAccess.debug = debug
            LlamaAccess.logger = logger if logger is not None else logging.getLogger(__name__)
            LlamaAccess.docker_exit_signal = False
            LlamaAccess.litellm_exit_signal = False
        if not LlamaAccess._check_if_docker_running():
            LlamaAccess._start_service(model_name, temperature, debug)
            openai.api_key = "xyz"
            openai.api_base = "http://0.0.0.0:8000"
            openai.api_requestor.TIMEOUT_SECS = 11*60
        pass

    def class_kill():
        LlamaAccess.kill()
    
    def _docker_service_logs():
        try:
            while not LlamaAccess.docker_exit_signal:
                line = LlamaAccess.process.stdout.readline().strip()
                if line:
                    LlamaAccess.logger.info(f'Docker:\n {line}')
                else:
                    # sleep for a bit to avoid busy waiting
                    time.sleep(0.02)
        except:
            pass
    
    def _litellm_logs():
        try:
            while not LlamaAccess.litellm_exit_signal:
                line = LlamaAccess.proxy_process.stdout.readline().strip()
                if line:
                    LlamaAccess.logger.info(f'Litellm:\n {line}')
                else:
                    # sleep for a bit to avoid busy waiting
                    time.sleep(0.02)
        except:
            pass

    def _start_service(model_name: str, temperature = 0.0, debug = False) -> None:
        # Change the openai.api_key to the llama api key
        # Start the docker container for llama TGI
        docker_container_name = LlamaAccess._get_docker_container_name(model_name)
        cmd = f'sh src/gpts/start_llama.sh {docker_container_name} {model_name}'
        LlamaAccess.process = Popen(
            cmd, 
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = root_dir, 
            bufsize = 1, 
            universal_newlines = True)
        exit_wait = False
        start_time = time.time()
        retry = 3
        while not exit_wait and retry > 0:
            line = LlamaAccess.process.stdout.readline().strip()
            if line:
                LlamaAccess.logger.info(f'Docker:\n {line}')
                if "Error" in line or "error" in line:
                    LlamaAccess.process.kill()
                    raise Exception(f'Failed to start docker container {docker_container_name}, because of error: \n{line}')
                if line.endswith('Connected'):
                    time.sleep(1)
                    exit_wait = True
            else:
                # sleep for a bit to avoid busy waiting
                time.sleep(0.02)
            end_time = time.time()
            if end_time - start_time > 400:
                LlamaAccess.process.kill()
                LlamaAccess.kill()
                LlamaAccess.process = Popen(
                    cmd, 
                    shell = True, 
                    stdin = PIPE, 
                    stdout = PIPE, 
                    stderr = STDOUT,
                    cwd = root_dir, 
                    bufsize = 1, 
                    universal_newlines = True)
                exit_wait = False
                start_time = time.time()
                retry -= 1
        
        # Start the docker logging thread
        LlamaAccess.docker_exit_signal = False
        LlamaAccess.docker_logging_thread = threading.Thread(target=LlamaAccess._docker_service_logs)
        LlamaAccess.docker_logging_thread.start()
        # Kill if litellm is running
        try:
            os.popen('pkill litellm').read()
        except:
            pass

        LlamaAccess.proxy_process = Popen(
            f'litellm --model huggingface/{model_name} --api_base http://localhost:8080 --temperature {temperature}',
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = root_dir, 
            bufsize = 1, 
            universal_newlines = True
        )
        time.sleep(5)
        LlamaAccess.litellm_exit_signal = False
        LlamaAccess.litellm_logging_thread = threading.Thread(target=LlamaAccess._litellm_logs)
        LlamaAccess.litellm_logging_thread.start()
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
            if debug:
                print(test_process.stdout.read())
            test_process.kill()
            raise Exception('litellm test failed')
        else:
            if debug:
                print(test_process.stdout.read())
            test_process.kill()
        pass

    def num_tokens_from_messages(self, messages, model=None):
        model = model if model is not None else self.model_name
        num_tokens = token_counter(model, messages=messages)
        return num_tokens
    
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
        try:
            return super().complete_chat(messages, model, n, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop)
        except:
            if not LlamaAccess._check_if_docker_running():
                raise ServiceDownError("Docker is shut down, restart the service")
            else:
                raise
        pass

    def kill():
        LlamaAccess.logger.info("Killing the docker and litellm processes")
        # kill the litellm process
        try:
            LlamaAccess.proxy_process.kill()
        except:
            pass
        time.sleep(2)
        try:
            os.popen('pkill litellm').read()
        except:
            pass
        time.sleep(2)
        LlamaAccess.logger.info("Litellm stopped")
        docker_container_name = LlamaAccess._get_docker_container_name(LlamaAccess.model_name)
        if LlamaAccess._check_if_docker_running():
            docker_name = os.popen(f"docker stop {docker_container_name}").read().strip()
            assert docker_name == docker_container_name, f"Failed to stop container {docker_container_name}"
            time.sleep(2)
            LlamaAccess.logger.info(f"Docker Container {docker_container_name} stopped")
        else:
            LlamaAccess.logger.info(f"Docker Container {docker_container_name} already stopped")
        # Remove the docker container
        docker_name = os.popen(f"docker rm {docker_container_name}").read().strip()
        time.sleep(2)
        if docker_name == '':
            LlamaAccess.logger.info(f"Docker Container {docker_container_name} already removed")
        else:
            assert docker_name == docker_container_name, f"Failed to remove container {LlamaAccess._get_docker_container_name(LlamaAccess.model_name)}"
            LlamaAccess.logger.info(f"Docker Container {docker_container_name} removed")
        try:
            LlamaAccess.process.kill()
        except:
            pass
        time.sleep(2)
        # Stop logging threads
        LlamaAccess.docker_exit_signal = True
        LlamaAccess.litellm_exit_signal = True
        LlamaAccess.docker_logging_thread.join()
        LlamaAccess.litellm_logging_thread.join()
        time.sleep(2)
        LlamaAccess.logger.info("Docker and litellm processes killed and logging threads stopped")
        try:
            LlamaAccess.process.stdin.close()
        except:
            pass
        time.sleep(2)
        try:
            LlamaAccess.proxy_process.stdin.close()
        except:
            pass
        time.sleep(2)
        try:
            LlamaAccess.process.stdout.close()
        except:
            pass
        time.sleep(2)
        try:
            LlamaAccess.proxy_process.stdout.close()
        except:
            pass
        LlamaAccess.logger.info("Docker and litellm stdin and stdout closed")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.log(logging.INFO, "Testing LlamaAccess")
    LlamaAccess.class_init(logger=logger)
    try:
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
                    "role": "system",
                    "name": "example_user",
                    "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
                },
                {
                    "role": "system",
                    "name": "example_assistant", 
                    "content": "Our idea seems to be scooped, don't know how to change direction now."
                },
                {
                    "role": "user",
                    "content": "We changed the direction of the project, but we don't have time to do it.",
                }
            ]
            messages = [messages[1+(i%len(messages[1:-1]))] for i in range(300)] + [messages[-1]]
            print(llama.num_tokens_from_messages(messages))
            print("Will call complete_chat soon")
            time.sleep(30)
            print(llama.complete_chat(messages, max_tokens=50, n=2, temperature=0.2, stop=['.']))
    finally:
        LlamaAccess.class_kill()