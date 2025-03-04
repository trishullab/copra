#!/usr/bin/env python3

import time
import typing
import os
import random
import logging
import threading
# from litellm import token_counter
from subprocess import Popen, PIPE, STDOUT
from copra.gpts.gpt_access import GptAccess
from copra.gpts.llama2_chat_format import Llama2FormatChat
from huggingface_hub import InferenceClient

class ServiceDownError(Exception):
    pass

class LlamaAccess(GptAccess):
    # Use this https://huggingface.co/blog/codellama#conversational-instructions for formatting instructions
    """
    This is not thread safe"""
    process = None
    model_name = None
    debug = False
    random_suffix = random.randint(0, 10**16)
    models_supported_name = ['codellama/CodeLlama-7b-Instruct-hf', 'EleutherAI/llemma_7b', 'morph-labs/morph-prover-v0-7b']
    logger : logging.Logger = None
    port = 8080
    docker_exit_signal = False
    litellm_exit_signal = False
    docker_logging_thread = None
    litellm_logging_thread = None
    def __init__(self, model_name: str | None = None, temperature = 0.0) -> None:
        assert model_name == LlamaAccess.model_name or model_name is None, "Model name must be initialized before use"
        assert LlamaAccess.process is not None, "LlamaAccess class must be initialized before use"
        self.secret_filepath = None
        self.model_name = model_name if model_name is not None else LlamaAccess.models_supported_name[0]
        self.temperature = temperature
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.is_open_ai_model = False
        self._llama2_format_chat = Llama2FormatChat()

    def __enter__(self):
        self.model_name = f"huggingface/{self.model_name}"
        self.interface = InferenceClient(model=f"http://localhost:{LlamaAccess.port}")
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

    def class_init(model_name: str = None, temperature = 0.0, port = 8080, debug = False, logger: logging.Logger = None):            
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
            LlamaAccess.docker_logging_thread = None
            LlamaAccess.port = port
        if not LlamaAccess._check_if_docker_running():
            LlamaAccess._start_service(model_name, temperature, port, debug)
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

    def _start_service(model_name: str, temperature = 0.0, port = 8080, debug = False) -> None:
        # Change the openai.api_key to the llama api key
        # Start the docker container for llama TGI
        docker_container_name = LlamaAccess._get_docker_container_name(model_name)
        cuda_visible_devices = os.popen('echo $CUDA_VISIBLE_DEVICES').read().strip()
        if cuda_visible_devices == '':
            cuda_visible_devices = '0'
        cmd = f'sh src/gpts/start_llama.sh {docker_container_name} {model_name} {port}'
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

    def token_counter(self, model_name: str, text: typing.List[str]) -> int:
        tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        text = " ".join([message["content"] for message in messages])
        enc = tokenizer.encode(text)
        num_tokens = len(enc.ids)
        return num_tokens

    def num_tokens_from_messages(self, messages, model=None):
        model = model if model is not None else self.model_name
        num_tokens = self.token_counter(model, messages=messages)
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
        temperature = None if temperature == 0.0 else temperature
        top_p = None if top_p == 1.0 else top_p
        try:
            outputs = []
            prompt_tokens = self.num_tokens_from_messages(messages)
            completion_tokens = 0
            prompt, role_names = self._llama2_format_chat(messages)
            # LlamaAccess.logger.debug(f"Prompt Received:\n{prompt}")
            for i in range(n):
                output = self.interface.text_generation(
                    prompt=prompt,
                    details=True,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop,
                    do_sample=i>0,
                )
                generated_text = output.generated_text
                finish_reason = output.details.finish_reason
                # LlamaAccess.logger.debug(f"Generated Text:\n{generated_text}")
                if finish_reason.value == "stop_sequence":
                    finish_reason = "stop"
                else:
                    finish_reason = finish_reason.value
                if finish_reason == "stop":
                    # Remove the stop token
                    for stop_token in stop:
                        if generated_text.endswith(stop_token):
                            generated_text = generated_text[:generated_text.rfind(stop_token)]
                            break
                generated_text = generated_text.strip()
                for role_name in role_names:
                    if generated_text.startswith(role_name):
                        generated_text = generated_text[len(role_name):].strip()
                        break
                    elif generated_text.startswith(f"`{role_name}`"):
                        generated_text = generated_text[len(f"`{role_name}`:"):].strip()
                        break
                outputs.append({'role': 'assistant', 'content': generated_text, 'finish_reason': finish_reason})
                completion_tokens += output.details.generated_tokens
            total_tokens = prompt_tokens + completion_tokens
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reason": outputs[-1]["finish_reason"] if len(outputs) > 0 else "stop"
            }
            self.usage['prompt_tokens'] += prompt_tokens
            self.usage['completion_tokens'] += completion_tokens
            self.usage['total_tokens'] += total_tokens
            return outputs, usage
        except:
            if not LlamaAccess._check_if_docker_running():
                raise ServiceDownError("Docker is shut down, restart the service")
            else:
                raise

    def kill():
        LlamaAccess.logger.info("Killing the docker processes")
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
        LlamaAccess.logger.info("Docker processes killed and logging threads stopped")
        try:
            LlamaAccess.process.stdin.close()
        except:
            pass
        time.sleep(2)
        try:
            LlamaAccess.process.stdout.close()
        except:
            pass
        time.sleep(2)
        LlamaAccess.logger.info("Docker stdin and stdout closed")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.log(logging.INFO, "Testing LlamaAccess")
    LlamaAccess.class_init(port=10005, logger=logger)
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
            messages = [messages[0]] + [messages[1+(i%len(messages[1:-1]))] for i in range(300)] + [messages[-1]]
            print(llama.num_tokens_from_messages(messages))
            print("Will call complete_chat soon")
            #time.sleep(30)
            print(llama.complete_chat(messages, max_tokens=50, n=2, temperature=0.0, stop=['.']))
    finally:
        LlamaAccess.class_kill()