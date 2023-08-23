#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import torch
import numpy as np

class CudaContext:
    __cuda_context = None
    
    @staticmethod
    def get_default_context():
        if CudaContext.__cuda_context is None:
            CudaContext.__cuda_context = CudaContext()
        return CudaContext.__cuda_context
    
    @staticmethod
    def set_default_context(turn_off_cuda=False):
        CudaContext.__cuda_context = CudaContext(turn_off_cuda=turn_off_cuda)

    def __init__(self, turn_off_cuda=False, device_id = 0):
        self.turn_off_cuda = turn_off_cuda
        self.device_id = device_id

    def is_cuda_available(self):
        return not self.turn_off_cuda and torch.cuda.is_available()

    def free_cuda_cache_if_possible(self):
        if self.is_cuda_available():
            torch.cuda.empty_cache()

    def try_get_gpu(self, model):
        self.free_cuda_cache_if_possible()
        if isinstance(model, np.ndarray):
            return torch.from_numpy(model).to(torch.device(f'cuda:{self.device_id}')) if self.is_cuda_available() else torch.from_numpy(model)
        else:
            return model.to(torch.device(f'cuda:{self.device_id}')) if self.is_cuda_available() else model

    def get_param_count(self, model, only_trainable=False):
        return sum(p.numel() for p in model.parameters() if (not only_trainable) or p.requires_grad)

    def is_cuda_model(self, model):
        return next(model.parameters()).is_cuda

    def get_gpu_usage(self):
        return torch.cuda.get_device_properties(f"cuda:{self.device_id}").total_memory

    def get_gpu_memory_map(self):
        import subprocess
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        gpu_memory = [str(int(_)/2**10) + " GiB" for _ in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map