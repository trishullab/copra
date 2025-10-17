# src/copra/vllm_min.py
from __future__ import annotations
import os, sys, time, signal, socket, subprocess
from typing import Optional, Iterable
import requests

def has_vllm():
    try:
        import vllm
        return True
    except ImportError:
        return False

def _healthy(base_url: str, api_key: str, model_id: str, t: float = 1.5) -> bool:
    try:
        r = requests.get(f"{base_url}/models",
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=t)
        if r.status_code != 200: 
            return False
        data = r.json().get("data", [])
        return any(m.get("id") == model_id for m in data)
    except Exception:
        return False

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def start_server(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    api_key: str = "token-abc123",
    gpu_ids: Optional[Iterable[int]] = None,     # e.g., [0,1]
    tensor_parallel: Optional[int] = None,       # default: len(gpu_ids) if provided
    max_model_len: Optional[int] = None,         # e.g., 16384
    gpu_mem_util: Optional[float] = None,        # e.g., 0.9
    extra_args: Optional[list[str]] = None,      # pass raw vLLM flags if needed
    wait_seconds: int = 120,
) -> tuple[str, Optional[subprocess.Popen]]:
    """
    Returns (base_url, proc). If a matching server is already running on host:port,
    proc will be None (we didn't start a new one).
    """
    if not has_vllm():
        raise ImportError("vLLM is not installed. Please install vLLM to use this function.")
    if port is None:
        port = _free_port()
    base_url = f"http://{host}:{port}/v1"

    # Reuse if already up and serving the requested model
    if _healthy(base_url, api_key, model):
        return base_url, None

    env = os.environ.copy()

    # Determine GPU configuration
    if gpu_ids is not None:
        # Explicit gpu_ids provided - use those
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
        if tensor_parallel is None:
            tensor_parallel = len(list(gpu_ids))
    elif "CUDA_VISIBLE_DEVICES" in env:
        # CUDA_VISIBLE_DEVICES is already set - parse it for tensor_parallel
        cuda_devices = env["CUDA_VISIBLE_DEVICES"]
        if cuda_devices.strip():  # Not empty
            # Parse comma-separated GPU IDs
            gpu_list = [d.strip() for d in cuda_devices.split(",") if d.strip()]
            if gpu_list and tensor_parallel is None:
                tensor_parallel = len(gpu_list)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--api-key", api_key,
        "--dtype", "auto",
        "--trust-remote-code",
    ]
    if tensor_parallel and tensor_parallel > 1:
        cmd += ["--tensor-parallel-size", str(tensor_parallel)]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if gpu_mem_util:
        cmd += ["--gpu-memory-utilization", str(gpu_mem_util)]
    if extra_args:
        cmd += extra_args

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # Wait until healthy
    t0 = time.time()
    while time.time() - t0 < wait_seconds:
        if proc.poll() is not None:
            # surface a bit of log if it crashed early
            tail = ""
            if proc.stdout:
                try:
                    tail = "".join(proc.stdout.readlines()[-50:])
                except Exception:
                    pass
            raise RuntimeError("vLLM exited early.\n" + tail)
        if _healthy(base_url, api_key, model):
            return base_url, proc
        time.sleep(0.5)

    # Time out: try to stop and raise
    stop_server(proc)
    raise TimeoutError(f"vLLM not ready after {wait_seconds}s at {base_url}")

def stop_server(proc: subprocess.Popen | None) -> None:
    if not has_vllm():
        return
    if not proc or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=6)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
