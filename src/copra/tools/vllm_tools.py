# src/copra/vllm_min.py
from __future__ import annotations
import os, sys, time, signal, socket, subprocess
from typing import Optional, Iterable
import requests
import logging

def has_vllm():
    try:
        import vllm
        return True
    except ImportError:
        return False

def _healthy(base_url: str, api_key: str, model_id: str, t: float = 10.0, logger: Optional[logging.Logger] = None) -> bool:
    try:
        r = requests.get(f"{base_url}/models",
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=t)
        if r.status_code != 200:
            if logger:
                logger.info(f"vLLM health check failed: HTTP {r.status_code}")
            return False
        data = r.json().get("data", [])
        is_healthy = any(m.get("id") == model_id for m in data)
        if logger:
            if is_healthy:
                logger.info(f"✓ vLLM server is healthy and serving model: {model_id}")
            else:
                available_models = [m.get("id") for m in data]
                logger.info(f"vLLM health check failed: model {model_id} not in available models {available_models}")
        return is_healthy
    except requests.exceptions.Timeout:
        if logger:
            logger.info(f"vLLM health check timed out after {t}s - server may still be loading model")
        return False
    except requests.exceptions.ConnectionError:
        if logger:
            logger.info(f"vLLM health check failed: cannot connect to {base_url}")
        return False
    except Exception as e:
        if logger:
            logger.info(f"vLLM health check failed with exception: {e}")
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
    max_num_seqs: Optional[int] = None,          # max parallel sequences, reduce if OOM
    extra_args: Optional[list[str]] = None,      # pass raw vLLM flags if needed
    wait_seconds: int = 120,
    logger: Optional[logging.Logger] = None,     # optional logger for progress tracking
    log_file: Optional[str] = None,              # optional path to save vLLM server logs
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

    if logger:
        logger.info(f"Starting vLLM server for model: {model}")
        logger.info(f"Server will be accessible at: {base_url}")

    # Reuse if already up and serving the requested model
    if logger:
        logger.info(f"Checking if vLLM server is already running at {base_url}...")
    if _healthy(base_url, api_key, model, logger=logger):
        if logger:
            logger.info(f"✓ vLLM server already running and healthy, reusing existing server")
        return base_url, None

    if logger:
        logger.info(f"No existing server found, starting new vLLM server...")

    env = os.environ.copy()

    # Determine GPU configuration
    if gpu_ids is not None:
        # Explicit gpu_ids provided - use those
        gpu_ids_list = list(gpu_ids)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids_list)
        if tensor_parallel is None:
            tensor_parallel = len(gpu_ids_list)
        if logger:
            logger.info(f"GPU configuration: using GPUs {gpu_ids_list} with tensor_parallel={tensor_parallel}")
    elif "CUDA_VISIBLE_DEVICES" in env:
        # CUDA_VISIBLE_DEVICES is already set - parse it for tensor_parallel
        cuda_devices = env["CUDA_VISIBLE_DEVICES"]
        if cuda_devices.strip():  # Not empty
            # Parse comma-separated GPU IDs
            gpu_list = [d.strip() for d in cuda_devices.split(",") if d.strip()]
            if gpu_list and tensor_parallel is None:
                tensor_parallel = len(gpu_list)
            if logger:
                logger.info(f"GPU configuration: using CUDA_VISIBLE_DEVICES={cuda_devices} with tensor_parallel={tensor_parallel}")
    else:
        if logger:
            logger.info(f"GPU configuration: using default (single GPU or auto-detect)")

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
        if logger:
            logger.info(f"Setting tensor-parallel-size={tensor_parallel}")
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
        if logger:
            logger.info(f"Setting max-model-len={max_model_len}")
    if gpu_mem_util:
        cmd += ["--gpu-memory-utilization", str(gpu_mem_util)]
        if logger:
            logger.info(f"Setting gpu-memory-utilization={gpu_mem_util}")
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
        if logger:
            logger.info(f"Setting max-num-seqs={max_num_seqs} (controls parallel sequences)")
    if extra_args:
        cmd += extra_args
        if logger:
            logger.info(f"Additional vLLM args: {' '.join(extra_args)}")

    if logger:
        logger.info(f"Launching vLLM server process...")
        logger.info(f"Command: {' '.join(cmd)}")

    # Determine where to redirect vLLM server output
    if log_file:
        if logger:
            logger.info(f"vLLM server logs will be saved to: {log_file}")
        # Create parent directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        # Open log file for writing
        log_fd = open(log_file, 'w', buffering=1)
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fd, stderr=subprocess.STDOUT, text=True
        )
    else:
        # Keep output in PIPE (original behavior)
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

    if logger:
        logger.info(f"vLLM server process started (PID: {proc.pid})")
        if log_file:
            logger.info(f"Monitor server logs with: tail -f {log_file}")
        logger.info(f"Waiting up to {wait_seconds}s for server to become healthy...")

    # Wait until healthy
    t0 = time.time()
    last_log_time = t0
    check_count = 0
    while time.time() - t0 < wait_seconds:
        elapsed = time.time() - t0

        # Log progress every 10 seconds
        if logger and (elapsed - (last_log_time - t0)) >= 10:
            logger.info(f"Still waiting for vLLM server... ({int(elapsed)}s / {wait_seconds}s)")
            last_log_time = time.time()

        if proc.poll() is not None:
            # surface a bit of log if it crashed early
            tail = ""
            if proc.stdout:
                try:
                    tail = "".join(proc.stdout.readlines()[-50:])
                except Exception:
                    pass
            error_msg = f"vLLM server process exited early (after {int(elapsed)}s)\n{tail}"
            if logger:
                logger.error(error_msg)
            raise RuntimeError(error_msg)

        check_count += 1
        if _healthy(base_url, api_key, model, logger=logger):
            if logger:
                logger.info(f"✓ vLLM server is ready! (took {int(elapsed)}s, {check_count} health checks)")
            return base_url, proc
        time.sleep(5)

    # Time out: try to stop and raise
    if logger:
        logger.error(f"vLLM server failed to become healthy after {wait_seconds}s")
        logger.info(f"Attempting to stop vLLM server process...")
    stop_server(proc)
    raise TimeoutError(f"vLLM not ready after {wait_seconds}s at {base_url}")

def stop_server(proc: subprocess.Popen | None, logger: Optional[logging.Logger] = None) -> None:
    if not has_vllm():
        return
    if not proc or proc.poll() is not None:
        if logger:
            logger.info("vLLM server process already stopped or not running")
        return

    if logger:
        logger.info(f"Stopping vLLM server process (PID: {proc.pid})...")

    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=6)
        if logger:
            logger.info("✓ vLLM server stopped gracefully (SIGINT)")
    except Exception:
        if logger:
            logger.info("SIGINT failed, trying SIGTERM...")
        try:
            proc.terminate()
            proc.wait(timeout=4)
            if logger:
                logger.info("✓ vLLM server stopped (SIGTERM)")
        except Exception:
            if logger:
                logger.info("SIGTERM failed, forcing kill...")
            try:
                proc.kill()
                if logger:
                    logger.info("✓ vLLM server killed (SIGKILL)")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to kill vLLM server process: {e}")
