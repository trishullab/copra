#!/usr/bin/env python3

"""
Parallel Execution Module

This module provides a unified interface for parallel execution that automatically
chooses between free-threading (Python 3.14t+) and multiprocessing (older versions).
"""

import sys
import time
from typing import Dict, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod


class ParallelExecutor(ABC):
    """Abstract base class for parallel execution."""

    @abstractmethod
    def execute_with_timeout(
        self,
        target: Callable,
        args: Tuple,
        timeout: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Execute a function in parallel with a timeout.

        Args:
            target: Function to execute
            args: Arguments to pass to the function
            timeout: Timeout in seconds

        Returns:
            Tuple of (result_dict, elapsed_time)
        """
        pass


class MultiprocessingExecutor(ParallelExecutor):
    """Executor using multiprocessing (for Python < 3.14)."""

    def execute_with_timeout(
        self,
        target: Callable,
        args: Tuple,
        timeout: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Execute function using multiprocessing."""
        import multiprocessing

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        # Add return_dict as first argument
        process_args = (return_dict,) + args

        p = multiprocessing.Process(target=target, args=process_args)

        tic_start = time.time()
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.kill()
            p.join()
        p.close()

        toc_end = time.time()
        elapsed_time = toc_end - tic_start

        # Convert to regular dict
        result = dict(return_dict) if len(return_dict) > 0 else None
        return_dict.clear()

        return result, elapsed_time


class FreeThreadingExecutor(ParallelExecutor):
    """Executor using free-threading (for Python 3.14t+)."""

    def execute_with_timeout(
        self,
        target: Callable,
        args: Tuple,
        timeout: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Execute function using threading with timeout."""
        import threading
        from queue import Queue

        result_queue: Queue = Queue()
        return_dict: Dict[str, Any] = {}

        def wrapper():
            """Wrapper to capture results."""
            try:
                # Add return_dict as first argument
                thread_args = (return_dict,) + args
                target(*thread_args)
                result_queue.put(("success", return_dict.copy()))
            except Exception as e:
                result_queue.put(("error", e))

        thread = threading.Thread(target=wrapper)

        tic_start = time.time()
        thread.start()
        thread.join(timeout=timeout)
        toc_end = time.time()
        elapsed_time = toc_end - tic_start

        # Check if thread completed
        if thread.is_alive():
            # Thread is still running after timeout
            # Note: We cannot forcefully kill threads in Python
            # The thread will continue running but we return timeout
            return None, elapsed_time

        # Get result from queue
        if not result_queue.empty():
            status, result = result_queue.get()
            if status == "success":
                return result, elapsed_time
            else:
                # Error occurred
                return None, elapsed_time

        return None, elapsed_time


def get_parallel_executor() -> ParallelExecutor:
    """
    Get the appropriate parallel executor based on Python version.

    Returns:
        ParallelExecutor instance (either FreeThreadingExecutor or MultiprocessingExecutor)
    """
    if sys.version_info >= (3, 14):
        # Check if free-threading is actually enabled
        try:
            if hasattr(sys, '_is_gil_enabled'):
                gil_enabled = sys._is_gil_enabled()
                if not gil_enabled:
                    # GIL is disabled, use free-threading
                    return FreeThreadingExecutor()
        except:
            pass

        # Fall back to free-threading executor anyway for 3.14+
        # (it will work even with GIL enabled, just less efficiently)
        return FreeThreadingExecutor()
    else:
        # Use multiprocessing for older Python versions
        return MultiprocessingExecutor()


# Singleton instance
_executor: Optional[ParallelExecutor] = None


def get_executor() -> ParallelExecutor:
    """
    Get the singleton parallel executor instance.

    Returns:
        ParallelExecutor instance
    """
    global _executor
    if _executor is None:
        _executor = get_parallel_executor()
    return _executor
