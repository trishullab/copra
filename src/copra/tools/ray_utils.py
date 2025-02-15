#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import ray
import typing
import psutil
import logging
import gc

class RayUtils(object):

    @staticmethod
    def init_ray(num_of_cpus: int = 10, object_store_memory_in_gb: float = 25, memory_in_gb: float = 0.5):
        gb = 2**30
        object_store_memory = int(object_store_memory_in_gb * gb)
        memory = int(memory_in_gb * gb)
        return ray.init(num_cpus=num_of_cpus, object_store_memory=object_store_memory, _memory=memory, ignore_reinit_error=True)

    @staticmethod
    def ray_run_within_parallel_limits(
        max_parallel: int,
        num_objects: int,
        transform_outputs: typing.Callable[[typing.List[typing.Any]], None],
        prepare_next: typing.Callable[[int], typing.List[typing.Any]],
        create_remotes: typing.Callable[[typing.List[typing.Any]], typing.List[typing.Any]],
        logger: logging.Logger = None,
        turn_off_logging: bool = False
    ):
        logger = logger or logging.getLogger(__name__)
        idx = 0
        next_batch = prepare_next(max_parallel)
        if not turn_off_logging:
            logger.info(f"Loading next_batch: {len(next_batch)}, max_parallel: {max_parallel}")
        assert len(next_batch) <= max_parallel, f"next_batch: {len(next_batch)}, max_parallel: {max_parallel}"
        process = psutil.Process()
        if not turn_off_logging:
            logger.info(f"[Process Id = {process.pid}] [After Next Batch] Memory used: {process.memory_info().rss/2**30} GiB")
        remotes = create_remotes(next_batch)
        process = psutil.Process()
        if not turn_off_logging:
            logger.info(f"[Process Id = {process.pid}] [After Create] Memory used: {process.memory_info().rss/2**30} GiB")
            logger.info(f"Created remotes: {len(remotes)}")
        diff_remotes = len(remotes)
        while idx < num_objects or len(remotes) > 0:
            idx += diff_remotes
            idx = min(idx, num_objects)
            if not turn_off_logging:
                logger.info(f"Waiting for idx: {idx}, num_objects: {num_objects}, len(remotes): {len(remotes)}")
            ready, remotes = ray.wait(remotes)
            if len(ready) > 0:
                if not turn_off_logging:
                    logger.info(f"Got ready: {len(ready)}")
                process = psutil.Process()
                if not turn_off_logging:
                    logger.info(f"[Process Id = {process.pid}] [After Ready] Memory used: {process.memory_info().rss/2**30} GiB")
                results = ray.get(ready)
                transform_outputs(results)
                process = psutil.Process()
                if not turn_off_logging:
                    logger.info(f"[Process Id = {process.pid}] [After Transform] Memory used: {process.memory_info().rss/2**30} GiB")
                next_batch = prepare_next(len(results))
                process = psutil.Process()
                if not turn_off_logging:
                    logger.info(f"[Process Id = {process.pid}] [After Next Batch] Memory used: {process.memory_info().rss/2**30} GiB")
                assert len(next_batch) <= len(results), f"next_batch: {len(next_batch)}, ready: {len(results)}"
                new_remotes = create_remotes(next_batch)
                process = psutil.Process()
                if not turn_off_logging:
                    logger.info(f"[Process Id = {process.pid}] [After Create] Memory used: {process.memory_info().rss/2**30} GiB")
                remotes.extend(new_remotes)
                diff_remotes = len(new_remotes)
                # Delete results to free up memory
                del results
                process = psutil.Process()
                if not turn_off_logging:
                    logger.info(f"[Process Id = {process.pid}] [After Delete] Memory used: {process.memory_info().rss/2**30} GiB")
                    logger.info(f"Running GC collect")
                gc.collect()
                process = psutil.Process()
                if not turn_off_logging:
                    logger.info(f"[Process Id = {process.pid}] [After GC] Memory used: {process.memory_info().rss/2**30} GiB")
            else:
                diff_remotes = 0

if __name__ == "__main__":
    import os
    import time
    import random

    log_folder = f".log/ray_utils"
    os.makedirs(log_folder, exist_ok=True)
    log_file = f"{log_folder}/ray_utils-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)
    size = 1000
    example_cnt = 100000
    last_job_idx = 0
    total_sum = 0
    total_sum_serial = 0
    job_spec = [[random.random() for _ in range(example_cnt)] for _ in range(size)]

    @ray.remote
    def _do_job(job):
        idx, arr = job
        for i in range(size*10):
            # This is just to stress the CPUs
            sum_val = sum(arr)
        return sum_val, arr

    def _prepare_remotes(num: int):
        global last_job_idx
        job_list = job_spec[last_job_idx:last_job_idx+num]
        job_list = [(last_job_idx + idx, job) for idx, job in enumerate(job_list)]
        last_job_idx += len(job_list)
        return job_list

    def _create_remotes(job_list: typing.List[typing.Tuple[int, typing.List[float]]]):
        remotes = []
        for job in job_list:
            logger.info(f"Queuing job {job[0]}")
            job_ref = ray.put(job)
            remotes.append(_do_job.remote(job_ref))
        return remotes
    
    def _transform_output(results):
        global total_sum, total_sum_serial
        for sum_val, arr in results:
            total_sum += sum_val
            total_sum_serial += sum(arr)
        del results # This is important to free up memory
    parallel = 30
    RayUtils.init_ray(num_of_cpus=parallel)
    RayUtils.ray_run_within_parallel_limits(parallel, size, _transform_output, _prepare_remotes, _create_remotes, logger=logger)
    assert total_sum == total_sum_serial, f"total_sum: {total_sum}, total_sum_serial: {total_sum_serial}"
    logger.info(f"total_sum: {total_sum}, total_sum_serial: {total_sum_serial}")