#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import ray
import logging
import typing
import shutil
import psutil
import gc
from src.tools.ray_utils import RayUtils
from src.tools.training_data import TrainingData
from src.tools.coq_build_tool import CoqRepoBuilder
from src.tools.coq_executor import CoqExecutor
from src.tools.coq_local_data_generation_transform import LocalDataGenerationTransform
from src.tools.coq_training_data_generator import GenericTrainingDataGenerationTransform, TrainingDataGenerationType

class RunDataGenerationTransforms(object):
    def __init__(self, transforms: typing.List[GenericTrainingDataGenerationTransform], logging_dir: str, save_intermidiat_transforms: bool = True, logger: logging.Logger = None):
        assert transforms is not None, "transforms should not be None"
        assert isinstance(transforms, list), "transforms should be a list"
        assert len(transforms) > 0, "transforms should not be empty"
        assert all(isinstance(transform, GenericTrainingDataGenerationTransform) for transform in transforms), "transforms should be a list of GenericTrainingDataGenerationTransform"
        assert logging_dir is not None, "logging_dir should not be None"
        assert os.path.isdir(logging_dir), "logging_dir should be a directory"
        self.logging_dir = logging_dir
        self.transforms = transforms
        self.save_intermidiate_transforms = save_intermidiat_transforms
        self.logger = logger if logger is not None else logging.getLogger("DataGenerationTransforms")
        pass

    @staticmethod
    def _get_transform_name(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType]) -> str:
        name = ""
        if isinstance(transform, GenericTrainingDataGenerationTransform):
            name = transform.name
        elif isinstance(transform, TrainingDataGenerationType):
            name = transform.name.lower()
        else:
            raise Exception("Unknown transform type")
        return name

    @staticmethod
    def get_meta_file_name(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType]) -> str:
        return f"{RunDataGenerationTransforms._get_transform_name(transform)}.meta.json"
    
    @staticmethod
    def get_data_file_name(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType], file_name_suffix: int = 0) -> str:
        name = RunDataGenerationTransforms._get_transform_name(transform)
        return f"{name}_data_{file_name_suffix:010d}.json"
    
    @staticmethod
    def get_data_filename_prefix(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType]) -> str:
        name = RunDataGenerationTransforms._get_transform_name(transform)
        return f"{name}_data_"
    
    @staticmethod
    def get_data_filename_suffix(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType]) -> int:
        return ".json"
    
    @staticmethod
    def get_lemma_ref_filename_prefix(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType]) -> str:
        name = RunDataGenerationTransforms._get_transform_name(transform)
        return f"{name}_lemma_"
    
    @staticmethod
    def get_lemma_ref_filename_suffix(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType]) -> int:
        return ".json"
    
    @staticmethod
    def is_transform_data_file(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType], file_name: str) -> bool:
        name = RunDataGenerationTransforms._get_transform_name(transform)
        return file_name.startswith(f"{name}_data_") and file_name.endswith(".json")

    @staticmethod
    def is_transform_meta_file(transform: typing.Union[GenericTrainingDataGenerationTransform, TrainingDataGenerationType], file_name: str) -> bool:
        name = RunDataGenerationTransforms._get_transform_name(transform)
        return file_name.startswith(name) and file_name.endswith(".meta.json")

    @staticmethod
    def call_local_transform(
        logger: logging.Logger,
        transform,
        output_dir,
        project_path,
        file_path,
        log_error,
        use_human_readable) -> typing.Any:
        if not isinstance(transform, GenericTrainingDataGenerationTransform):
            raise Exception("transform should be a GenericTrainingDataGenerationTransform")
        def _print_coq_callback():
            search_coq_exec = CoqExecutor(project_path, file_path, use_human_readable_proof_context=use_human_readable, suppress_error_log=log_error)
            search_coq_exec.__enter__()
            return search_coq_exec
        if isinstance(transform, LocalDataGenerationTransform):
            with CoqExecutor(project_path, file_path, use_human_readable_proof_context=True, suppress_error_log=True) as coq_exec:
                project_id = project_path.replace('/', '.')
                metadata = transform.get_meta_object()
                metadata.training_data_buffer_size = transform.buffer_size
                metadata.data_filename_prefix = RunDataGenerationTransforms.get_data_filename_prefix(transform)
                metadata.data_filename_suffix = RunDataGenerationTransforms.get_data_filename_suffix(transform)
                metadata.lemma_ref_filename_prefix = RunDataGenerationTransforms.get_lemma_ref_filename_prefix(transform)
                metadata.lemma_ref_filename_suffix = RunDataGenerationTransforms.get_lemma_ref_filename_suffix(transform)
                training_data = TrainingData(
                    output_dir,
                    RunDataGenerationTransforms.get_meta_file_name(transform),
                    metadata,
                    transform.max_parallelism,
                    remove_from_store_after_loading=True,
                    logger=logger)
                transform(training_data, project_id, coq_exec, _print_coq_callback)
        else:
            raise Exception("Unknown transform")
        return training_data

    # @ray.remote(max_retries=-1)
    # def _save_training_data(storename: str, training_data: TrainingData):
    #     start_time = time.time()
    #     ray.logger.info(f"Saving training data to {training_data.folder}")
    #     save_res = training_data.save()
    #     ray.logger.info(f"Saved training data to {training_data.folder} in {time.time() - start_time} seconds")
    #     return save_res

    @ray.remote(max_retries=-1)
    def run_local_transform_on_file(idx, log_file: str, output_dir: str, project_path: str, file_path: str, use_human_readable: bool, transform: GenericTrainingDataGenerationTransform, log_error: bool, save_transform: bool = True):
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("FullTransform")
        logger.info(f"Process ID: {os.getpid()}")
        transform.logger = logger
        meta_file_name = RunDataGenerationTransforms.get_meta_file_name(transform)
        if os.path.exists(os.path.join(output_dir, meta_file_name)):
            logger.info(f"[{transform.name}] Skipping transform for file {file_path} as it is already present")
            return None
        else:
            logger.info(f"==============================>[{transform.name}] Running transform over file {file_path}<==============================")
            try:
                training_data : TrainingData = RunDataGenerationTransforms.call_local_transform(logger, transform, output_dir, project_path, file_path, log_error, use_human_readable)
                logger.info(f"==============================>[{transform.name}] Successfully ran transform over file {file_path}<==============================")
            except:
                logger.warning(f"XXXXXXXXXXXXXXXXXXXXXXX>[{transform.name}] Failed in running transform over file {file_path}<XXXXXXXXXXXXXXXXXXXXXXXXXX")
                logger.error(f"Got an exception while running transform over {file_path}")
                logger.exception(f"Exception Log")
                pass
            return idx, training_data

    def merge_local_transforms(self,
                final_training_data: TrainingData,
                tds: typing.List[TrainingData],
                transform: LocalDataGenerationTransform):
        self.logger.info(f"==============================>[{transform.name}] Merging local transforms for all projects<==============================")
        process = psutil.Process()
        for idx in range(len(tds)):
            if tds[idx] is None:
                continue
            training_data = tds[idx]
            self.logger.info(f"[Process Id = {process.pid}], Memory used (Before GC): {process.memory_info().rss/2**30} GiB")
            folder = training_data.folder
            self.logger.info(f"==============================>[{transform.name}] Merging local transforms for project {folder}<==============================")
            final_training_data.merge(training_data)
            tds[idx] = None # free up memory
            del training_data # free up memory
            training_data = None # free up memory
            self.logger.info(f"==============================>[{transform.name}] Merged local transforms for project {folder}<==============================")
            gc.collect()
            self.logger.info(f"[Process Id = {process.pid}], Memory used (After GC): {process.memory_info().rss/2**30} GiB")
            idx += 1
        self.logger.info(f"==============================>[{transform.name}] Merged local transforms for all projects<==============================")

    def run_local_transform(self, pool_size: int , transform: LocalDataGenerationTransform, root: str, projects: typing.List[str], files: typing.List[str], use_human_readable: bool, new_output_dir: str, log_error: bool, save_transform: bool = True, preserve_temp: bool = True):
        assert root is not None, "builder should not be None"
        assert isinstance(root, str), "builder should be a string"
        assert pool_size > 0, "pool_size should be greater than 0"
        assert transform is not None, "transform should not be None"
        assert projects is not None, "projects should not be None"
        assert files is not None, "files should not be None"
        assert isinstance(projects, list), "projects should be a list"
        assert len(projects) > 0, "projects should not be empty"
        assert isinstance(files, list), "files should be a list"
        assert len(files) > 0, "files should not be empty"
        assert len(projects) <= len(files)
        temp_output_dir = os.path.join(new_output_dir, f"temp_{transform.name}")
        os.makedirs(temp_output_dir, exist_ok=True)
        temporary_files_found: typing.List[str] = []
        object_store_memory_in_gb = 100
        memory_in_gb = 0.25
        ray_dashboard = RayUtils.init_ray(num_of_cpus=pool_size, object_store_memory_in_gb=object_store_memory_in_gb)
        self.logger.info(f"==============================>[{transform.name}] Ray initialized with {transform.max_parallelism} CPUs, Memory=({memory_in_gb} GiB, Object Memory = {object_store_memory_in_gb} GiB)<==============================")
        self.logger.info(f"Ray Context:\n {ray_dashboard}")
        job_spec = []
        job_idx = 0
        for project in sorted(projects):
            # Create temporary directory for each project
            temp_project_dir = os.path.join(temp_output_dir, project)
            os.makedirs(temp_project_dir, exist_ok=True)
            self.logger.info(f"==============================>[{transform.name}] Discovering transform jobs over project {project}<==============================")
            project_path = os.path.join(root, project)
            assert os.path.exists(project_path), f"project_path {project_path} does not exist"
            some_files_processed = False
            for file_path in sorted(files):
                if file_path.startswith(project_path):
                    some_files_processed = True
                    # Create temporary directory for each file
                    relative_file_path = os.path.relpath(file_path, project_path)
                    relative_file_path = relative_file_path.replace("/", ".").replace(".v", "")
                    temp_file_dir = os.path.join(temp_project_dir, relative_file_path)
                    os.makedirs(temp_file_dir, exist_ok=True)
                    log_file = os.path.join(self.logging_dir, f"{relative_file_path}.log")
                    job_spec.append((job_idx, log_file, temp_file_dir, project_path, file_path, use_human_readable, transform, log_error, save_transform))
                    temporary_files_found.append(temp_file_dir)
                    job_idx += 1
            if not some_files_processed:
                self.logger.info(f"==============================>[{transform.name}] No files processed for project {project}<==============================")
            else:
                self.logger.info(f"==============================>[{transform.name}] Finished discovering transform jobs over project {project}<==============================")

        final_training_meta = transform.get_meta_object()
        final_training_meta.training_data_buffer_size = transform.buffer_size
        final_training_meta.data_filename_prefix = RunDataGenerationTransforms.get_data_filename_prefix(transform)
        final_training_meta.data_filename_suffix = RunDataGenerationTransforms.get_data_filename_suffix(transform)
        final_training_meta.lemma_ref_filename_prefix = RunDataGenerationTransforms.get_lemma_ref_filename_prefix(transform)
        final_training_meta.lemma_ref_filename_suffix = RunDataGenerationTransforms.get_lemma_ref_filename_suffix(transform)
        final_training_data = TrainingData(
            new_output_dir,
            RunDataGenerationTransforms.get_meta_file_name(transform),
            final_training_meta,
            transform.max_parallelism,
            remove_from_store_after_loading=True,
            logger=self.logger)
        last_job_idx = 0
        tds = [None]*len(job_spec)
        def _create_remotes(job_list):
            remotes = []
            for job in job_list:
                self.logger.info(f"[{transform.name}] Starting transform for {job[5]}")
                remotes.append(RunDataGenerationTransforms.run_local_transform_on_file.remote(*job))
            return remotes
        
        def _prepare_remotes(num: int):
            nonlocal last_job_idx
            job_list = job_spec[last_job_idx:last_job_idx+num]
            last_job_idx += len(job_list)
            return job_list

        def _transform_output(results):
            for idx, training_data in results:
                self.logger.info(f"[{transform.name}] Transform finished for [{idx}] {job_spec[idx]}")
                tds[idx] = training_data
            process = psutil.Process()
            self.logger.info(f"[{transform.name}] Process Id = {process.pid}, Memory used: {process.memory_info().rss/2**30} GiB")
        
        RayUtils.ray_run_within_parallel_limits(pool_size, len(job_spec), _transform_output, _prepare_remotes, _create_remotes, logger=self.logger)

        # Merge all the files into one
        self.merge_local_transforms(final_training_data, tds, transform)

        self.logger.info(f"==============================>[{transform.name}] Saving Final Transform over file {final_training_data.folder}<==============================")
        final_training_data.save()
        final_training_data_details = final_training_data.meta.to_json(indent=4)
        self.logger.info(f"Final Transform details:\n{final_training_data_details}")
        self.logger.info(f"==============================>[{transform.name}] Final Transform saved<==============================")

        # if preserve_temp:
        #     self.logger.info(f"==============================>[{transform.name}] Temporary files preserved at {temp_output_dir}<==============================")
        #     last_save_idx = 0
        #     def _create_save_remotes(idxs):
        #         remotes = []
        #         for idx in idxs:
        #             self.logger.info(f"[{transform.name}] Saving transform for {job_spec[idx][5]}")
        #             remotes.append(RunDataGenerationTransforms._save_training_data.remote(self.storename, idx))
        #         return remotes

        #     def _prepare_save_remotes(num: int):
        #         nonlocal last_save_idx
        #         idxs = list(range(last_save_idx, max(last_save_idx + num, len(job_spec))))
        #         last_save_idx += len(idxs)
        #         return idxs
            
        #     # def _transform_save_output(remote):
        #     #     res = ray.get(remote)
        #     #     self.logger.info(f"[{transform.name}] Transform saved for [{res}]")

        #     def _transform_save_output(results):
        #         for res in results:
        #             self.logger.info(f"[{transform.name}] Transform saved for [{res}]")
            
        #     RayUtils.ray_run_within_parallel_limits(pool_size, len(job_spec), _transform_save_output, _prepare_save_remotes, _create_save_remotes)
        # else:
        self.logger.warning(f"==============================>[{transform.name}] Removing temp directory {temp_output_dir}<==============================")
        shutil.rmtree(temp_output_dir)

    def run_all_local_transforms(self, pool_size: int, root: str, projects: typing.List[str], files: typing.List[str], use_human_readable: bool, new_output_dir: str, log_error: bool):
        for idx, transform in enumerate(self.transforms):
            last_transform = idx == len(self.transforms) - 1
            save_transform = self.save_intermidiate_transforms or last_transform
            self.run_local_transform(pool_size, transform, root, projects, files, use_human_readable, new_output_dir, log_error, save_transform, preserve_temp=self.save_intermidiate_transforms)
        pass

# nohup python3 src/tools/run_data_generation_transforms.py --use_human_readable --buffer_size 2500 --pool_size 20 --transform_type LOCAL --dep_depth 0 --output_dir .log/run_data_generation_transforms/data/benchmarks/CompCert --info_file data/benchmarks/compcert_projs_build.log.json > .log/run_data_generation_transforms/local_transform.log 2>&1 &

# nohup python3 src/tools/run_data_generation_transforms.py --use_human_readable --save_intermidiate_transforms --buffer_size 10000 --pool_size 15 --transform_type FULL --dep_depth 0 --output_dir data/generated/compcert/full --info_file data/compilable_projects_compcert_build.log.json > .temp_cache/logs/training_data_filter/transforms/generated_compcert_full.log 2>&1 &
# nohup python3 src/tools/run_data_generation_transforms.py --use_human_readable --save_intermidiate_transforms --buffer_size 750 --pool_size 20 --transform_type FULL --dep_depth 0 --output_dir data/generated/compcert/full --info_file data/compilable_projects_compcert_build.log.json > .temp_cache/logs/training_data_filter/transforms/generated_compcert_full.log 2>&1 &
# nohup python3 src/tools/run_data_generation_transforms.py --use_human_readable --save_intermidiate_transforms --buffer_size 500 --pool_size 30 --transform_type FULL --dep_depth 0 --output_dir data/generated/compcert/full --info_file data/compilable_projects_compcert_build.log.json > .temp_cache/logs/training_data_filter/transforms/generated_compcert_full.log 2>&1 &
# nohup python3 src/tools/run_data_generation_transforms.py --use_human_readable --buffer_size 500 --pool_size 15 --transform_type FULL --dep_depth 0 --output_dir data/generated/compcert/full --info_file data/compilable_projects_compcert_build.log.json > .temp_cache/logs/training_data_filter/transforms/generated_compcert_full.log 2>&1 &
# nohup python3 src/tools/run_data_generation_transforms.py --pool_size 15 --buffer_size 500 --transform_type BM25 --output_dir data/generated/compcert/bm25 --info_file data/generated/compcert/full > .temp_cache/logs/training_data_filter/generated_compcert_bm25.log 2>&1 &
# nohup python3 src/tools/run_data_generation_transforms.py --pool_size 3 --buffer_size 50 --transform_type BM25 --output_dir data/generated_very_small_test/bm25 --info_file data/generated_very_small_test/full > .temp_cache/logs/training_data_filter/generated_test_bm25.log 2>&1 &
if __name__ == "__main__":
    import time
    import argparse
    os.chdir(root_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logging_dir = f".log/run_data_generation_transforms/logs/{current_time}"
    os.makedirs(logging_dir, exist_ok=True)
    log_file = f"{os.path.join(logging_dir, f'_coq_data_gen.log')}"
    with open(log_file, "w") as f:
        f.write("") # Clear the file
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("DataGenerationTransforms")
    logger.info(f"Process ID: {os.getpid()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_readable", action="store_true", help="Whether to use human readable proof context")
    parser.add_argument("--save_intermidiate_transforms", action="store_true", help="Whether to save the intermidiate transforms")
    parser.add_argument("--buffer_size", type=int, default=50, help="The buffer size to use for the training data generation")
    parser.add_argument("--pool_size", type=int, default=20, help="The number of processes to use for running the transforms")
    parser.add_argument("--transform_type", type=str, choices=["LOCAL", "FULL", "BM25"], default="LOCAL", help="The type of transform to run")
    parser.add_argument("--dep_depth", type=int, default=0, help="The depth of the dependency tree to use for the training data generation")
    parser.add_argument("--max_search_results", type=int, default=None, help="The maximum number of search results to use for the training data generation")
    parser.add_argument("--output_dir", type=str, default=".log/run_data_generation_transforms/data/test/custom_group_theory", help="The root folder where the training data will be generated")
    parser.add_argument("--info_file", type=str, default="data/test/coq/custom_group_theory/compilable_projects_info.log.json", help="The file which will have the info about the compilable projects")
    args = parser.parse_args()
    use_human_readable = args.use_human_readable
    buffer_size = args.buffer_size
    pool_size = args.pool_size
    output_dir = args.output_dir
    info_file = args.info_file
    dep_depth = args.dep_depth
    max_search_results = args.max_search_results
    transform_type = args.transform_type
    save_intermidiate_transforms = args.save_intermidiate_transforms
    max_parallelism = pool_size
    k1 = 1.2
    b = 0.75
    epsilon = 0.2
    os.makedirs(output_dir, exist_ok=True)

    try:
        transforms = []
        if transform_type == "LOCAL":
            transform = LocalDataGenerationTransform(dep_depth, max_search_results=max_search_results, buffer_size=buffer_size, logger=logger)
            builder : CoqRepoBuilder = CoqRepoBuilder.load_from_file(info_file)
            builder.set_logger(logging.getLogger("CoqRepoBuilder"))
            builder.enable_error_loggging()
            transforms.append(transform)
        else:
            raise ValueError(f"Unexpected transform_type: {transform_type}")
        data_transform = RunDataGenerationTransforms(transforms, logging_dir, save_intermidiat_transforms=len(transforms) > 1 or save_intermidiate_transforms, logger=logger)
        for data_type in ["train", "test"]:
            new_output_dir = os.path.join(output_dir, data_type)
            os.makedirs(new_output_dir, exist_ok=True)
            if transform_type == "LOCAL":
                projects = list(builder.compilable_projects)
            else:
                projects = []
            files = []
            if data_type == "train":
                if transform_type == "LOCAL":
                    files = builder.train_compilable_files + builder.train_uncompilable_files
                else:
                    raise ValueError(f"Unexpected transform_type: {transform_type}")
            elif data_type == "test":
                if transform_type == "LOCAL":
                    files = builder.test_compilable_files + builder.test_uncompilable_files
                else:
                    raise ValueError(f"Unexpected transform_type: {transform_type}")
            else:
                raise ValueError(f"Unexpected data_type: {data_type}")
            if transform_type == "LOCAL":
                if len(files) > 0:
                    data_transform.run_all_local_transforms(pool_size, builder.root, projects, files, use_human_readable=use_human_readable, new_output_dir=new_output_dir, log_error=True)
    except Exception as e:
        logger.exception(e)
        raise e 