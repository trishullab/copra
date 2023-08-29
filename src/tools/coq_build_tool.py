#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import logging
import typing
import argparse
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from src.tools.coq_executor import CoqExecutor
from src.tools.coq_build_spec import CoqBuildSpec
from torch.multiprocessing import Pool, set_start_method
from multiprocessing.pool import AsyncResult
from multiprocessing import Lock
global_locks: typing.Dict[str, Lock] = {
    "train_lock": Lock(), # The lock has to be global because the lock needs to be shared across processes
    "test_lock": Lock(), # The lock has to be global because the lock needs to be shared across processes
}
try:
     set_start_method('spawn')
except RuntimeError:
    pass

@dataclass_json
@dataclass
class CoqRepoBuilder:
    root: str
    compilable_projects: typing.Set[str] = field(default_factory=set)
    compilable_files: typing.List[str] = field(default_factory=list)
    uncompilable_projects: typing.Set[str] = field(default_factory=set)
    uncompilable_files: typing.List[str] = field(default_factory=list)
    train_compilable_files: typing.List[str] = field(default_factory=list)
    train_uncompilable_files: typing.List[str] = field(default_factory=list)
    test_compilable_files: typing.List[str] = field(default_factory=list)
    test_uncompilable_files: typing.List[str] = field(default_factory=list)

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.transform = None
        self.build_specs = None
        self.build_disabled = False
        self.projects_to_exclude = set()
        self.log_error = False
    

    def enable_error_loggging(self):
        self.log_error = True

    def set_logger(self, logger: logging.Logger):
        self.logger = logger
    
    def add_project_to_exclude(self, project_name: str):
        self.projects_to_exclude.add(project_name)
    
    def set_build_spec(self, build_specs: typing.List[CoqBuildSpec]):
        assert isinstance(build_specs, list), "build_specs must be a list"
        assert all([isinstance(spec, CoqBuildSpec) for spec in build_specs]), "build_specs must be a list of CoqBuildSpec"
        self.build_specs = build_specs
    
    def set_transform(self, transform: typing.Callable[[str, CoqExecutor], None]):
        assert callable(transform), "transform must be a callable"
        self.transform = transform
    
    def disable_build(self):
        self.build_disabled = True
    
    def _compile_file(self, project_path: str, file_path: str):
        global global_locks
        t_compiled_files_successfully = True
        if os.path.exists(file_path):
            # Compile the file
            try:
                with CoqExecutor(project_path, file_path, suppress_error_log=self.log_error) as coq_exec:
                    coq_exec.run_to_finish()
                    self.logger.info(f"[PID: {os.getpid()}] Compiled {file_path} successfully")
            except:
                self.logger.warning(f"[PID: {os.getpid()}] Couldn't Compile {file_path}")
                t_compiled_files_successfully = False
                pass
        else:
            self.logger.warning(f"File {file_path} does not exist")
        return t_compiled_files_successfully

    def build_from_spec(self, spec: CoqBuildSpec):
        assert isinstance(spec, CoqBuildSpec), "spec must be a CoqBuildSpec"
        if spec.project_name == '.':
                spec.project_name = os.path.basename(os.path.normpath(self.root))
                project_path = self.root
        else:
            project_path = os.path.join(self.root, spec.project_name)
        assert os.path.exists(project_path), f"Project {spec.project_name} does not exist (project_path: {project_path})"
        if not self.build_disabled:
            # Build the project
            build_command = f"cd {project_path}"
            if spec.switch is not None:
                build_command += f" && eval \"$(opam env --set-switch --switch={spec.switch})\""
            if spec.build_command is not None:
                build_command += f" && {spec.build_command}"
            else:
                build_command += f" && make"
            self.logger.info(f"[START] Build Attempt for project {spec.project_name} with command {build_command}")
            try:
                error_code = os.system(build_command)
                if error_code != 0:
                    self.uncompilable_projects.add(spec.project_name)
                    self.logger.warning(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX>ERROR building project {spec.project_name} with command {build_command}<XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                else:
                    self.compilable_projects.add(spec.project_name)
                    self.logger.info(f"==============================>Successfully compiled project {spec.project_name}<==============================")
            except:
                self.logger.error(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX>FAILED to compile project {spec.project_name}<XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        else:
            self.logger.info(f"Build is disabled for all projects.")
        
        # Compile the files in the project parallelly
        compiled_files_successfully = True
        results: typing.List[AsyncResult] = []
        self.logger.info(f"Compiling files for project \"{spec.project_name}\" parallelly")
        with Pool() as pool:
            train_results: typing.List[AsyncResult] = []
            test_results: typing.List[AsyncResult] = []
            for file in spec.train_files:
                file = file.lstrip('./')
                file_path = os.path.join(project_path, file)
                res = pool.apply_async(self._compile_file, 
                    (project_path, file_path))
                results.append(res)
                train_results.append(res)
                # if not _compile_file(file_path, self.train_compilable_files, self.train_uncompilable_files):
                #     compiled_files_successfully = False
            for file in spec.test_files:
                file = file.lstrip('./')
                file_path = os.path.join(project_path, file)
                res = pool.apply_async(self._compile_file, 
                    (project_path, file_path))
                results.append(res)
                test_results.append(res)
                # if not _compile_file(file_path, self.test_compilable_files, self.test_uncompilable_files):
                #     compiled_files_successfully = False
            for result in results:
                result.wait()
            compiled_files_successfully = all([result.get() for result in results])
            for result, file in zip(train_results, spec.train_files):
                file = file.lstrip('./')
                file_path = os.path.join(project_path, file)
                if result.get():
                    self.train_compilable_files.append(file_path)
                    self.compilable_files.append(file_path)
                else:
                    self.train_uncompilable_files.append(file_path)
                    self.uncompilable_files.append(file_path)
            for result, file in zip(test_results, spec.test_files):
                file = file.lstrip('./')
                file_path = os.path.join(project_path, file)
                if result.get():
                    self.test_compilable_files.append(file_path)
                    self.compilable_files.append(file_path)
                else:
                    self.test_uncompilable_files.append(file_path)
                    self.uncompilable_files.append(file_path)
        if not self.build_disabled:
            self.logger.info(f"[END] Build Attempt for project {spec.project_name} with command {build_command}")
        else:
            if compiled_files_successfully:
                self.compilable_projects.add(spec.project_name)
            else:
                self.uncompilable_projects.add(spec.project_name)

    def build_all(self):
        assert self.build_specs is not None, "build_specs is not set. Please use set_build_spec to set it"
        self.logger.info(f"Starting to build all projects...")
        for spec in self.build_specs:
            if spec.project_name not in self.projects_to_exclude:
                self.build_from_spec(spec)
            else:
                self.logger.warning(f"Skipping project {spec.project_name} since it is in the exclusion list")
        self.logger.info(f"Compilable Projects: {self.compilable_projects}")
        self.logger.info(f"Uncompilable Projects: {self.uncompilable_projects}")

    def serialize(self):
        return self.to_json(indent=4)
    
    def run_transform_over_compilable_files(self, use_human_readable: bool = False):
        assert self.transform is not None, "transform is not set. Please use set_transform to set it"
        assert callable(self.transform), "transform must be a callable"
        for project in sorted(self.compilable_projects):
            self.logger.info(f"==============================>Running transform over project {project}<==============================")
            project_path = os.path.join(self.root, project)
            # Use os.walk with sorted to ensure that the files are processed in the same order
            for root, _, files in sorted(os.walk(project_path)):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    if file_path in self.compilable_files:
                        try:
                            with CoqExecutor(project_path, file_path, use_human_readable_proof_context=use_human_readable) as coq_exec:
                                project_id = project_path.replace('/', '.')
                                self.transform(project_id, coq_exec)
                        except:
                            self.logger.error(f"Got an exception while running transform over {file}")
                            self.logger.exception(f"Exception Log")
                            pass
    
    def run_local_transform(self, project_path: str, file_path: str, use_human_readable: bool, transform: typing.Callable[[str, CoqExecutor, typing.Callable[[], CoqExecutor]], None]):
        try:
            def _print_coq_callback():
                t_coq_exec = CoqExecutor(project_path, file_path, suppress_error_log=self.log_error, use_human_readable_proof_context=use_human_readable)
                t_coq_exec.__enter__()
                return t_coq_exec
            self.logger.info(f"==============================>Running transform over file {file_path}<==============================")
            with CoqExecutor(project_path, file_path, suppress_error_log=self.log_error, use_human_readable_proof_context=use_human_readable) as coq_exec:
                project_id = project_path.replace('/', '.')
                transform(project_id, coq_exec, _print_coq_callback)
            self.logger.info(f"==============================>Successfully ran transform over file {file_path}<==============================")
        except:
            self.logger.warning(f"XXXXXXXXXXXXXXXXXXXXXXX>Failed in running transform over file {file_path}<XXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.logger.error(f"Got an exception while running transform over {file_path}")
            self.logger.exception(f"Exception Log")
            pass

    def run_transform(self, transform: typing.Callable[[str, CoqExecutor, typing.Callable[[], CoqExecutor]], None], projects: typing.List[str], files: typing.List[str], use_human_readable: bool = False):
        assert callable(transform), "transform must be a callable"
        assert projects is not None, "projects should not be None"
        assert files is not None, "files should not be None"
        assert isinstance(projects, list), "projects should be a list"
        assert len(projects) > 0, "projects should not be empty"
        assert isinstance(files, list), "files should be a list"
        assert len(files) > 0, "files should not be empty"
        assert len(projects) <= len(files)

        for project in sorted(projects):
            self.logger.info(f"==============================>Running transform over project {project}<==============================")
            project_path = os.path.join(self.root, project)
            assert os.path.exists(project_path), f"project_path {project_path} does not exist"
            some_files_processed = False
            for file_path in sorted(files):
                if file_path.startswith(project_path):
                    some_files_processed = True
                    
                    try:
                        def _print_coq_callback():
                            t_coq_exec = CoqExecutor(project_path, file_path, suppress_error_log=self.log_error, use_human_readable_proof_context=use_human_readable)
                            t_coq_exec.__enter__()
                            return t_coq_exec
                        self.logger.info(f"==============================>Running transform over file {file_path}<==============================")
                        with CoqExecutor(project_path, file_path, suppress_error_log=self.log_error, use_human_readable_proof_context=use_human_readable) as coq_exec:
                            project_id = project_path.replace('/', '.')
                            transform(project_id, coq_exec, _print_coq_callback)
                        self.logger.info(f"==============================>Successfully ran transform over file {file_path}<==============================")
                    except:
                        self.logger.warning(f"XXXXXXXXXXXXXXXXXXXXXXX>Failed in running transform over file {file_path}<XXXXXXXXXXXXXXXXXXXXXXXXXX")
                        self.logger.error(f"Got an exception while running transform over {file_path}")
                        self.logger.exception(f"Exception Log")
                        pass
            if not some_files_processed:
                self.logger.info(f"==============================>No files processed for project {project}<==============================")
            else:
                self.logger.info(f"==============================>Finished transform over project {project}<==============================")

    def load_from_file(file_path: str):
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return CoqRepoBuilder.schema().loads(json_text)

# nohup python3 src/tools/coq_build_tool.py --root_project data/benchmarks/CompCert --build_spec data/benchmarks/compcert_projs_splits.json --info_file data/benchmarks/compcert_projs_build.log.json --option build &
if __name__ == "__main__":
    import time
    os.chdir(root_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logging_dir = f".log/build_logs"
    try:
        os.mkdir(logging_dir)
    except FileExistsError:
        pass
    log_file = f"{os.path.join(logging_dir, f'coq_build_tool_{current_time}.log')}"
    with open(log_file, "w") as f:
        f.write("") # Clear the file
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_project", type=str, default="data/test/coq/custom_group_theory", help="The root folder of the coq project")
    parser.add_argument("--build_spec", type=str, default="data/test/coq/custom_group_theory/grp_theory_splits.json", help="The file which has the build spec")
    parser.add_argument("--info_file", type=str, default="data/test/coq/custom_group_theory/compilable_projects_info.log.json", help="The file which will have the info about the compilable projects")
    parser.add_argument("--option", choices=["build", "load"], default="build", help="The option to run. Either build or load")
    parser.add_argument("--projects_to_exclude", type=str, default=None, required=False)
    parser.add_argument("--only_compile_files", type=bool, default=False, required=False)
    args = parser.parse_args()
    root_project = args.root_project
    info_file = args.info_file
    option = args.option
    if option == "build":
        builder = CoqRepoBuilder(root_project)
        # Create build spec
        with open(args.build_spec, 'r') as f:
            build_specs = CoqBuildSpec.schema().loads(f.read(), many=True)
        builder.set_build_spec(build_specs)
        if args.projects_to_exclude is not None:
            projects_to_exclude = args.projects_to_exclude.split(',')
            for project in projects_to_exclude:
                project = project.strip()
                assert os.path.exists(os.path.join(root_project, project)), f"{project} is not a valid project"
                builder.add_project_to_exclude(project)
        if args.only_compile_files:
            builder.disable_build()
        builder.build_all()
        with open(info_file, 'w') as f:
            f.write(builder.serialize())
    elif option == "load":
        builder = CoqRepoBuilder.load_from_file(info_file)
        logger.info(builder.compilable_projects)
        logger.info(builder.compilable_files)
        logger.info(builder.uncompilable_projects)
        logger.info(builder.uncompilable_files)
    else:
        raise Exception("Invalid option")