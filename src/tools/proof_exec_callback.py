#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import typing
from src.tools.coq_context_helper import CoqContextHelper
from src.tools.lean_context_helper import Lean3ContextHelper
from src.tools.coq_executor import CoqExecutor
from src.tools.lean_executor import Lean3Executor
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from src.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor

class ProofExecutorCallback(object):
    def __init__(self,
                project_folder: str,
                file_path: str,
                context_type: typing.Union[DynamicCoqProofExecutor.ContextType, DynamicLeanProofExecutor.ContextType] = DynamicCoqProofExecutor.ContextType.NoContext,
                use_hammer: bool = False,
                timeout_in_secs: int = 60,
                use_human_readable_proof_context: bool = True,
                suppress_error_log: bool = True,
                search_depth: int = 0,
                logger: logging.Logger = None):
        self.project_folder = project_folder
        self.file_path = file_path
        self.context_type = context_type
        self.use_hammer = use_hammer
        self.timeout_in_secs = timeout_in_secs
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.suppress_error_log = suppress_error_log
        self.search_depth = search_depth
        self.logger = logger
        pass

    def get_proof_executor(self) -> typing.Union[DynamicCoqProofExecutor, DynamicLeanProofExecutor]:
        if isinstance(self.context_type, DynamicCoqProofExecutor.ContextType):
            search_exec = CoqExecutor(self.project_folder, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
            coq_context_helper = CoqContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicCoqProofExecutor(coq_context_helper, self.project_folder, self.file_path, context_type=self.context_type, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
        elif isinstance(self.context_type, DynamicLeanProofExecutor.ContextType):
            search_exec = Lean3Executor(self.project_folder, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
            lean_context_helper = Lean3ContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicLeanProofExecutor(lean_context_helper, self.project_folder, self.file_path, context_type=self.context_type, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
        else:
            raise Exception(f"Unknown context type: {self.context_type}")