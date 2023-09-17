#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
from src.tools.coq_context_helper import CoqContextHelper
from src.tools.coq_executor import CoqExecutor
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor

class ProofExecutorCallback(object):
    def __init__(self,
                project_folder: str,
                file_path: str,
                context_type: DynamicProofExecutor.ContextType = DynamicProofExecutor.ContextType.NoContext,
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

    def get_proof_executor(self) -> DynamicProofExecutor:
        search_exec = CoqExecutor(self.project_folder, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log)
        coq_context_helper = CoqContextHelper(search_exec, self.search_depth, logger=self.logger)
        return DynamicProofExecutor(coq_context_helper, self.project_folder, self.file_path, context_type=self.context_type, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log)