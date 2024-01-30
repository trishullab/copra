#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import typing
from src.rl.proof_action import ProofAction
from src.tools.coq_context_helper import CoqContextHelper
from src.tools.lean_context_helper import Lean3ContextHelper
from src.tools.isabelle_context_helper import IsabelleContextHelper
from src.tools.coq_executor import CoqExecutor
from src.tools.lean_cmd_executor import Lean3Executor
from src.tools.isabelle_executor import IsabelleExecutor
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from src.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from src.tools.dynamic_isabelle_proof_exec import DynamicProofExecutor as DynamicIsabelleProofExecutor

class ProofExecutorCallback(object):
    def __init__(self,
                project_folder: str,
                file_path: str,
                language: ProofAction.Language = ProofAction.Language.COQ,
                prefix: str = None,
                use_hammer: ProofAction.HammerMode = ProofAction.HammerMode.ALLOW,
                timeout_in_secs: int = 60,
                use_human_readable_proof_context: bool = True,
                suppress_error_log: bool = True,
                search_depth: int = 0,
                logger: logging.Logger = None,
                always_use_retrieval: bool = False):
        self.project_folder = project_folder
        self.file_path = file_path
        self.language = language
        self.use_hammer = use_hammer
        self.timeout_in_secs = timeout_in_secs
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.suppress_error_log = suppress_error_log
        self.search_depth = search_depth
        self.logger = logger
        self.prefix = prefix
        self.always_use_retrieval = always_use_retrieval
        pass

    def get_proof_executor(self) -> typing.Union[DynamicCoqProofExecutor, DynamicLeanProofExecutor, DynamicIsabelleProofExecutor]:
        if self.language == ProofAction.Language.COQ:
            use_hammer = False
            search_exec = CoqExecutor(self.project_folder, self.file_path, use_hammer=use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
            coq_context_helper = CoqContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicCoqProofExecutor(coq_context_helper, self.project_folder, self.file_path, context_type=DynamicCoqProofExecutor.ContextType.BestContext, use_hammer=use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
        elif self.language == ProofAction.Language.LEAN:
            use_hammer = False
            search_exec = Lean3Executor(self.project_folder, self.prefix, self.file_path, use_hammer=use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context, enable_search=self.always_use_retrieval)
            lean_context_helper = Lean3ContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicLeanProofExecutor(lean_context_helper, self.project_folder, self.file_path, context_type=DynamicLeanProofExecutor.ContextType.NoContext, use_hammer=use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
        elif self.language == ProofAction.Language.ISABELLE:
            search_exec = IsabelleExecutor(self.project_folder, self.file_path, use_hammer=self.use_hammer, timeout_in_sec=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
            isabelle_context_helper = IsabelleContextHelper(search_exec, self.search_depth, logger=self.logger)
            return DynamicIsabelleProofExecutor(isabelle_context_helper, self.project_folder, self.file_path, context_type=DynamicIsabelleProofExecutor.ContextType.BestContext, use_hammer=self.use_hammer, timeout_in_seconds=self.timeout_in_secs, suppress_error_log=self.suppress_error_log, use_human_readable_proof_context=self.use_human_readable_proof_context)
        else:
            raise Exception(f"Unknown context type: {self.context_type}")