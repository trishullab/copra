#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import random
import logging
import re
import time
import json
from src.lean_server.lean_context import ProofContext
from src.lean_server.lean4_utils import Lean4Utils
from src.tools.lean_parse_utils import LeanLineByLineReader
from src.tools.theorem_details import TheoremDetails
from src.lean_server.lean4_repl_interface import ProcessInterface
from typing import Iterator, List, Optional, Tuple, OrderedDict, Generator, Dict

class Lean4SyncExecutor:
    theorem_start_regex = r"[\s]*(theorem|lemma|example)[\s]+"
    # Non tactic mode support removed because of complications in parsing
    # theorem_end_regex = r"(theorem|lemma|example) [\S|\s]*?(:=|\|)[\s]*?"
    # theorem_regex = r"((((theorem|lemma) ([\S]*))|example)([\S|\s]*?)(:=|\|)[\s]*?)[\s]+"
    # We ONLY support proofs which are written in tactic mode i.e. with := syntax
    theorem_endings = r"(:=|(\|[\S|\s]*=>))"
    theorem_end_regex = r"(theorem|lemma|example)([\s|\S]*?)(:=|=>)"
    theorem_regex = r"((((theorem|lemma)[\s]+([\S]*))|example)([\S|\s]*?)(:=|=>)[\s]*?)[\s]+"
    remove_proof_regex = r"([\s|\S]*(:=|\|))[\s|\S]*?"
    proof_context_separator = "⊢"
    proof_context_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
    goal_regex = rf"([\s|\S]*?){proof_context_separator}([\s|\S]*)"
    theorem_match = re.compile(theorem_regex, re.MULTILINE)
    proof_context_match = re.compile(proof_context_regex, re.MULTILINE)
    goal_match = re.compile(goal_regex, re.MULTILINE)
    theorem_start_match = re.compile(theorem_start_regex, re.MULTILINE)
    theorem_end_match = re.compile(theorem_end_regex, re.MULTILINE)
    remove_proof_match = re.compile(remove_proof_regex, re.MULTILINE)
    proof_context_generation_tactic = "\nend"
    proof_context_generation_tactic_curlies = "\n}"
    proof_state_running_message = "tactic failed, there are unsolved goals\nstate:"
    def __init__(self, 
        project_root: Optional[str] = None, 
        prefix: Optional[str] = None, 
        main_file: Optional[str] = None, 
        use_hammer: bool = False, 
        timeout_in_sec: int = 60, 
        use_human_readable_proof_context: bool = True, 
        proof_step_iter: Optional[Iterator[str]] = None, 
        suppress_error_log: bool = False, 
        mathlib_root: Optional[str] = None, 
        enable_search: bool = False, 
        namespaces: Optional[List[str]] = None, 
        keep_local_context: bool = False,
        logger: Optional[logging.Logger] = None):
        assert proof_step_iter is None or isinstance(proof_step_iter, Iterator), \
            "proof_step_iter must be an iterator"
        assert main_file is not None or proof_step_iter is not None, \
            "Either main_file or proof_step_iter must be provided"
        assert main_file is None or proof_step_iter is None, \
            "Only one of main_file or proof_step_iter must be provided"
        assert main_file is None or (os.path.exists(main_file) and main_file.endswith(".lean")), \
            "main_file must be a valid path to a '.lean' file"
        assert project_root is None or (os.path.exists(project_root) and os.path.isdir(project_root)), \
            "project_root must be a valid path to a directory"
        assert not use_hammer, "Hammer is not supported for Lean4"
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.project_root = project_root if project_root is not None else "."
        self.main_file = main_file
        self.ticks = str(time.time()).replace(".", "") # This ensures that the temp file name is unique and doesn't clash with other temp files
        # This helps in running parallel instances of prover
        self.random_num = str(random.randint(0, 100000000))
        self.temp_filename_suffix = f"temptodel{self.ticks}{self.random_num}.lean"
        self.temp_file = os.path.join(prefix, self.temp_filename_suffix) if prefix is not None else self.temp_filename_suffix
        self.temp_file_full_path = os.path.join(self.project_root, self.temp_file)
        self.temp_file_full_path = os.path.abspath(self.temp_file_full_path)
        self.use_hammer = use_hammer
        self.timeout_in_sec = min(timeout_in_sec, 120) # Maximum 120s timeout
        self.current_stmt = None
        self.line_num = 0
        self.main_file_iter = proof_step_iter
        self.suppress_error_log = suppress_error_log
        self.process_interace : ProcessInterface = None
        self.execution_complete = False
        self._max_memory_in_mib = 40000 # 40 GiB is needed for mathlib to work seemlessly
        self._lines_executed = []
        self.proof_context : ProofContext = None
        self.curr_lemma_name : Optional[str] = None
        self.curr_lemma : Optional[str] = None
        self.lean_error_messages : List[str] = []
        self._proof_running = False
        self._file_content = ""
        self.local_file_lemmas: OrderedDict[str, str] = OrderedDict()
        self.local_theorem_lemma_description: OrderedDict[str, str] = OrderedDict()
        self._proof_start_idx: Optional[int] = None
        self._import_end_idx: Optional[int] = None
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.use_file = False
        if mathlib_root is not None:
            self._mathlib_root = mathlib_root
        else:
            self._mathlib_root = os.path.join(self.project_root, "_target", "deps", "mathlib")
        self._mathlib_src_root = os.path.join(self._mathlib_root, "src")
        self._enable_search = enable_search
        self._theorem_started = False
        self._content_till_last_theorem_stmt = None
        self._last_theorem = None
        self._last_env_idx = None
        self._last_proof_state_idx = None
        self._line_to_env_idx_map = {}
        self._line_to_proof_state_idx_map = {}
        self._anon_theorem_count = 0
        self._namespaces = []
        self._last_file_seek = 0
        self._line_num_seek_map = {}
        self._file_handle = None
        if self._enable_search:
            pass
        pass

    def __enter__(self):
        tools_dir = os.path.dirname(__file__)
        repl_path = os.path.join(tools_dir, "repl")
        abs_path = os.path.abspath(repl_path)
        path_to_repl_exec = os.path.join(abs_path, ".lake", "build", "bin", "repl")
        if 'Mathlib' in self.project_root:
            self.use_file = True
        assert os.path.exists(path_to_repl_exec), f"Lean4 repl executable does not exist at {path_to_repl_exec}, you may need to build it"
        self.process_interace = ProcessInterface(
            command=f"lake env {path_to_repl_exec}",
            cwd=self.project_root,
            logger=self.logger,
            log_level=logging.INFO)
        if self.main_file_iter is None:
            self.main_file_iter = LeanLineByLineReader(self.main_file, remove_comments=True, no_strip=True).instruction_step_generator()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.process_interace is not None, "ProcessInterface is not initialized"
        self.process_interace.close()
        try:
            self.main_file_iter.close() # Close the file handle
        except:
            pass
        # delete if the main file is a temporary file
        if self._file_handle is not None:
            self._file_handle.close()
        if os.path.exists(self.temp_file_full_path):
            os.remove(self.temp_file_full_path)

    def is_in_proof_mode(self):
        return True if self.proof_context else False

    def run_next(self) -> bool:
        try:
            stmt = next(self.main_file_iter)
        except StopIteration:
            self.execution_complete = True
            return False
        self.current_stmt = stmt
        self.line_num += 1
        try:
            idx = len(self._lines_executed)
            self._run_stmt_on_lean_server(idx, stmt)
        except:
            if not self.suppress_error_log:
                self.logger.error(f"Got an exception while running '{stmt}' on lean. File name: {self.main_file}")
                self.logger.exception(f"Exception Log")
            raise
        self._lines_executed.append(stmt)
        return True

    def needs_qed(self):
        return self.proof_context is not None and len(self.proof_context.all_goals) == 0
    
    def needs_cut_close(self):
        return self.proof_context is not None and len(self.proof_context.fg_goals) == 0 and len(self.proof_context.all_goals) > 0

    def run_next_without_exec(self) -> bool:
        raise NotImplementedError

    def run_all_without_exec(self) -> bool:
        raise NotImplementedError

    def find_all_theorems_names(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def get_tokens_in_given_stmt(self, stmt: str, ignore_first_token: bool = False) -> Iterator[str]:
        raise NotImplementedError

    def tokenize(self, stmt: str) -> Iterator[str]:
        raise NotImplementedError

    def search_type_matching_defns(self, name: str) -> List:
        raise NotImplementedError

    def get_all_type_matching_defns(self, name: str) -> Iterator:
        raise NotImplementedError

    def search_exact(self, name: str) -> List:
        raise NotImplementedError

    def search_defn(self, name: str, match_until: Tuple[str], max_search_res: Optional[int] = None) -> List[Tuple[str, str, bool]]:
        raise NotImplementedError

    def run_without_executing(self, stmt: str):
        while True:
            try:
                stmt = next(self.main_file_iter)
            except StopIteration:
                return
            idx = len(self._lines_executed)
            self._set_content_to_run(stmt)
            if stmt.startswith("theorem") and self._import_end_idx is None:
                self._import_end_idx = idx
            self.current_stmt = stmt
            self.line_num += 1
            self._set_content_to_run(stmt)
            self._lines_executed.append(stmt)

    def run_lemma_without_executing(self):
        while True:
            try:
                stmt = next(self.main_file_iter)
                self.current_stmt = stmt
                self.line_num += 1
                if "Qed." in stmt or "Defined." in stmt or "Admitted." in stmt:
                    return True
            except StopIteration:
                return False

    def run_till_next_lemma(self) -> Tuple[bool, Optional[str]]:
        # Run the coq file until the next lemma is found
        next_stmt = None
        in_proof_mode = self.is_in_proof_mode()
        if in_proof_mode or self.execution_complete:
            # If we are already in proof mode, then we have already found a lemma
            return False, next_stmt
        prev_stmt = self.current_stmt
        ran_last_cmd = self.run_next()
        next_stmt = self.current_stmt
        if not ran_last_cmd:
            return False, None
        assigned = False
        while ran_last_cmd and not in_proof_mode:
            if not assigned:
                prev_stmt = next_stmt
            ran_last_cmd = self.run_next()
            in_proof_mode = self.is_in_proof_mode()
            if not assigned:
                next_stmt = self.current_stmt
                if in_proof_mode:
                    assigned = True
        lemma_name = next_stmt if next_stmt.startswith("Theorem") or next_stmt.startswith("Lemma") else prev_stmt
        return in_proof_mode, lemma_name

    def run_till_next_lemma_return_exec_stmt(self) -> Generator[str, None, None]:
        # Run the coq file until the next lemma is found
        next_stmt = None
        in_proof_mode = self.is_in_proof_mode()
        if in_proof_mode or self.execution_complete:
            # If we are already in proof mode, then we have already found a lemma
            yield from []
        else:
            ran_last_cmd = self.run_next()
            next_stmt = self.current_stmt
            if not ran_last_cmd:
                yield from []
            else:
                yield next_stmt
            while ran_last_cmd and not in_proof_mode:
                ran_last_cmd = self.run_next()
                next_stmt = self.current_stmt
                if ran_last_cmd:
                    yield next_stmt
                in_proof_mode = self.is_in_proof_mode()

    def run_to_finish_lemma_return_exec(self) -> Generator[str, None, None]:
        # Run the coq file until the next lemma is found
        next_stmt = None
        in_proof_mode = self.is_in_proof_mode()
        if not in_proof_mode or self.execution_complete:
            # If we are already in proof mode, then we have already found a lemma
            yield from []
        else:
            ran_last_cmd = self.run_next()
            next_stmt = self.current_stmt
            if not ran_last_cmd:
                yield from []
            else:
                yield next_stmt
            while ran_last_cmd and in_proof_mode:
                ran_last_cmd = self.run_next()
                next_stmt = self.current_stmt
                if ran_last_cmd:
                    yield next_stmt
                in_proof_mode = self.is_in_proof_mode()

    def run_to_finish_lemma(self) -> bool:
        # Run the coq file and finish the current lemma
        in_proof_mode = self.is_in_proof_mode()
        if not in_proof_mode or self.execution_complete:
            # If we are not in proof mode, then we are not finishing a lemma
            return False
        ran_last_cmd = self.run_next()
        if not ran_last_cmd:
            return False
        while ran_last_cmd and in_proof_mode:
            ran_last_cmd = self.run_next()
            in_proof_mode = self.is_in_proof_mode()
        return not in_proof_mode

    def run_till_line_num(self, line_num: int):
        assert line_num >= self.line_num
        ran_last_cmd = True
        while ran_last_cmd and self.line_num < line_num:
            ran_last_cmd = self.run_next()
        return self.line_num
    
    def run_to_finish(self):
        ran_last_cmd = True
        while ran_last_cmd:
            ran_last_cmd = self.run_next()
        
    def get_lemma_name_if_running(self) -> Optional[str]:
        if not self.is_in_proof_mode():
            return None
        else:
            try:
                return self.curr_lemma_name
            except:
                return None
    
    def get_lemma_stmt_if_running(self) -> Optional[str]:
        if not self.is_in_proof_mode():
            return None
        else:
            try:
                return self.local_theorem_lemma_description[self.curr_lemma_name]
            except:
                return None
        
    def get_current_lemma_name(self) -> Optional[str]:
        if self.curr_lemma_name is None:
            return None
        else:
            return self.curr_lemma_name
    
    def _parse_theorem_stmt(self, stmt: str) -> str:
        matches = list(Lean4SyncExecutor.theorem_match.finditer(stmt))
        if len(matches) == 0:
            return None, None, None
        # if len(matches) == 0:
        #     raise ValueError(f"Could not find the theorem in the statement: {stmt}")
        # We are only interested in the last theorem
        match = matches[-1]
        _, span_end = match.span()
        full_stmt = match.group(1)
        theorem_name = match.group(5)
        theorem_stmt = match.group(6)
        thm_end_style = match.group(7)
        if thm_end_style == "=>":
            # Find the last '|' in the full_stmt
            thm_end_idx = full_stmt.rfind('|')
            if thm_end_idx == -1:
                remaining_stmt = stmt[span_end:]
                thm_end_idx = remaining_stmt.find(':=')
                if thm_end_idx == -1:
                    return None
                full_stmt = full_stmt + remaining_stmt[:thm_end_idx] + ' :='
            else:
                full_stmt = full_stmt[:thm_end_idx]
            thm_end_idx = theorem_stmt.rfind('|')
            if thm_end_idx == -1:
                remaining_stmt = stmt[span_end:]
                thm_end_idx = remaining_stmt.find(':=')
                if thm_end_idx == -1:
                    return None
                theorem_stmt = theorem_stmt + remaining_stmt[:thm_end_idx]
            else:
                theorem_stmt = theorem_stmt[:thm_end_idx]
        return theorem_name, theorem_stmt, full_stmt

    def _stmt_has_lemma(self, stmt: str) -> bool:
        # Match the theorem regex
        has_content = self._content_till_last_theorem_stmt is not None
        full_stmt = (stmt if self._content_till_last_theorem_stmt is None else self._content_till_last_theorem_stmt + '\n' + stmt) + '\n'
        theorem_started = Lean4SyncExecutor.theorem_start_match.findall(full_stmt)
        theorem_ended = Lean4SyncExecutor.theorem_end_match.findall(full_stmt)
        is_theorem_started = len(theorem_started) > 0
        is_theorem_ended = len(theorem_ended) > 0
        # Case one where the theorem has started and ended in the same line
        self._content_till_last_theorem_stmt = full_stmt[:-1]
        process_namespaces(full_stmt, self._namespaces, has_content)
        if is_theorem_started and is_theorem_ended:
            last_thm = self._parse_theorem_stmt(full_stmt)
        else:
            last_thm = None
        self._theorem_started = last_thm is not None
        is_theorem_started = self._theorem_started
        is_theorem_ended = is_theorem_started and is_theorem_ended
        if last_thm is not None:
            self._last_theorem = last_thm
        return is_theorem_started
    
    def _get_env(self, idx) -> Optional[int]:
        env_idx = None
        if idx in self._line_to_env_idx_map:
            env_idx = self._line_to_env_idx_map[idx]
        else:
            self._line_to_env_idx_map[idx] = self._last_env_idx
            env_idx = self._last_env_idx
        return env_idx
    
    def _update_env(self, idx: int):
        self._last_env_idx = idx
    
    def _update_proof_state_idx(self, idx: int):
        self._last_proof_state_idx = idx
    
    def _should_start_proof(self, stmt: str) -> bool:
        return self._theorem_started
    
    def _remove_proof_add_sorry(self) -> str:
        # Find the last ':= by' and replace it with 'sorry' and remove the rest of the proof
        matches = list(Lean4SyncExecutor.theorem_end_match.finditer(self._content_till_last_theorem_stmt))
        if len(matches) == 0:
            raise ValueError(f"Could not find the proof in the statement: {self._content_till_last_theorem_stmt}")
        last_match = matches[-1]
        _, span_end = last_match.span()
        full_stmt = self._content_till_last_theorem_stmt[:span_end]
        thm_end_style = last_match.group(3)
        if thm_end_style == "=>":
            # Find the last '|' in the full_stmt
            thm_end_idx = full_stmt.rfind('|')
            if thm_end_idx == -1:
                remaining_stmt = self._content_till_last_theorem_stmt[span_end:]
                thm_end_idx = remaining_stmt.find(':=')
                if thm_end_idx == -1:
                    raise ValueError(f"Could not find the start of proof in the statement: {self._content_till_last_theorem_stmt}")
                full_stmt = full_stmt + remaining_stmt[:thm_end_idx] + ' :='
                new_stmt = full_stmt
            else:
                full_stmt = full_stmt[:thm_end_idx] + ' :='
                new_stmt = full_stmt
        else:
            new_stmt = full_stmt
        if not self.use_file:
            new_stmt += " by sorry"
        else:
            new_stmt += " by\n"
        self._content_till_last_theorem_stmt = new_stmt

    def _run_stmt_on_lean_server(self, idx : int, stmt: str):
        always_use_file = True
        self.use_file = always_use_file
        if "sorry" in stmt and self._proof_running:
            # We don't need to run the sorry statements. This should be treated as a failed proof step
            self.lean_error_messages = ["The tactic 'sorry' was found in the statement, this is not allowed"]
            return
        elif len(stmt.strip()) == 0 and self._proof_running:
            # We don't need to run the empty statements. This should be treated as a failed proof step
            self.lean_error_messages = ["There is no tactic in the statement, it is just empty line or whitespace"]
            return
        proof_should_run = False
        if not self._proof_running and self._stmt_has_lemma(stmt):
            proof_should_run = self._should_start_proof(stmt)
            if proof_should_run:
                theorem_name, theorem_stmt, full_stmt = self._last_theorem
                self.curr_lemma_name = theorem_name
                self.curr_lemma = theorem_stmt
                if len(theorem_name) == 0:
                    self._anon_theorem_count += 1
                    theorem_name = f"anon_theorem____{self._anon_theorem_count}"
                self.local_file_lemmas[theorem_name] = theorem_stmt
                self.local_theorem_lemma_description[theorem_name] = full_stmt
        if not self._proof_running and not proof_should_run:
            return
        if proof_should_run:
            # We need to augment the statement with a sorry
            self._remove_proof_add_sorry()
        env_idx = self._get_env(idx)
        cmd_was_executed = False
        use_file = self.use_file or always_use_file
        response = None
        while not cmd_was_executed:
            if not self._proof_running:
                # Run the statement in cmd mode
                if not use_file:
                    cmd = {"cmd": self._content_till_last_theorem_stmt}
                else:
                    # This might be due to not being able to sync with text
                    if self._file_handle is None:
                        self._file_handle = open(self.temp_file_full_path, "a+")
                    self._last_file_seek = self._file_handle.tell()
                    self._line_num_seek_map[idx] = self._last_file_seek
                    self._file_handle.write(self._content_till_last_theorem_stmt)
                    self._file_handle.flush()
                    # with open(self.temp_file_full_path, "a") as f:
                    #     f.write(self._content_till_last_theorem_stmt)
                    cmd = {"path": self.temp_file_full_path}
                self._content_till_last_theorem_stmt = None
            else:
                # Run the statement in tactic mode
                last_proof_state_idx = self._last_proof_state_idx
                if use_file:
                    if self._file_handle is None:
                        self._file_handle = open(self.temp_file_full_path, "a+")
                    self._last_file_seek = self._file_handle.tell()
                    self._line_num_seek_map[idx] = self._last_file_seek
                    if not stmt.endswith("\n"):
                        stmt += "\n"
                    self._file_handle.write(stmt)
                    self._file_handle.flush()
                    # with open(self.temp_file_full_path, "a") as f:
                    #     f.write(stmt)
                    cmd = {"path": self.temp_file_full_path}
                else:
                    assert last_proof_state_idx is not None, "Proof state index is not set"
                    cmd = {"tactic": stmt, "proofState": last_proof_state_idx}
            if env_idx is not None:
                cmd["env"] = env_idx
            self.process_interace.send_command(cmd)
            timed_out = False
            try:
                timed_out_in_secs = self.timeout_in_sec
                if use_file:
                    timed_out_in_secs *= 4 # File can be big and take time
                response = self.process_interace.read_response(timed_out_in_secs)
                relevant_messages = []
                if 'messages' in response and (use_file or ('proofState' not in response and 'sorries' not in response)):
                    messages = response['messages']
                    # Go over all sev after the line number and check if there is an error
                    for msg in messages:
                        if 'pos' in msg and 'endPos' in msg and \
                        msg['endPos'] is not None and \
                        'line' in msg['endPos'] and \
                        msg['endPos']['line'] >= idx + 1:
                            relevant_messages.append(msg)
                    sevierities = [msg['severity'] for msg in messages]
                    if 'error' in sevierities:
                        cmd_was_executed = use_file
                        use_file = True
                    else:
                        cmd_was_executed = True
                elif 'message' in response and 'proofState' not in response and 'sorries' not in response:
                    self.lean_error_messages = [response['message']]
                    cmd_was_executed = True # There is an irrecoverable error
                    if not self.process_interace.process_is_running():
                        raise Exception("Lean server got killed, probably due to an error in the line executed.\n" + 
                        f"Check the error message: {self.lean_error_messages}")
                else:
                    cmd_was_executed = True
            except TimeoutError:
                if not self.suppress_error_log:
                    self.logger.error(f"Timeout error while running '{stmt}' on lean. File name: {self.main_file}")
                timed_out = True
                cmd_was_executed = True
            except:
                if not self.suppress_error_log:
                    self.logger.error(f"Got an exception while running '{stmt}' on lean. File name: {self.main_file}")
                    self.logger.exception(f"Exception Log")
                cmd_was_executed = use_file # This will force it to run at most twice
                use_file = True
                if cmd_was_executed:
                    raise
        if timed_out:
            self.lean_error_messages = ["The tactic timed out, probably because of repeated application of a tactic which created a very big goal."]
        else:
            if response is None:
                raise ValueError(f"Response is None for the statement: {stmt}")
            if 'env' in response:
                env_idx = response['env']
            else:
                env_idx = None
            self._update_env(env_idx)
            proof_running = 'sorries' in response or 'proofState' in response
            error_messages = response.get('message', None)
            goal_text = None
            if error_messages is None and 'proofState' in response:
                error_messages = response.get('messages', None)
            elif error_messages is None:
                # Go over all the relevant messages and see if there are messages other than unproved goals
                error_messages = []
                for msg in relevant_messages:
                    text_msg = msg.get('data', None)
                    if text_msg is not None and text_msg.startswith('unsolved goals'):
                        goal_text = text_msg[len('unsolved goals'):]
                    else:
                        error_messages.append(msg)
                if len(error_messages) == 0:
                    error_messages = None
                if len(relevant_messages) == 0:
                    goal_text = ''
            elif error_messages is not None:
                error_messages = [error_messages]
            if error_messages is not None:
                self.lean_error_messages = []
                for error_message in error_messages:
                    if isinstance(error_message, dict):
                        error_message = error_message['severity'] + ": " + error_message['data']
                    else:
                        error_message = str(error_message)
                    self.lean_error_messages.append(error_message)
            else:
                self.lean_error_messages = []
                proof_running = proof_running or goal_text is not None
            if error_messages is None:
                assert proof_running, f"Proof is not running but no error message is present, response:\n{response}, \nstmt: \n{stmt}, \nlemma: \n{self.curr_lemma_name}, \nlemma_stmt: \n{self.curr_lemma}, \nline_num: \n{self.line_num}"
                self._proof_running = proof_running
                if self._proof_running:
                    proof_state_idx = None
                    proof_goals = []
                    if goal_text is not None:
                        if len(goal_text) == 0:
                            proof_goals = []
                        else:
                            proof_goals = [goal_text]
                    elif 'sorries' in response:
                        sorries = response['sorries']
                        # TODO: Go over all the sorries and find the one which matches the line number with idx + 1
                        # Now we are only supporting the last sorry
                        proof_state = sorries[-1]
                        proof_state_idx = proof_state['proofState']
                        proof_goals = [proof_state['goal']]
                    elif 'proofState' in response:
                        proof_state = response
                        proof_state_idx = response['proofState']
                        proof_goals = response['goals']
                    self._update_proof_state_idx(proof_state_idx)
                    self.proof_context = self._parse_proof_context(proof_goals)
                    if self.proof_context == ProofContext.empty():
                        self._proof_running = False
                        self.proof_context = None
                        self.curr_lemma = None
                        self.curr_lemma_name = None
                else:
                    self.proof_context = None
        pass

    def _skip_to_theorem(self, theorem: str):
        # Skip to the given theorem
        found_theorem = False
        thm_namespace, given_theorem_name = parse_thm_name(theorem)
        while not found_theorem and not self.execution_complete:
            try:
                stmt = next(self.main_file_iter)
            except StopIteration:
                self.execution_complete = True
                break
            self.current_stmt = stmt
            self.line_num += 1
            if self._stmt_has_lemma(stmt):
                proof_should_run = self._should_start_proof(stmt)
                if proof_should_run:
                    thm_name, thm_stmt, full_thm_stmt = self._last_theorem
                    last_namespace = self._namespaces[-1] if len(self._namespaces) > 0 else ""
                    if thm_name is not None and thm_name == given_theorem_name and (len(thm_namespace) == 0 or thm_namespace == last_namespace):
                        found_theorem = True
                        self._theorem_started = True
                        self._content_till_last_theorem_stmt = '\n'.join(self._lines_executed)
                        self._run_stmt_on_lean_server(len(self._lines_executed), stmt)
                    elif thm_name is not None:
                        if len(thm_name) == 0:
                            self._anon_theorem_count += 1
                            thm_name = f"anon_theorem____{self._anon_theorem_count}"
                        self.local_file_lemmas[thm_name] = thm_stmt
                        self.local_theorem_lemma_description[thm_name] = full_thm_stmt
                    self._content_till_last_theorem_stmt = None
            self._lines_executed.append(stmt)
        if not found_theorem:
            raise ValueError(f"The theorem '{theorem}' was not found in the file '{self.main_file}'")

    def _parse_proof_context(self, proof_goals: list) -> ProofContext:
        goals = []
        for proof_goal in proof_goals:
            if self.use_human_readable_proof_context:
                goals.extend(Lean4Utils.parse_proof_context_human_readable_as_goals(proof_goal))
            else:
                raise NotImplementedError("Parsing of non-human readable proof context is not implemented")
        if len(goals) == 0:
            return ProofContext.empty()
        else:
            return ProofContext(goals, [], [], [])
    

theorem_names_in_file_cache: Dict[str, List[TheoremDetails]] = {}
namespace_regex = r"^namespace[ ]+([\S]+)"
namespace_match = re.compile(namespace_regex, re.MULTILINE)
namespace_end_regex = r"^end[ ]+([\S]+)*"
namespace_end_match = re.compile(namespace_end_regex, re.MULTILINE)

def parse_thm_name(theorem_name: str) -> Tuple[str, str]:
    if theorem_name.startswith("{") and theorem_name.endswith("}"):
        thm_dict = json.loads(theorem_name)
        return thm_dict["namespace"], thm_dict["name"]
    else:
        return "", theorem_name

def process_namespaces(file_cotent: str, open_namespaces: List[str], is_full_content: bool=False):
    # Match the namespace regex
    # Break the content line by line and match the namespace and end namespace
    file_lines = file_cotent.split('\n')
    for line in file_lines:
        namespace_matches = namespace_match.findall(line)
        namespace_end_matches = namespace_end_match.findall(line)
        for ns in namespace_matches:
            if not is_full_content or ns not in open_namespaces:
                open_namespaces.append(ns)
        for ns in namespace_end_matches:
            try:
                open_namespaces.remove(ns)
            except ValueError:
                pass

def get_all_theorems_in_file(file_path: str, use_cache: bool=False) -> List[TheoremDetails]:
    if use_cache and file_path in theorem_names_in_file_cache:
        return theorem_names_in_file_cache[file_path]
    file_content = ""
    open_namespaces = []
    with open(file_path, "r") as f:
        file_content = f.read()
    line_by_line_reader = LeanLineByLineReader(file_content=file_content, remove_comments=True, no_strip=True)
    all_stmts = list(line_by_line_reader.instruction_step_generator())
    full_content = '\n'.join(all_stmts)
    # all_matches = Lean4SyncExecutor.theorem_match.findall(full_content)
    all_matches = list(Lean4SyncExecutor.theorem_match.finditer(full_content))
    all_theorems = []
    last_namespace_processed_idx = 0
    for match in all_matches:
        span_start, span_end = match.span()
        process_namespaces(full_content[last_namespace_processed_idx:span_start], open_namespaces)
        theorem_name = match.group(5)
        theorem_name = theorem_name if theorem_name is not None else f"\"{match.group(6).strip(': ')}\""
        theorem_namespace = '.'.join(open_namespaces) if len(open_namespaces) > 0 else ''
        theorem_details = TheoremDetails(theorem_name=theorem_name, theorem_namespace=theorem_namespace, theorem_file_path=file_path)
        all_theorems.append(theorem_details)
        last_namespace_processed_idx = span_end
    if use_cache:
        theorem_names_in_file_cache[file_path] = all_theorems
    return all_theorems

def get_fully_qualified_theorem_name(theorem_details: TheoremDetails) -> str:
    if len(theorem_details.theorem_namespace) == 0:
        return theorem_details.theorem_name
    else:
        dict_thm = {"namespace": theorem_details.theorem_namespace, "name": theorem_details.theorem_name}
        return json.dumps(dict_thm)

def get_theorem_name_resembling(file_path: str, theorem_name: str, use_cache: bool=False) -> Optional[str]:
    all_theorems = get_all_theorems_in_file(file_path, use_cache=use_cache)
    all_theorems_name_unique_map : Dict[str, List[TheoremDetails]] = {}
    for thm in all_theorems:
        if thm.theorem_name in all_theorems_name_unique_map:
            all_theorems_name_unique_map[thm.theorem_name].append(thm)
        else:
            all_theorems_name_unique_map[thm.theorem_name] = [thm]
    all_parts = theorem_name.split('.')
    thm_start_idx = len(all_parts) - 1
    thm_found = False
    while not thm_found and thm_start_idx >= 0:
        full_name = '.'.join(all_parts[thm_start_idx:])
        # look for any theorems matching with full_name
        thm_found = full_name in all_theorems_name_unique_map
        thm_start_idx -= 1
    if not thm_found:
        full_name = '_root_.' + full_name
        # look for any theorems matching with the full_name
        thm_found = full_name in all_theorems_name_unique_map
        if not thm_found:
            raise ValueError(f"The theorem '{theorem_name}' was not found in the file '{file_path}'")
    assert thm_found, "The theorem was not found some code bug in finding the theorem names"
    theorem_name_matches = all_theorems_name_unique_map[full_name]
    if len(theorem_name_matches) == 1:
        if len(theorem_name_matches[0].theorem_namespace) == 0:
            return theorem_name_matches[0].theorem_name
        else:
            dict_thm = {"namespace": theorem_name_matches[0].theorem_namespace, "name": theorem_name_matches[0].theorem_name}
            return json.dumps(dict_thm)
    else:
        # We need to find the namespace which matches with the theorem_name
        for thm in theorem_name_matches:
            if theorem_name.endswith(thm.theorem_namespace + '.' + thm.theorem_name) or\
            (theorem_name.strip() == thm.theorem_name and len(thm.theorem_namespace) == 0):
                dict_thm = {"namespace": thm.theorem_namespace, "name": thm.theorem_name}
                return json.dumps(dict_thm)
        raise ValueError(f"The theorem '{theorem_name}' was not found in the file '{file_path}'")

if __name__ == "__main__":
    # project_root = 'data/test/lean4_proj/'
    # file_path = 'data/test/lean4_proj/Lean4Proj/Basic.lean'
    project_root = 'data/test/lean4_proj/'
    file_path = 'data/test/lean4_proj/Lean4Proj/putnam_test15.lean'
    os.chdir(root_dir)
    assert os.path.exists(project_root), "Project root does not exist"
    assert os.path.exists(file_path), "File path does not exist"
    print("Finding all theorems in the file")
    all_theorems = get_all_theorems_in_file(file_path, use_cache=True)
    print(all_theorems)
    theorem_name = "putnam_1988_b1"
    theorems_similar_to_test = get_theorem_name_resembling(file_path, theorem_name, use_cache=True)
    print("Theorem similar to ", theorem_name, " is ", theorems_similar_to_test)
    with Lean4SyncExecutor(main_file=file_path, project_root=project_root) as executor:
        executor._skip_to_theorem(theorems_similar_to_test)
        while not executor.execution_complete:
            executor.run_next()
            print("Current statement:", executor.current_stmt)
            if executor.proof_context is not None:
                for goal in executor.proof_context.all_goals:
                    for hyp in goal.hypotheses:
                        print(hyp)
                    print('-'*10)
                    print(goal.goal)
                print('-'*20)
            if executor.lean_error_messages:
                print("Error messages:\n", executor.lean_error_messages)
    # mathlib_test_file = 'data/test/Mathlib/.lake/packages/mathlib/Mathlib/Data/Nat/Bits.lean'
    # project_root = 'data/test/Mathlib'
    # assert os.path.exists(mathlib_test_file), "Mathlib test file does not exist"
    # assert os.path.exists(project_root), "Project root does not exist"
    # with Lean4SyncExecutor(main_file=mathlib_test_file, project_root=project_root, timeout_in_sec=120) as executor:
    #     executor._skip_to_theorem("one_bits")
    #     assert executor.proof_context is not None, "Proof context should be present"
    #     print("Starting the proof")
    #     for goal in executor.proof_context.all_goals:
    #         for hyp in goal.hypotheses:
    #             print(hyp)
    #         print('-'*10)
    #         print(goal.goal)
    #     while not executor.execution_complete and executor.proof_context is not None:
    #         executor.run_next()
    #         print("Current statement:", executor.current_stmt)
    #         if executor.proof_context is not None:
    #             for goal in executor.proof_context.all_goals:
    #                 for hyp in goal.hypotheses:
    #                     print(hyp)
    #                 print('-'*10)
    #                 print(goal.goal)
    #             print('-'*20)
    #         if executor.lean_error_messages:
    #             print("Error messages:\n", executor.lean_error_messages)
