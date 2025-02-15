#!/usr/bin/env python3

import os
import bisect
import random
import logging
import uuid
import re
import time
import json
import shutil
import typing
from copra.lean_server.lean_context import ProofContext
from copra.lean_server.lean4_utils import Lean4Utils
from copra.tools.lean_parse_utils import LeanLineByLineReader
from copra.tools.theorem_details import TheoremDetails
from copra.lean_server.lean4_repl_interface import ProcessInterface
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
    theorem_name_regex = r"(((theorem|lemma)[\s]+([\S]*))|example)"
    remove_proof_regex = r"([\s|\S]*(:=|\|))[\s|\S]*?"
    proof_context_separator = "⊢"
    proof_context_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
    goal_regex = rf"([\s|\S]*?){proof_context_separator}([\s|\S]*)"
    theorem_match = re.compile(theorem_regex, re.MULTILINE)
    theorem_name_match = re.compile(theorem_name_regex, re.MULTILINE)
    proof_context_match = re.compile(proof_context_regex, re.MULTILINE)
    goal_match = re.compile(goal_regex, re.MULTILINE)
    theorem_start_match = re.compile(theorem_start_regex, re.MULTILINE)
    theorem_end_match = re.compile(theorem_end_regex, re.MULTILINE)
    remove_proof_match = re.compile(remove_proof_regex, re.MULTILINE)
    proof_context_generation_tactic = "\nend"
    proof_context_generation_tactic_curlies = "\n}"
    proof_state_running_message = "tactic failed, there are unsolved goals\nstate:"
    unsolved_message = "unsolved goals"
    theorem_detection_message = "unexpected end of input; expected '{'"
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
        self._in_tactic_mode = False
        self._env_idx_last_thm = None
        self._last_tactics = {}
        self._last_tactic_line_idx = None
        self._error_messages_so_far = set()
        self._error_messages_since_last_thm = {}
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
        return True if self.proof_context else (len(self.lean_error_messages) > 0) # It is still in proof mode if we encountered a wrong proof

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
        try:
            return self.local_theorem_lemma_description[self.curr_lemma_name]
        except:
            return None
        
    def get_current_lemma_name(self) -> Optional[str]:
        if self.curr_lemma_name is None:
            return None
        else:
            return self.curr_lemma_name
    
    def _check_if_thm_read(self, idx: int, full_stmt: str) -> bool:
        cmd = self._get_cmd_tactic_mode(idx, full_stmt)
        self.process_interace.send_command(cmd)
        timeout_in_secs = self.timeout_in_sec
        response = self.process_interace.read_response(timeout_in_secs)
        messages = response.get('messages', [])
        has_cnt = 0
        has_unfocussed_goal = 0
        has_new_errors = 0
        for msg_idx, msg in enumerate(messages):
            if msg['severity'] == 'error' and 'pos' in msg and 'endPos' in msg and \
            ((msg['endPos'] is not None and 'line' in msg['endPos']) or \
                (msg['pos'] is not None and 'line' in msg['pos'])):
                if msg['data'].startswith(Lean4SyncExecutor.theorem_detection_message) and msg['endPos'] is None:
                    has_cnt += 1
                elif msg['data'].startswith(Lean4SyncExecutor.unsolved_message):
                    has_unfocussed_goal += 1
                else:
                    full_error_msg = self._get_error_msg(msg_idx, msg)
                    if full_error_msg in self._error_messages_so_far:
                        continue
                    has_new_errors += 1
                    self._errors_since_last_thm(idx, full_error_msg)
            elif msg['severity'] == 'warning' and 'pos' in msg and 'endPos' in msg and 'sorry' in msg['data']:
                full_error_msg = self._get_error_msg(msg_idx, msg)
                if full_error_msg in self._error_messages_so_far:
                    continue
                self._error_messages_so_far.add(full_error_msg)
                self._errors_since_last_thm(idx, full_error_msg)
        return has_cnt == 1 and has_unfocussed_goal == 1 and has_new_errors == 0
    
    def _parse_theorem_stmt(self, idx: int, stmt: str, do_full_check: bool = False, interesting_span: typing.Tuple[int, int] = None) -> str:
        if interesting_span is not None:
            span_start, span_end = interesting_span
            full_stmt = stmt[span_start:span_end]
            thm_name_matches = list(Lean4SyncExecutor.theorem_name_match.finditer(full_stmt))
            if len(thm_name_matches) == 0:
                return None
            thm_end_style = '=>' if stmt.strip().endswith('=>') else ':='
            thm_name_match = thm_name_matches[-1]
            _, nspan_end = thm_name_match.span()
            theorem_name = thm_name_match.group(4)
            theorem_stmt = full_stmt[nspan_end:].strip().strip(thm_end_style)
        else:
            matches = list(Lean4SyncExecutor.theorem_match.finditer(stmt))
            if len(matches) == 0:
                return None
            # We are only interested in the last theorem
            match = matches[-1]
            span_start, _ = match.span()
            full_stmt = match.group(1)
            theorem_name = match.group(5)
            theorem_stmt = match.group(6)
            thm_end_style = match.group(7)
        if thm_end_style == "=>":
            # Find the last '|' in the full_stmt
            thm_end_idx = full_stmt.rfind('|')
            if thm_end_idx == -1:
                return None
            else:
                full_stmt = full_stmt[:thm_end_idx] + ' :='
            thm_end_idx = theorem_stmt.rfind('|')
            if thm_end_idx == -1:
                return None
            else:
                theorem_stmt = theorem_stmt[:thm_end_idx]
        if do_full_check:
            check_stmt = stmt[:span_start] + full_stmt + ' by\n'
            if not self._check_if_thm_read(idx, check_stmt):
                return None
        return theorem_name, theorem_stmt, full_stmt
    
    def _execute_till_last_theorem(self, idx: int, full_stmt: str):
        self._write_lean_file(idx, full_stmt)
        self._run_file_on_lean_server(self.timeout_in_sec * 4)

    def _stmt_has_lemma(self, idx: int, stmt: str, do_full_check: bool = False) -> bool:
        # Match the theorem regex
        has_content = self._content_till_last_theorem_stmt is not None
        full_stmt = (stmt if self._content_till_last_theorem_stmt is None else self._content_till_last_theorem_stmt + '\n' + stmt) + '\n'
        theorem_started = list(Lean4SyncExecutor.theorem_start_match.finditer(full_stmt))
        theorem_ended = Lean4SyncExecutor.theorem_end_match.findall(full_stmt)
        is_theorem_started = len(theorem_started) > 0
        is_theorem_ended = len(theorem_ended) > 0
        # Case one where the theorem has started and ended in the same line
        self._content_till_last_theorem_stmt = full_stmt[:-1]
        process_namespaces(full_stmt, self._namespaces, has_content)
        if is_theorem_started and is_theorem_ended:
            last_span_start, last_span_end = theorem_started[-1].span()
            stmt_before_theorem = full_stmt[:last_span_start]
            if len(stmt_before_theorem.strip()) > 0 and do_full_check:
                # We need to run the statement before the theorem
                self._execute_till_last_theorem(0, stmt_before_theorem)
                self._content_till_last_theorem_stmt = self._content_till_last_theorem_stmt[last_span_start:]
                last_span_end -= last_span_start
                full_stmt = full_stmt[last_span_start:]
                last_span_start = 0
            # Look for all ':=' in the full_stmt
            endings = [i for i in range(last_span_end, len(full_stmt)) if full_stmt.startswith(':=', i)]
            last_thm = None
            for ending in endings:
                interesting_stmt = full_stmt[:ending] + ' := ' # We need to add ':=' to the end
                interesting_span = (last_span_start, len(interesting_stmt))
                last_thm = self._parse_theorem_stmt(idx, interesting_stmt, do_full_check, interesting_span) 
                if last_thm is not None:
                    self._content_till_last_theorem_stmt = full_stmt[:last_span_start] + last_thm[2] + ' by\n'
                    break
            if last_thm is None:
                endings = [i for i in range(last_span_end, len(full_stmt)) if full_stmt.startswith('=> ', i)]
                for ending in endings:
                    interesting_stmt = full_stmt[:ending] + ' => ' # We need to add '=>' to the end
                    interesting_span = (last_span_start, len(interesting_stmt))
                    last_thm = self._parse_theorem_stmt(idx, interesting_stmt, do_full_check, interesting_span) 
                    if last_thm is not None:
                        self._content_till_last_theorem_stmt = full_stmt[:last_span_start] + last_thm[2] + ' by\n'
                        break
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
        span_start, span_end = last_match.span()
        full_stmt = self._content_till_last_theorem_stmt[:span_end]
        thm_end_style = last_match.group(3)
        if thm_end_style == "=>":
            # Find the last '|' in the full_stmt
            thm_end_idx = full_stmt.rfind('|', start = span_start)
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

    def _write_lean_file(self, idx: int, file_content: str):
        if self._file_handle is None:
            self._file_handle = open(self.temp_file_full_path, "a+")
        self._last_file_seek = self._file_handle.tell()
        self._line_num_seek_map[idx] = self._last_file_seek
        self._file_handle.write(file_content)
        self._file_handle.flush()
        pass
    
    def _get_error_msg(self, msg_idx, msg) -> str:
        line_start = msg['pos']['line'] if msg['pos'] is not None else ""
        line_end = msg['endPos']['line'] if msg['endPos'] is not None else ""
        full_error_msg = str(msg_idx) + ' ' + str(line_start) + " - " + str(line_end) + ": " + str(msg['data'])
        return full_error_msg

    def _run_file_on_lean_server(self, timeout_in_sec: int):
        cmd = {"path": self.temp_file_full_path}
        self.process_interace.send_command(cmd)
        response = self.process_interace.read_response(timeout_in_sec)
        if 'messages' in response:
            messages = response['messages']
            # Go over all sev after the line number and check if there is an error
            for msg_idx, msg in enumerate(messages):
                if msg['severity'] == 'error' and 'pos' in msg and 'endPos' in msg and \
                ((msg['endPos'] is not None and 'line' in msg['endPos']) or \
                    (msg['pos'] is not None and 'line' in msg['pos'])):
                    if msg['data'].startswith(Lean4SyncExecutor.theorem_detection_message) and msg['endPos'] is None:
                        continue # Ignore this error
                    full_error_msg = self._get_error_msg(msg_idx, msg)
                    self._error_messages_so_far.add(full_error_msg)
                elif msg['severity'] == 'warning' and 'pos' in msg and 'endPos' in msg and 'sorry' in msg['data']:
                    full_error_msg = self._get_error_msg(msg_idx, msg)
                    if full_error_msg in self._error_messages_so_far:
                        continue
                    self._error_messages_so_far.add(full_error_msg)
        if 'env' in response:
            self._update_env(response['env'])
        self._env_idx_last_thm = response.get('env', None)
        return response
    
    def _add_last_tactic(self, idx: int, stmt: str):
        if idx not in self._last_tactics:
            self._last_tactics[idx] = stmt
            self._last_tactic_line_idx = idx

    def _get_cmd_tactic_mode(self, idx: int, stmt: str):
        self._add_last_tactic(idx, stmt)
        tactics_so_far = [(k, v) for k, v in self._last_tactics.items()]
        tactics_so_far = sorted(tactics_so_far, key=lambda x: x[0])
        tactics_so_far = [v for _, v in tactics_so_far]
        if self._env_idx_last_thm is None:
            return {"cmd": "\n".join(tactics_so_far)}
        else:
            return {"cmd": "\n".join(tactics_so_far), "env": self._env_idx_last_thm}
    
    def _errors_since_last_thm(self, idx, error_message: str):
        if idx not in self._error_messages_since_last_thm:
            self._error_messages_since_last_thm[idx] = error_message

    def _backtrack_tactic_line(self, idx: int):
        # identify the keys to remove
        idx_to_remove = []
        backtracked = False
        for k in self._last_tactics.keys():
            if k >= idx:
                idx_to_remove.append(k)
        for k in idx_to_remove:
            backtracked = True
            del self._last_tactics[k]
        idx_to_remove = []
        for k in self._error_messages_since_last_thm.keys():
            if k >= idx:
                idx_to_remove.append(k)
        for k in idx_to_remove:
            backtracked = True
            msg = self._error_messages_since_last_thm[k]
            if msg in self._error_messages_so_far:
                self._error_messages_so_far.remove(msg)
            del self._error_messages_since_last_thm[k]
        self._last_tactic_line_idx = max(self._last_tactics.keys(), default=None) 
        return backtracked

    def _clear_tacitcs(self):
        tactics_so_far = [(k, v) for k, v in self._last_tactics.items()]
        tactics_so_far = sorted(tactics_so_far, key=lambda x: x[0])
        tactics_so_far = [v for _, v in tactics_so_far]
        self._write_lean_file(self._last_tactic_line_idx, "\n".join(tactics_so_far))
        self._last_tactics = {}
        self._last_tactic_line_idx = None
        self._error_messages_since_last_thm = {}
        pass

    def get_all_proofs_in_file(self) -> Dict[str, List[Tuple[ProofContext, str]]]:
        assert self.main_file is not None, "Main file is not provided"
        abs_main_file = os.path.abspath(self.main_file)
        assert os.path.exists(abs_main_file), f"Main file does not exist at {abs_main_file}"
        assert abs_main_file.endswith(".lean"), "Main file must be a '.lean' file"
        temp_file = os.path.join(self.project_root, self.temp_filename_suffix)
        abs_temp_file = os.path.abspath(temp_file)
        # Copy the file to a temporary file
        shutil.copyfile(abs_main_file, abs_temp_file)
        try:
            abs_main_file = abs_temp_file
            # Remove all the comments from the file
            line_by_line_reader = LeanLineByLineReader(abs_main_file, remove_comments=True, no_strip=True)
            all_stmts = list(line_by_line_reader.instruction_step_generator())
            new_content = "\n".join(all_stmts)
            with open(abs_main_file, "w") as f:
                f.write(new_content)
            # Run the file on the lean server
            self.process_interace.send_command({"path": abs_main_file, "allTactics": True})
            response = self.process_interace.read_response(self.timeout_in_sec*20)
            tactics_resp = response.get('tactics', [])
            # Parse all goals in these tactics
            goals = [self._parse_proof_context([t['goals']]) for t in tactics_resp]
            tactics = [t['tactic'] for t in tactics_resp]
            line_nums = [t['pos']['line'] for t in tactics_resp]
            line_num_dx = 0
            result = {}
            thm_id = uuid.uuid4().hex
            thm_cnt = 0
            all_theorems = get_all_theorems_in_file(abs_main_file, use_cache = True)
            line_to_thm_map : typing.Dict[int, TheoremDetails] = {}
            for thm_detail in all_theorems:
                line_num = thm_detail.theorem_pos['line_start']
                line_to_thm_map[line_num] = thm_detail
            with open(abs_main_file, "r") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if Lean4SyncExecutor.theorem_start_match.match(line):
                        thm_detail = line_to_thm_map.get(idx + 1, None)
                        if thm_detail is not None:
                            thm_id = get_fully_qualified_theorem_name(thm_detail)
                        else:
                            thm_id = uuid.uuid4().hex + f"_{thm_cnt}"
                        thm_cnt += 1
                    while line_num_dx < len(line_nums) and line_nums[line_num_dx] == idx + 1:
                        if thm_id not in result:
                            result[thm_id] = [(goals[line_num_dx], tactics[line_num_dx])]
                        else:
                            result[thm_id].append((goals[line_num_dx], tactics[line_num_dx]))
                        line_num_dx += 1
            return result
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _run_stmt_on_lean_server(self, idx : int, stmt: str, theorem_started: bool = False):
        if "sorry" in stmt and self._proof_running:
            # We don't need to run the sorry statements. This should be treated as a failed proof step
            self.lean_error_messages = ["The tactic 'sorry' was found in the statement, this is not allowed"]
            return
        elif len(stmt.strip()) == 0 and self._proof_running:
            # We don't need to run the empty statements. This should be treated as a failed proof step
            self.lean_error_messages = ["There is no tactic in the statement, it is just empty line or whitespace"]
            return
        proof_should_run = False
        if theorem_started or (not self._proof_running and self._stmt_has_lemma(idx, stmt, do_full_check = True)):
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
        env_idx = self._get_env(idx)
        cmd_was_executed = False
        response = None
        while not cmd_was_executed:
            # Run the statement in tactic mode
            if self._env_idx_last_thm is None:
                self._env_idx_last_thm = env_idx
            if self.process_interace.is_rebooted():
                self._run_file_on_lean_server(self.timeout_in_sec * 4)
            cmd = self._get_cmd_tactic_mode(idx, stmt)
            if env_idx is not None and 'env' not in cmd:
                cmd["env"] = env_idx
            self.process_interace.send_command(cmd)
            self._content_till_last_theorem_stmt = None
            timed_out = False
            try:
                timed_out_in_secs = self.timeout_in_sec
                response = self.process_interace.read_response(timed_out_in_secs)
                relevant_messages = []
                if 'messages' in response:
                    messages = response['messages']
                    # Go over all sev after the line number and check if there is an error
                    for msg_idx, msg in enumerate(messages):
                        full_error_msg = self._get_error_msg(msg_idx, msg)
                        unsolved_goal_never_seen_before = not (full_error_msg in self._error_messages_since_last_thm.values())
                        if msg['severity'] == 'error' and 'pos' in msg and 'endPos' in msg and \
                        ((msg['endPos'] is not None and 'line' in msg['endPos']) or \
                         (msg['pos'] is not None and 'line' in msg['pos'])):
                            if msg['data'].startswith(Lean4SyncExecutor.theorem_detection_message) and msg['endPos'] is None:
                                continue # Ignore this error
                            if full_error_msg in self._error_messages_so_far and unsolved_goal_never_seen_before:
                                continue
                            self._error_messages_so_far.add(full_error_msg)
                            self._errors_since_last_thm(idx, full_error_msg)
                            if not unsolved_goal_never_seen_before:
                                msg['data'] = 'error: ' + msg['data']
                            relevant_messages.append(msg)
                        elif msg['severity'] == 'warning' and 'pos' in msg and 'endPos' in msg and 'sorry' in msg['data']:
                            full_error_msg = self._get_error_msg(msg_idx, msg)
                            if full_error_msg in self._error_messages_so_far and unsolved_goal_never_seen_before:
                                continue
                            self._error_messages_so_far.add(full_error_msg)
                            self._errors_since_last_thm(idx, full_error_msg)
                            if not unsolved_goal_never_seen_before:
                                msg['data'] = 'error: ' + msg['data']
                            relevant_messages.append(msg)
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
                cmd_was_executed = True
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
            if self._env_idx_last_thm is None and not self._proof_running:
                # self._add_last_tactic(idx, last_content)
                self._env_idx_last_thm = env_idx
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
                    if text_msg is not None and text_msg.startswith(Lean4SyncExecutor.unsolved_message):
                        goal_text = text_msg[len(Lean4SyncExecutor.unsolved_message):]
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
                        self._clear_tacitcs()
                        self._env_idx_last_thm = None
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
            if self._stmt_has_lemma(self.line_num - 1, stmt):
                proof_should_run = self._should_start_proof(stmt)
                if proof_should_run:
                    thm_name, thm_stmt, full_thm_stmt = self._last_theorem
                    last_namespace = ".".join(self._namespaces) if len(self._namespaces) > 0 else ""
                    if thm_name is not None and thm_name == given_theorem_name and (len(thm_namespace) == 0 or thm_namespace == last_namespace):
                        found_theorem = True
                        orig_thm_started = self._theorem_started
                        self._theorem_started = True
                        self._content_till_last_theorem_stmt = '\n'.join(self._lines_executed)
                        if self._stmt_has_lemma(self.line_num - 1, stmt, do_full_check=True):
                            self._run_stmt_on_lean_server(len(self._lines_executed), stmt, theorem_started=True)
                        else:
                            found_theorem = False
                            self._theorem_started = orig_thm_started
                    else:
                        if thm_name is None or len(thm_name) == 0:
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
    line_positions = [0] + [len(stmt) + 1 for stmt in all_stmts]
    # Cumulative sum of the line positions
    for i in range(1, len(line_positions)):
        line_positions[i] += line_positions[i - 1]
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
        line_number_start = bisect.bisect_left(line_positions, span_start)
        line_number_end = bisect.bisect_left(line_positions, span_end)
        theorem_pos = {
            'line_start': line_number_start + 1,
            'line_end': line_number_end + 1,
            'global_pos_start': span_start,
            'global_pos_end': span_end,
            'line_pos_start': span_start - line_positions[line_number_start] if line_number_start < len(line_positions) else 0,
            'line_pos_end': span_end - line_positions[line_number_end] if line_number_end < len(line_positions) else 0
        }
        theorem_details = TheoremDetails(
            theorem_name=theorem_name, 
            theorem_namespace=theorem_namespace, 
            theorem_file_path=file_path, 
            theorem_pos=theorem_pos)
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
    file_path = 'data/test/lean4_proj/Lean4Proj/Basic.lean'
    assert os.path.exists(project_root), "Project root does not exist"
    assert os.path.exists(file_path), "File path does not exist"
    print("Finding all theorems in the file")
    all_theorems = get_all_theorems_in_file(file_path, use_cache=True)
    print(all_theorems)
    theorem_name = "test3"
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
