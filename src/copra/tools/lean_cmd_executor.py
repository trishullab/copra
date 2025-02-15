#!/usr/bin/env python3

import sys


root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import subprocess
import os
import logging
import typing
import time
import random
import re
import copy
from collections import OrderedDict
from src.tools.lean_parse_utils import LeanLineByLineReader
from src.lean_server.lean_cmd_server import LeanCmdServer
from src.lean_server.lean_utils import Lean3Utils
from src.lean_server.lean3_search_tool import Constants, Lean3Lemma, Lean3SearchTool
logger = logging.getLogger()

class Obligation(typing.NamedTuple):
    hypotheses: typing.List[str]
    goal: str

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {"hypotheses": self.hypotheses,
                "goal": self.goal}


class ProofContext(typing.NamedTuple):
    fg_goals: typing.List[Obligation]
    bg_goals: typing.List[Obligation]
    shelved_goals: typing.List[Obligation]
    given_up_goals: typing.List[Obligation]

    @classmethod
    def empty(cls: typing.Type['ProofContext']):
        return ProofContext([], [], [], [])

    @classmethod
    def from_dict(cls, data):
        fg_goals = list(map(Obligation.from_dict, data["fg_goals"]))
        bg_goals = list(map(Obligation.from_dict, data["bg_goals"]))
        shelved_goals = list(map(Obligation.from_dict, data["shelved_goals"]))
        given_up_goals = list(map(Obligation.from_dict,
                                  data["given_up_goals"]))
        return cls(fg_goals, bg_goals, shelved_goals, given_up_goals)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {"fg_goals": list(map(Obligation.to_dict, self.fg_goals)),
                "bg_goals": list(map(Obligation.to_dict, self.bg_goals)),
                "shelved_goals": list(map(Obligation.to_dict,
                                          self.shelved_goals)),
                "given_up_goals": list(map(Obligation.to_dict,
                                           self.given_up_goals))}

    @property
    def all_goals(self) -> typing.List[Obligation]:
        return self.fg_goals + self.bg_goals + \
            self.shelved_goals + self.given_up_goals

    @property
    def focused_goal(self) -> str:
        if self.fg_goals:
            return self.fg_goals[0].goal
        else:
            return ""

    @property
    def focused_hyps(self) -> typing.List[str]:
        if self.fg_goals:
            return self.fg_goals[0].hypotheses
        else:
            return []

class Lean3Executor(object):
    # Write now this doesn't do the best job of capturing each state changing
    # tactic separately. It relies on the fact the tactics are usually written in an
    # atomic way. For example, if the user writes:
    #   intros,
    #   split,
    #   cases a,...
    # However for a simple REPL doesn't need to capture changes of every state change
    # (which can be done by a tideous character by character comparision of the state on lean --server
    # and the state after running the tactic). Instead, we can just assume that the user is intelligent
    # enough to write tactics in an atomic way. This is a reasonable assumption for a simple REPL.
    # The keywords below are taken from https://github.com/leanprover/lean-mode/blob/master/lean-syntax.el
    keywords = {
        "import", "prelude", "protected", "private", "noncomputable", "definition", "meta", "renaming",
        "hiding", "exposing", "parameter", "parameters", "begin", "constant", "constants",
        "lemma", "variable", "variables", "theorem", "example", "abbreviation",
        "open", "export", "axiom", "axioms", "inductive", "coinductive", "with", "without",
        "structure", "universe", "universes", "hide", "precedence", "reserve", "declare_trace", "add_key_equivalence",
        "match", "infix", "infixl", "infixr", "notation", "postfix", "prefix", "instance",
        "end", "this", "using", "using_well_founded", "namespace", "section",
        "attribute", "local", "set_option", "extends", "include", "omit", "classes", "class",
        "attributes", "raw", "replacing", "calc", "have", "show", "suffices", "by", "in", "at", 
        "do", "let", "forall", #"Pi", 
        "fun", "exists", "if", "then", "else", "assume", "from", "mutual", "def", "run_cmd"
        # Note that there are UTF-8 characters in the following list
        # "#", "@", "!", "$", "->", "∼", "↔", "/", "==", "=", ":=", "<->", "/\\", "\\/", "∧", "∨",
        # "≠", "<", ">", "≤", "≥", "¬", "<=", ">=", "⁻¹", "⬝", "▸", "+", "*", "-", "/", "λ",
        # "→", "∃", "∀", "∘", "×", "Σ", "Π", "~", "||", "&&", "≃", "≡", "≅",
        # "ℕ", "ℤ", "ℚ", "ℝ", "ℂ", "𝔸",
        # "⬝e", "⬝i", "⬝o", "⬝op", "⬝po", "⬝h", "⬝v", "⬝hp", "⬝vp", "⬝ph", "⬝pv", "⬝r", "◾", "◾o",
        # "∘n", "∘f", "∘fi", "∘nf", "∘fn", "∘n1f", "∘1nf", "∘f1n", "∘fn1",
        # "^c", "≃c", "≅c", "×c", "×f", "×n", "+c", "+f", "+n", "ℕ₋₂"
    }
    theorem_regex = r"(((theorem ([\w+|\d+]*))|example)([\S|\s]*?):=[\S|\s]*?)(begin|by|calc)"
    proof_context_separator = "⊢"
    proof_context_regex = r"((\d+) goals)*([\s|\S]*?)\n\n"
    goal_regex = rf"([\s|\S]*?){proof_context_separator}([\s|\S]*)"
    theorem_match = re.compile(theorem_regex, re.MULTILINE)
    proof_context_match = re.compile(proof_context_regex, re.MULTILINE)
    goal_match = re.compile(goal_regex, re.MULTILINE)
    proof_context_generation_tactic = "\nend"
    proof_state_running_message = "tactic failed, there are unsolved goals\nstate:"
    search_tools: typing.Dict[str, typing.Any] = {}
    def __init__(self, project_root: str = None, prefix: str = None, main_file: str = None, use_hammer: bool = False, timeout_in_sec: int = 60, use_human_readable_proof_context: bool = False, proof_step_iter: typing.Iterator[str] = None, suppress_error_log: bool = False, mathlib_root: typing.Optional[str] = None, enable_search: bool = False, namespaces: typing.List[str] = None):
        assert proof_step_iter is None or isinstance(proof_step_iter, typing.Iterator), \
            "proof_step_iter must be an iterator"
        assert main_file is not None or proof_step_iter is not None, \
            "Either main_file or proof_step_iter must be provided"
        assert main_file is None or proof_step_iter is None, \
            "Only one of main_file or proof_step_iter must be provided"
        assert main_file is None or (os.path.exists(main_file) and main_file.endswith(".lean")), \
            "main_file must be a valid path to a '.lean' file"
        assert project_root is None or (os.path.exists(project_root) and os.path.isdir(project_root)), \
            "project_root must be a valid path to a directory"
        assert not use_hammer, "Hammer is not supported for Lean3"
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.project_root = project_root if project_root is not None else "."
        self.main_file = main_file
        self.ticks = str(time.time()).replace(".", "") # This ensures that the temp file name is unique and doesn't clash with other temp files
        # This helps in running parallel instances of prover
        self.random_num = str(random.randint(0, 100000000))
        self.temp_filename_suffix = f"temptodel{self.ticks}{self.random_num}.lean"
        self.temp_file = os.path.join(prefix, self.temp_filename_suffix) if prefix is not None else self.temp_filename_suffix
        self.temp_file_full_path = os.path.join(self.project_root, self.temp_file)
        self.use_hammer = use_hammer
        self.timeout_in_sec = min(timeout_in_sec, 120) # Maximum 120s timeout
        self.current_stmt = None
        self.line_num = 0
        self.main_file_iter = proof_step_iter
        self.suppress_error_log = suppress_error_log
        self.lean_server : LeanCmdServer = None
        self.execution_complete = False
        self._max_memory_in_mib = 40000 # 40 GiB is needed for mathlib to work seemlessly
        self._lines_executed = []
        self.proof_context : ProofContext = None
        self.curr_lemma_name : typing.Optional[str] = None
        self.curr_lemma : typing.Optional[str] = None
        self.lean_error_messages : typing.List[str] = []
        self._proof_running = False
        self._file_content = ""
        self.local_file_lemmas: typing.OrderedDict[str, str] = OrderedDict()
        self.local_theorem_lemma_description: typing.OrderedDict[str, str] = OrderedDict()
        self._proof_start_idx: typing.Optional[int] = None
        self._import_end_idx: typing.Optional[int] = None
        if mathlib_root is not None:
            self._mathlib_root = mathlib_root
        else:
            self._mathlib_root = os.path.join(self.project_root, "_target", "deps", "mathlib")
        self._mathlib_src_root = os.path.join(self._mathlib_root, "src")
        self._enable_search = enable_search
        self._namespaces = namespaces if namespaces is not None else Constants.lean_useful_imports + Constants.mathlib_useful_imports
        if self._enable_search:
            self._search_tool = Lean3Executor._init_search(self._mathlib_root, self._namespaces)
            assert self._search_tool is not None, "Search tool cannot be None"
        else:
            self._search_tool = Lean3SearchTool()

    def _init_search(mathlib_root: str, namespaces: typing.List[str]) -> Lean3SearchTool:
        assert os.path.exists(mathlib_root), f"Mathlib root {mathlib_root} does not exist"
        assert os.path.isdir(mathlib_root), f"Mathlib root {mathlib_root} is not a directory"
        if mathlib_root in Lean3Executor.search_tools:
            search_tool = Lean3Executor.search_tools[mathlib_root]
        else:
            search_tool = Lean3SearchTool(mathlib_root, imports=namespaces)
            search_tool.initialize()
            Lean3Executor.search_tools[mathlib_root] = search_tool
        deep_copy = copy.deepcopy(search_tool)
        return deep_copy

    def __enter__(self):
        self.lean_server = LeanCmdServer(memory_in_mibs=self._max_memory_in_mib, cwd=self.project_root, debug=False)
        if self.main_file_iter is None:
            self.main_file_iter = LeanLineByLineReader(self.main_file).instruction_step_generator()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.main_file_iter.close() # Close the file handle
        except:
            pass
        # delete if the main file is a temporary file
        if os.path.exists(self.temp_file_full_path):
            os.remove(self.temp_file_full_path)

    @property
    def token_separator_set(self):
        return set(" ()[]{}.,;:+-*/=<>!~?@#$%^&|`\"\\")

    @property
    def token_separator(self):
        return " ()[]{}.,;:+-*/=<>!~?@#$%^&|`\"\\"
    
    @property
    def token_separator_regex(self):
        return "\s+|\(|\)|\[|\]|\{|\}|\.|,|;|:|\?|@|#|\$|%|\^|&|\||`|\"|\\\\" +\
            ""#"|\+|-|\*|/|=|<|>|!|~"

    @staticmethod
    def get_token_separator_set():
        return set(" ()[]{}.,;:+-*/=<>!~?@#$%^&|`\"\\")

    @staticmethod
    def get_token_separators():
        return " ()[]{}.,;:+-*/=<>!~?@#$%^&|`\"\\"
    
    @staticmethod
    def get_token_separator_regex():
        return "\s+|\(|\)|\[|\]|\{|\}|\.|,|;|:|\?|@|#|\$|%|\^|&|\||`|\"|\\\\" +\
            ""#"|\+|-|\*|/|=|<|>|!|~"
    
    def is_in_proof_mode(self):
        return True if self.proof_context else False
    
    def needs_qed(self):
        return self.proof_context is not None and len(self.proof_context.all_goals) == 0
    
    def needs_cut_close(self):
        return self.proof_context is not None and len(self.proof_context.fg_goals) == 0 and len(self.proof_context.all_goals) > 0

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
                logger.error(f"Got an exception while running '{stmt}' on lean. File name: {self.main_file}")
                logger.exception(f"Exception Log")
            raise
        self._lines_executed.append(stmt)
        return True

    def run_next_without_exec(self) -> bool:
        try:
            stmt = next(self.main_file_iter)
        except StopIteration:
            self.execution_complete = True
            return False
        self.current_stmt = stmt
        self.line_num += 1
        try:
            self._set_content_to_run(stmt)
        except:
            if not self.suppress_error_log:
                logger.error(f"Got an exception while running '{stmt}' on lean. File name: {self.main_file}")
                logger.exception(f"Exception Log")
            raise
        self._lines_executed.append(stmt)
        return True
    
    def run_all_without_exec(self) -> bool:
        next_cmd_ran = self.run_next_without_exec()
        while next_cmd_ran:
            next_cmd_ran = self.run_next_without_exec()
    
    def find_all_theorems_names(self) -> typing.List[typing.Tuple[str, str]]:
        theorem_names = []
        matches = Lean3Executor.theorem_match.findall(self._file_content)
        for _, _, _, thm_name, _, _ in matches:
            theorem_names.append(thm_name)
        return theorem_names
    
    def get_tokens_in_given_stmt(self, stmt: str, ignore_first_token: bool = False) -> typing.Generator[str, None, None]:
        idx = -1
        for tok in re.split(self.token_separator_regex, stmt):
            idx += 1
            # skip the first token as it is usually a keyword
            if idx == 0 and ignore_first_token:
                continue
            tok1 = tok.strip()
            if len(tok1) > 0:
                yield tok1
    
    def tokenize(stmt: str) -> typing.Generator[str, None, None]:
        for tok in re.split(Lean3Executor.get_token_separator_regex(), stmt):
            tok1 = tok.strip()
            if len(tok1) > 0 and \
            tok1 not in Lean3Executor.keywords and \
            not (len(tok1) == 1 and tok1.isascii() and tok1.isalpha()):
                yield tok1

    # Make this chacheable
    # @functools.lru_cache(maxsize=10000)
    def search_type_matching_defns(self, name: str) -> typing.List[Lean3Lemma]:
        if name in Lean3Executor.keywords:
            return []
        return self._search_tool.lemmas
    
    def get_all_type_matching_defns(self, name: str) -> typing.Generator[Lean3Lemma, None, None]:
        return self.search_type_matching_defns(name)

    def search_exact(self, name: str) -> typing.List[Lean3Lemma]:
        return self.search_type_matching_defns(name)

    def search_defn(self, name: str, match_until: typing.Tuple[str], max_search_res: typing.Optional[int] = None) -> typing.List[typing.Tuple[str, str, bool]]:
        return self.search_type_matching_defns(name)
    
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
    
    def rewind_proof_steps(self) -> str:
        raise NotImplementedError("rewind_proof_steps is not implemented")

    def run_till_next_lemma(self) -> typing.Tuple[bool, typing.Optional[str]]:
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

    def run_till_next_lemma_return_exec_stmt(self) -> typing.Generator[str, None, None]:
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

    def run_to_finish_lemma_return_exec(self) -> typing.Generator[str, None, None]:
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
        
    def get_lemma_name_if_running(self) -> typing.Optional[str]:
        if not self.is_in_proof_mode():
            return None
        else:
            try:
                return self.curr_lemma_name
            except:
                return None
    
    def get_lemma_stmt_if_running(self) -> typing.Optional[str]:
        if not self.is_in_proof_mode():
            return None
        else:
            try:
                return self.local_theorem_lemma_description[self.curr_lemma_name]
            except:
                return None
        
    def get_current_lemma_name(self) -> typing.Optional[str]:
        if self.curr_lemma_name is None:
            return None
        else:
            return self.curr_lemma_name

    def _set_content_to_run(self, stmt: str) -> str:
        # Now add this new line to the context
        if len(self._file_content) > 0:
            # First create a context of all the lines executed so far
            self._file_content += "\n" + stmt.strip()
        else:
            self._file_content = stmt.strip()
        # Remove comments
        self._file_content = Lean3Utils.remove_comments(self._file_content)
    
    def _check_matching_end(self, file_content: str) -> bool:
        # The file_content must end with a matching end
        # TODO: This is a hack, since we are not tokenizing if there are variables which have suffix or prefix
        # of begin or end, then this won't be correct. But this is a rare case, so we can ignore it for now
        if not file_content.endswith("end"):
            return False # no need to check if we are no where near a closing of a proof
        return file_content.count("begin") == file_content.count("end")

    def _run_stmt_on_lean_server(self, idx : int, stmt: str):
        self._set_content_to_run(stmt)
        content = self._file_content
        # Check if the temporary file exists
        if not os.path.exists(self.temp_file_full_path):
            with open(self.temp_file_full_path, "w") as f:
                f.write("")
        # Now add the contents to the file

        if stmt.startswith("theorem") and self._import_end_idx is None:
            self._import_end_idx = idx - 1
        if not self._proof_running:
            last_thm_details = Lean3Executor.theorem_match.findall(content)
        else:
            last_thm_details = []
        if last_thm_details:
            # We might have found a new theorem
            full_thm_stmt, _, _, thm_name, thm_value, _ = last_thm_details[-1]
            full_thm_stmt = full_thm_stmt.strip()
            thm_name = thm_name.strip()
            thm_value = thm_value.strip()
            thm_value = thm_value.lstrip(":")
            if thm_name in self.local_file_lemmas:
                # We have already discovered this theorem
                # The state got added probably because of the end of a proof
                # So we need to remove the state
                self.proof_context = None
                self._proof_running = False
                self.curr_lemma_name, self.curr_lemma = None, None
                self._proof_start_idx = None
            else:
                self.local_theorem_lemma_description[thm_name] = full_thm_stmt
                self.local_file_lemmas[thm_name] = thm_value
                self._proof_running = True
                self.curr_lemma_name, self.curr_lemma = thm_name, thm_value
                self._proof_start_idx = idx
        if self._proof_running:
            content = content.rstrip()
            assert self._proof_start_idx is not None
            matching_end = self._check_matching_end(content)
            if not matching_end:
                content += Lean3Executor.proof_context_generation_tactic
            idx += 1
            with open(self.temp_file_full_path, "w") as f:
                f.write(content)

            timed_out = False
            try:
                response = self.lean_server.run(self.temp_file, self.timeout_in_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.path.exists(self.temp_file_full_path):
                    os.remove(self.temp_file_full_path)
                pass
            except:
                if os.path.exists(self.temp_file_full_path):
                    os.remove(self.temp_file_full_path)
                raise
            
            if not timed_out:
                prev_proof_context = self.proof_context
                self.proof_context = self._parse_proof_context(response.state)

                if len(response.messages) > 0:
                    lines = content.split("\n")
                    self.lean_error_messages = [
                        f"Got {msg.level} in '{lines[msg.line_num - 1][:25]}{'...' if len(lines[msg.line_num - 1]) > 25 else ''}': \n {msg.level}: {msg.text}" for msg in response.messages
                        if msg.line_num <= len(lines) 
                    ]
                else:
                    self.lean_error_messages = []

                if self.proof_context is None and prev_proof_context is not None:
                    if len(response.messages) > 0: # This has to be on the response messages not the error message
                        # Never give up the proof context because of an error
                        self.proof_context = prev_proof_context
                    elif len(response.messages) == 0 and not matching_end:
                        # No more goals
                        self.proof_context = ProofContext.empty()

                # Don't give up the proof context because someone tried to end early
                elif prev_proof_context is not None and self.proof_context is None:
                    # We have finished a proof
                    self._proof_running = False
                    self.curr_lemma_name, self.curr_lemma = None, None
                    self._proof_start_idx = None
            else:
                self.lean_error_messages = ["The tactic timed out, probably because of repeated application of a tactic which created a very big goal."]
                pass
        pass

    def _skip_to_theorem(self, theorem: str):
        found_thm = False
        while not found_thm:
            try:
                stmt = next(self.main_file_iter)
            except StopIteration:
                if not self.suppress_error_log:
                    logger.error(f"Could not find theorem '{theorem}' in the file '{self.main_file}'")
                    raise Exception(f"Could not find theorem '{theorem}' in the file '{self.main_file}'")
                return
            self.current_stmt = stmt
            self.line_num += 1
            file_content = self._file_content
            self._lines_executed = file_content.split("\n")
            idx = len(self._lines_executed)
            if (stmt.startswith("theorem") or stmt.startswith("lemma")) and self._import_end_idx is None:
                self._import_end_idx = idx - 1
            # Now add this new line to the context
            if len(file_content) > 0:
                # First create a context of all the lines executed so far
                file_content += "\n" + stmt.strip()
            else:
                file_content = stmt.strip()
            file_content = Lean3Utils.remove_comments(file_content)
            last_thm_details = Lean3Executor.theorem_match.findall(file_content)
            if last_thm_details:
                # We might have found a new theorem
                full_thm_stmt, _, _, thm_name, thm_value, _ = last_thm_details[-1]
                full_thm_stmt = full_thm_stmt.strip()
                thm_name = thm_name.strip()
                thm_value = thm_value.strip()
                thm_value = thm_value.lstrip(":")
                if thm_name not in self.local_file_lemmas:
                    if theorem != thm_name:
                        self.local_theorem_lemma_description[thm_name] = full_thm_stmt
                        self.local_file_lemmas[thm_name] = thm_value
                    else:
                        found_thm = True
            if found_thm:
                # Capture the proof context
                assert self._import_end_idx is not None
                # Remove all the theorems before the current theorem
                self._lines_executed = self._lines_executed[:self._import_end_idx + 1]
                full_thm_stmts = full_thm_stmt.split("\n")
                self._lines_executed.extend(full_thm_stmts)
                # Reset the file content to completely ignore the previous theorems
                self._file_content = '\n'.join(self._lines_executed)
                # Now change the idx
                idx = len(self._lines_executed)
                self.line_num = idx
                # Remove all the theorems discovered before the current theorem
                self.local_file_lemmas.clear()
                self.local_theorem_lemma_description.clear()
                self._run_stmt_on_lean_server(idx, stmt)
                self._lines_executed.append(stmt)
                self.line_num += 1 # This needs to be reset because the begin was ignored
            else:
                # Now run the lines till the theorem is found
                self._set_content_to_run(stmt)
                self._lines_executed.append(stmt)
        pass
    
    def get_all_theorems(self) -> typing.List[str]:
        self.run_without_executing()

    def _parse_proof_context(self, proof_context_str: str) -> ProofContext:
        if self.use_human_readable_proof_context:
            return self._parse_proof_context_human_readable(proof_context_str)
        else:
            raise NotImplementedError("Parsing of non-human readable proof context is not implemented")
    
    def _parse_proof_context_human_readable(self, proof_context_str: str) -> ProofContext:
        if proof_context_str is None or len(proof_context_str) == 0 or Lean3Executor.proof_context_separator not in proof_context_str:
            return None
        if proof_context_str == "no goals":
            return ProofContext.empty()
        proof_context_str = proof_context_str.strip()
        proof_context_str += "\n\n"
        all_matches = re.findall(Lean3Executor.proof_context_regex, proof_context_str, re.MULTILINE)
        goal_strs = []
        total_goal_cnt = 0
        for _, goal_cnt, goal_str in all_matches:
            if len(goal_cnt) > 0:
               total_goal_cnt = int(goal_cnt)
            goal_str = goal_str.strip()
            goal_strs.append(goal_str)
        if total_goal_cnt > 0:
            assert len(goal_strs) == total_goal_cnt, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
        else:
            assert len(goal_strs) == 1, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
            total_goal_cnt = 1
        assert len(goal_strs) == total_goal_cnt, f"Total goal count {total_goal_cnt} does not match the number of goals {len(goal_strs)}"
        goals = []
        for goal_str in goal_strs:
            goal = self._parse_goal(goal_str)
            goals.append(goal)
        return ProofContext(goals, [], [], [])
    
    def _parse_goal(self, goal_str: str):
        goal_str = goal_str.strip()
        goal = ""
        hyps_goals = re.findall(Lean3Executor.goal_regex, goal_str, re.MULTILINE)
        assert len(hyps_goals) == 1, f"Found more than one goal in the goal string: {goal_str}"
        hypotheses_str, goal = hyps_goals[0]
        hypotheses_str = hypotheses_str.strip()
        goal = goal.strip()
        hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
        # Get rid of all the empty hypotheses
        hypotheses = [hyp for hyp in hypotheses if len(hyp) > 0]
        goal = Obligation(hypotheses, goal)
        return goal

class LeanStdInOutExecutor:
    def __init__(self):
        self.lean_stdin_reader = LeanLineByLineReader()
        self.lean_exec : Lean3Executor = Lean3Executor(
            use_human_readable_proof_context=True, 
            proof_step_iter=self.lean_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.lean_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.lean_exec.__exit__(exc_type, exc_value, traceback)
    
    def run_in_loop(self):
        print("In> ", end="")
        while True:
            try:
                cmd_ran = self.lean_exec.run_next()
                if not cmd_ran:
                    break
                print(f"Lean> {self.lean_exec.current_stmt}")
                print(f"{self.lean_exec.lean_server.proof_context}")
                print("In> ", end="")
            except:
                pass
            pass

class LeanCustomFileExec:
    def __init__(self, file_path: str, project_root: str = '.'):
        self.lean_stdin_reader = LeanLineByLineReader(file_path)
        self.lean_exec : Lean3Executor = Lean3Executor(
            project_root=project_root,
            use_human_readable_proof_context=True, 
            proof_step_iter=self.lean_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.lean_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.lean_exec.__exit__(exc_type, exc_value, traceback)
    
    def run_in_loop(self, opt: str = None):
        print("In> Press 'Enter' for running next line, \n" + 
              "'c' + 'Enter' to cancel the last command and 're-run', \n" +
              "'e' + 'Enter' to run over all lines one-by-one. ", end="")
        last_stmt = None
        run_all = False
        while True:
            try:
                if not run_all:
                    if opt is None or opt != "e":
                        opt = input()
                    if opt == "e":
                        run_all = True
                        print("Running all lines one-by-one")
                elif opt == "c" and last_stmt is not None:
                    if self.lean_exec.is_in_proof_mode():
                        print(f"Goals before cancelling")
                        print(self.lean_exec.proof_context.all_goals)
                    else:
                        print("No goals before cancelling")
                    self.lean_exec.lean_server.cancel_last()
                    if self.lean_exec.is_in_proof_mode():
                        print(f"Goals after cancelling")
                        print(self.lean_exec.proof_context.all_goals)
                    else:
                        print("No goals after cancelling")
                    print(f"Canceled last statement: {last_stmt}")
                    print(f"Re-running: {last_stmt}")
                    self.lean_exec.lean_server.run_stmt(last_stmt)
                    print(f"Lean> Ran {last_stmt} again")
                    continue
                cmd_ran = self.lean_exec.run_next()
                last_stmt = self.lean_exec.current_stmt
                print(f"Lean> {self.lean_exec.current_stmt}")
                if self.lean_exec.is_in_proof_mode():
                    print(f"Goals after running {last_stmt}")
                    print(self.lean_exec.proof_context.all_goals)
                if self.lean_exec.lean_error_messages:
                    for msg_idx, lean_msgs in enumerate(self.lean_exec.lean_error_messages):
                        print(f"Lean [Message {msg_idx}]> {lean_msgs.text}")
                if not cmd_ran:
                    break
                # print(f"{self.lean_exec.proof_context}")
                print("In> ", end="")
            except:
                pass
            pass    

if __name__ == "__main__":
    logging.basicConfig(filename='lean_executor.log', filemode='w', level=logging.INFO)
    os.chdir(root_dir)
    project = "data/test/lean_proj"
    file = "data/test/lean_proj/src/simple.lean"
    with LeanCustomFileExec(file, project) as lean_exec:
        lean_exec.run_in_loop()