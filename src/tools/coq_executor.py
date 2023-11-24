#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import logging
import typing
import functools
import random
import re
from src.coq_ser_api import SerapiInstance
from src.tools.coq_parse_utils import CoqLineByLineReader, CoqStepByStepStdInReader
logger = logging.getLogger()

class CoqExecutor:
    keywords = {
        "Theorem", "Lemma", "Fact", "Remark", "Corollary", "Proposition", "Example", "Proof", "Qed", "Defined", "Admitted", "Abort",
        "Fixpoint", "CoFixpoint", "Function", "Program Fixpoint", "Program CoFixpoint", "Program Function", "Let", "Let Fixpoint", 
        "Let CoFixpoint", "Let Function", "Let Program Fixpoint", "Let Program CoFixpoint", "Let Program Function",
        "forall", "exists", "fun", "match", "if", "then", "else", "with", "as", "in", "end", "return", "Type", "Set", "Prop",
        "Require", "Import", "Export", "From", "Module", "Section", "End", "Variable", "Axiom", "Parameter", "Hypothesis", "Context",
        "Notation", "Reserved Notation", "Infix", "Notation", "Reserved Notation", "Infix", "Reserved Infix", "Notation", "Reserved Notation", "Definition",
        "intros", "intro", "apply", "assumption", "exact", "reflexivity", "symmetry", "transitivity", "rewrite", "simpl", "unfold", "cbn", "cbv", "compute",
        "destruct", "induction", "inversion", "injection", "split", "exists", "left", "right", "constructor", "auto", "eauto", "tauto", "omega", "lia", "ring",
        "repeat", "try", "assert", "cut", "cutrewrite", "pose", "pose proof", "remember", "set", "setoid_rewrite", "generalize", "generalize dependent",
        "move", "move =>", "move ->", "move => ->", 
        ":", ".", "=>", "{", "}"
    }
    def __init__(self, project_root: str = None, main_file: str = None, use_hammer: bool = False, timeout_in_sec: int = 60, use_human_readable_proof_context: bool = False, proof_step_iter: typing.Iterator[str] = None, suppress_error_log: bool = False):
        assert proof_step_iter is None or isinstance(proof_step_iter, typing.Iterator), \
            "proof_step_iter must be an iterator"
        assert main_file is not None or proof_step_iter is not None, \
            "Either main_file or proof_step_iter must be provided"
        assert main_file is None or proof_step_iter is None, \
            "Only one of main_file or proof_step_iter must be provided"
        assert main_file is None or (os.path.exists(main_file) and main_file.endswith(".v")), \
            "main_file must be a valid path to a '.v' file"
        assert project_root is None or (os.path.exists(project_root) and os.path.isdir(project_root)), \
            "project_root must be a valid path to a directory"
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.project_root = project_root if project_root is not None else "."
        self.main_file = main_file
        self.use_hammer = use_hammer
        self.timeout_in_sec = min(timeout_in_sec, 120) # Maximum 120s timeout
        self.current_stmt = None
        self.line_num = 0
        self.main_file_iter = proof_step_iter
        self.suppress_error_log = suppress_error_log
        self.coq : SerapiInstance = None
        self.execution_complete = False
    
    def __enter__(self):
        self._all_dep_handles = []
        self.coq = SerapiInstance(["sertop", "--implicit"], None, self.project_root,
                             use_hammer=self.use_hammer,
                             log_outgoing_messages=None,
                             timeout=self.timeout_in_sec,
                             use_human_readable_str=self.use_human_readable_proof_context)
        self.coq.quiet = self.suppress_error_log
        if self.main_file_iter is None:
            self.main_file_iter = CoqLineByLineReader(self.main_file).instruction_step_generator()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.main_file_iter.close() # Close the file handle
        except:
            pass
        self.coq.kill() # Kill the coq instance after use

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
        return True if self.coq.proof_context else False
    
    def needs_qed(self):
        return self.coq.proof_context is not None and len(self.coq.proof_context.all_goals) == 0
    
    def needs_cut_close(self):
        return self.coq.proof_context is not None and len(self.coq.proof_context.fg_goals) == 0 and len(self.coq.proof_context.all_goals) > 0

    def run_next(self) -> bool:
        try:
            stmt = next(self.main_file_iter)
        except StopIteration:
            self.execution_complete = True
            return False
        self.current_stmt = stmt
        self.line_num += 1
        try:
            self.coq.run_stmt(stmt, timeout=self.timeout_in_sec)
        except:
            if not self.suppress_error_log:
                logger.error(f"Got an exception while running '{stmt}' on coq. File name: {self.main_file}")
                logger.exception(f"Exception Log")
            raise
        return True
    
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
        for tok in re.split(CoqExecutor.get_token_separator_regex(), stmt):
            tok1 = tok.strip()
            if len(tok1) > 0:
                yield tok1

    @functools.lru_cache(maxsize=10000)
    def print_dfns(self, name: str) -> str:
        if name in CoqExecutor.keywords:
            return ""
        return self.coq.print_symbols(name)

    # Make this chacheable
    @functools.lru_cache(maxsize=10000)
    def search_type_matching_defns(self, name: str) -> typing.List[str]:
        if name in CoqExecutor.keywords:
            return []
        return self.coq.search_about(name)
    
    def get_all_type_matching_defns(self, name: str, should_print_symbol: bool = False) -> typing.Generator[typing.Tuple[str, str], None, None]:
        all_defns = self.search_type_matching_defns(name)
        # Try for an exact match
        for defn in all_defns:
            defn = defn.split(":")
            defn_name = defn[0].strip()
            if len(defn) > 1:
                if should_print_symbol:
                    defn_val = self.print_dfns(defn_name).strip()
                else:
                    defn_val = ("".join(defn[1:])).strip()
            else:
                defn_val = ""
            yield defn_name, defn_val

    def search_exact(self, name: str, should_print_symbol: bool = False) -> typing.List[typing.Tuple[str, str]]:
        symb_defn = self.search_type_matching_defns(name)
        main_matches = []
        match_until = set([name])
        # Try for an exact match
        for defn in symb_defn:
            defn = defn.split(":")
            defn_name = defn[0].strip()
            if len(defn) > 1:
                if should_print_symbol:
                    defn_val = self.print_dfns(defn_name).strip()
                else:                    
                    defn_val = ("".join(defn[1:])).strip()
                # print(f"should_print_symbol: {should_print_symbol}")
                # print(defn_name)
                # print(defn_val)
            else:
                defn_val = ""
            if defn_name in match_until:
                main_matches.append((defn_name, defn_val))
                break
        return main_matches

    def search_defn(self, name: str, match_until: typing.Tuple[str], max_search_res: typing.Optional[int] = None, should_print_symbol: bool = False) -> typing.List[typing.Tuple[str, str, bool]]:
        symb_defn = self.search_type_matching_defns(name)
        match_defns = []
        main_matches = []
        match_until = set(match_until)
        # Try for an exact match
        for defn in symb_defn:
            defn = defn.split(":")
            defn_name = defn[0].strip()
            if len(defn) > 1:
                if should_print_symbol:
                    defn_val = self.print_dfns(defn_name).strip()
                else:
                    defn_val = ("".join(defn[1:])).strip()
            else:
                defn_val = ""
            if defn_name in match_until:
                main_matches.append((defn_name, defn_val, True))
            else:
                match_defns.append((defn_name, defn_val, False))
        if max_search_res is not None:
            match_defns = random.sample(match_defns, max(0, min(max_search_res - 1, len(match_defns))))
        match_defns.extend(main_matches)
        return match_defns
    
    def run_without_executing(self, stmt: str):
        while True:
            try:
                stmt = next(self.main_file_iter)
            except StopIteration:
                return
            self.current_stmt = stmt
            self.line_num += 1

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
        # rewind the proof steps until the last lemma is found
        current_lemma = None
        while self.is_in_proof_mode():
            if current_lemma is None:
                current_lemma = self.coq.cur_lemma
            # If we are already in proof mode, then we have already found a lemma
            # should call run_to_finish_lemma instead
            self.coq.cancel_last(force_update_nonfg_goals=True)
            self.line_num -= 1
        return "Theorem " + current_lemma if current_lemma is not None else None

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
                return self.coq.cur_lemma_name
            except:
                return None
    
    def get_lemma_stmt_if_running(self) -> typing.Optional[str]:
        if not self.is_in_proof_mode():
            return None
        else:
            try:
                return self.coq.cur_lemma
            except:
                return None
    
    def get_current_lemma_name(self) -> typing.Optional[str]:
        if not self.is_in_proof_mode():
            return None
        try:
            return self.coq.cur_lemma_name
        except:
            return None

class CoqStdInOutExecutor:
    def __init__(self):
        self.coq_stdin_reader = CoqStepByStepStdInReader()
        self.coq_exec : CoqExecutor = CoqExecutor(
            use_human_readable_proof_context=True, 
            proof_step_iter=self.coq_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.coq_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.coq_exec.__exit__(exc_type, exc_value, traceback)
    
    def run_in_loop(self):
        print("In> ", end="")
        while True:
            try:
                cmd_ran = self.coq_exec.run_next()
                if not cmd_ran:
                    break
                print(f"Coq> {self.coq_exec.current_stmt}")
                print(f"{self.coq_exec.coq.proof_context}")
                print("In> ", end="")
            except:
                pass
            pass

class CoqCustomFileExec:
    def __init__(self, file_path: str):
        self.coq_stdin_reader = CoqLineByLineReader(file_path)
        self.coq_exec : CoqExecutor = CoqExecutor(
            use_human_readable_proof_context=True, 
            proof_step_iter=self.coq_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.coq_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.coq_exec.__exit__(exc_type, exc_value, traceback)
    
    def run_in_loop(self):
        print("In> Press 'Enter' for running next line and 'c' + 'Enter' to cancel the last command and 're-run'.", end="")
        last_stmt = None
        while True:
            try:
                opt = input()
                if opt == "c" and last_stmt is not None:
                    if self.coq_exec.is_in_proof_mode():
                        print(f"Goals before cancelling")
                        print(self.coq_exec.coq.proof_context.all_goals)
                    else:
                        print("No goals before cancelling")
                    self.coq_exec.coq.cancel_last()
                    if self.coq_exec.is_in_proof_mode():
                        print(f"Goals after cancelling")
                        print(self.coq_exec.coq.proof_context.all_goals)
                    else:
                        print("No goals after cancelling")
                    print(f"Canceled last statement: {last_stmt}")
                    print(f"Re-running: {last_stmt}")
                    self.coq_exec.coq.run_stmt(last_stmt)
                    print(f"Coq> Ran {last_stmt} again")
                    continue
                cmd_ran = self.coq_exec.run_next()
                last_stmt = self.coq_exec.current_stmt
                if self.coq_exec.is_in_proof_mode():
                    print(f"Goals after running {last_stmt}")
                    print(self.coq_exec.coq.proof_context.all_goals)
                if not cmd_ran:
                    break
                print(f"Coq> {self.coq_exec.current_stmt}")
                print(f"{self.coq_exec.coq.proof_context}")
                print("In> ", end="")
            except:
                pass
            pass    

if __name__ == "__main__":
    logging.basicConfig(filename='coq_executor.log', filemode='w', level=logging.INFO)
    # with CoqStdInOutExecutor() as coq_exec:
    #     coq_exec.run_in_loop()
    os.chdir(root_dir)
    with CoqCustomFileExec("data/test/SimpleAlgebra.v") as coq_exec:
        coq_exec.run_in_loop()