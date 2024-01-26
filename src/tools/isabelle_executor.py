#!/usr/bin/env python3

import signal
import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import subprocess
import logging
import typing
import functools
import re
import time
import threading
from collections import OrderedDict
from pathlib import Path
from src.pisa.src.main.python.pisa_client import PisaEnv, initialise_env, IsabelleLemma
from src.rl.proof_action import ProofAction
from src.tools.isabelle_parse_utils import IsabelleLineByLineReader, IsabelleStepByStepStdInReader
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

class IsabelleExecutor:
    # Like Lean, right now this doesn't do the best job of capturing each state changing
    # tactic separately. It relies on the fact the tactics are usually written in an
    # atomic way on separate lines. However for a simple REPL, we can just assume that 
    # the user writes tactics in an atomic way.

    # Keywords list may not be complete
    keywords = {
        "section", "theory", "imports", "begin", "end", "text", "lemma", "theorem", "assume", "assumes", "proof", "have",
        "fix", "show", "shows", "then", "with", "qed", "next", "obtain", "by", "for", "?thesis", "contradiction",
        "datatype", "fun", "where", "subsection", "term", "value", "declare", "using", "case"
    }

    # Matches theorem declarations
    theorem_regex = r"((((theorem\s+|lemma\s+)([\w+|\d+'_]*)))(\s*:\s*)([\S|\s]*))"
    theorem_match = re.compile(theorem_regex, re.MULTILINE)

    # Matches proof context returned by Isabelle engine
    proof_context_regex = r"\s*proof \((state|prove|chain)\)\s*((using )?this:([\s|\S]*?))?goal([\s|\S]*?):\s*([\s|\S]*)"
    proof_context_match = re.compile(proof_context_regex, re.MULTILINE)

    # Matches theory initialization
    begin_theory_regex = r"theory([\s\S]*)imports([\s\S]*)begin"
    begin_theory_match = re.compile(begin_theory_regex, re.MULTILINE)

    # Matches 'assms_x:' in hypotheses
    assms_regex = r"assms_(\d+):"
    assms_regex_match = re.compile(assms_regex)

    # Proof automation tactics: [tactics from LYRA] + [tactics from the *try0* keyword]
    auto_tactics = ["auto", "simp", "blast", "fastforce", "force", "eval", "presburger", "sos", "arith", "linarith", "(auto simp: field_simps)",
                    "metis", "argo", "algebra", "fast", "meson", "satx"]

    def __init__(self, project_root: str = None, main_file: str = None, use_hammer: ProofAction.HammerMode = ProofAction.HammerMode.AUTO, timeout_in_sec: int = 60, 
                 use_human_readable_proof_context: bool = False, proof_step_iter: typing.Iterator[str] = None, 
                 suppress_error_log: bool = False, port: int = 8000):
        assert proof_step_iter is None or isinstance(proof_step_iter, typing.Iterator), \
            "proof_step_iter must be an iterator"
        assert main_file is not None or proof_step_iter is not None, \
            "Either main_file or proof_step_iter must be provided"
        assert (main_file is None) or (project_root is not None), \
            "Project root must be provided for Isabelle, if main file is provided"
        assert main_file is None or (os.path.exists(main_file) and main_file.endswith(".thy")), \
            "main_file must be a valid path to a '.thy' file"
        assert project_root is None or (os.path.exists(project_root) and os.path.isdir(project_root)), \
            "project_root must be a valid path to a directory"
        self.use_human_readable_proof_context = use_human_readable_proof_context
        self.project_root = project_root if project_root is not None else "."
        self.main_file = main_file
        self.use_hammer = use_hammer
        self.timeout_in_sec = min(timeout_in_sec, 120) # Maximum 120s timeout
        self.current_stmt = None
        self.line_num = 0
        self.current_state = 0
        self.line_num_to_state = {}
        self.main_file_iter = proof_step_iter
        self.buffer = ""
        self.suppress_error_log = suppress_error_log
        self.pisa_env : PisaEnv = None
        self.proof_context : ProofContext = None
        self.curr_lemma_name : typing.Optional[str] = None
        self.curr_lemma : typing.Optional[str] = ""
        self._proof_running = False
        self._top_level = True
        self.local_theorem_lemma_description: typing.OrderedDict[str, str] = OrderedDict()
        self.execution_complete = False
        self.global_lemmas = []
        self.port = IsabelleExecutor._port if hasattr(IsabelleExecutor, "_port") else port
        self._seldgehammer_cache : typing.Dict[int,typing.Set[str]] = {}
        home_dir = str(Path.home())
        if os.path.exists(os.path.join(home_dir, "Isabelle2022")):
            self.isa_install_dir = os.path.join(home_dir, "Isabelle2022")
        elif os.path.exists(os.path.join(home_dir, ".local", "bin","Isabelle2022")):
            self.isa_install_dir = os.path.join(home_dir, ".local", "bin","Isabelle2022")
        else:
            raise Exception("Isabelle2022 installation not found. Please install Isabelle2022 and set the path to the installation directory in the environment variable 'ISABELLE_HOME'")
    
    def __enter__(self):
        self._all_dep_handles = []

        if self.main_file_iter is None:
            self.main_file_iter = IsabelleLineByLineReader(self.main_file).instruction_step_generator()

        # PISA clients must provide a file and working directory. If these are not provided,
        # or if the file path is not supported by PISA, use the default header, which may or may not be sufficient.
        if self.main_file is None:
            logger.warning("Initialising Isabelle environment with default theory header and imports (Complex_Main). Pass in a file and project root to import additional theories")
            default_theory_file_path = os.path.join(self.isa_install_dir, "src/HOL/Library/Discrete.thy")
            default_working_directory = os.path.join(self.isa_install_dir, "src/HOL/Library")
            self.pisa_env = initialise_env(port=self.port, isa_path=self.isa_install_dir, theory_file_path=default_theory_file_path, working_directory=default_working_directory)
            self.pisa_env.initialise()
        else:
            try:
                self.pisa_env = initialise_env(port=self.port, theory_file_path=self.main_file, working_directory=self.project_root)
                self.pisa_env.initialise()
            except:
                logger.warning("Theory initialization failed. Most likely this file path is not supported by PISA.")
                logger.warning("Initialising Isabelle environment with default theory header and imports (Complex_Main).")
                self.pisa_env = initialise_env(port=self.port)
                self.pisa_env.initialise()
        
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.main_file_iter.close() # Close the file handle
        except:
            pass

    def start_server(logger : logging.Logger = None, port: int = 8000):
        assert port > 0, "Port number must be greater than 0"
        assert port < 65536, "Port number must be less than 65536"
        jar_path = "src/pisa/target/scala-2.13/PISA-assembly-0.1.jar"
        assert os.path.exists(jar_path), "PISA jar file not found. Please build the project using 'sbt assembly' commnad"
        logger = logger if logger is not None else logging.getLogger('isabelle_pisa_executor')
        cmd = f"java -cp {jar_path} pisa.server.PisaOneStageServer{port}"
        IsabelleExecutor._port = port
        # Start the server in a separate process
        cwd = os.getcwd()
        IsabelleExecutor._server_process = subprocess.Popen(
            cmd, 
            cwd=cwd,
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid) # Create a new process group so that we can kill the process and all its children
        time.sleep(1)
        # Scan the first line
        # Wait for the server to start
        line = IsabelleExecutor._server_process.stdout.readline()
        logger.info(line)
        IsabelleExecutor._process_killed = False
        thread = threading.Thread(target=IsabelleExecutor._server_loggening_thread, args=(logger,))
        thread.start()
        IsabelleExecutor._server_read_thread = thread
        pass

    def _server_loggening_thread(logger : logging.Logger):
        # Keep checking the server is running
        while not IsabelleExecutor._process_killed:
            try:
                line = IsabelleExecutor._server_process.stdout.readline()
                if not line:
                    break
                logger.info(line)
            except:
                logger.info("Stdout is closed")
                time.sleep(1)
        logger.info("Server is shut down")
        time.sleep(1)
        pass

    def stop_server():
        IsabelleExecutor._process_killed = True
        # Kill the server process
        os.killpg(IsabelleExecutor._server_process.pid, signal.SIGTERM)
        # IsabelleExecutor._server_process.kill()
        IsabelleExecutor._server_read_thread.join(5)
        pass

    # The following token separators may not be completely correct
        
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
        # Not supported. Even if there is a nested proof, we still can use `qed` to close the proof
        return False

    def get_state_str(self, state_num):
        if state_num == 0:
            return 'default'
        return 'state' + str(state_num)

    def run_next(self, proof_search_mode=True) -> str:
        try:
            stmt = next(self.main_file_iter)
        except StopIteration:
            self.execution_complete = True
            return ""
        self.current_stmt = stmt
        self.line_num += 1
        try:
            stmt = self._run_stmt_on_isabelle_server(stmt, proof_search_mode)
        except:
            if proof_search_mode:
                if not self.suppress_error_log:
                    logger.error(f"Got an exception while running '{stmt}' on isabelle. File name: {self.main_file}")
                    logger.exception(f"Exception Log")
                raise
            else:
                # If we're not in proof search mode, we can assume any errors are expected
                pass
        return stmt
    
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
        for tok in re.split(IsabelleExecutor.get_token_separator_regex(), stmt):
            tok1 = tok.strip()
            if len(tok1) > 0:
                yield tok1
                
    # Make this chacheable
    # @functools.lru_cache(maxsize=10000)
    def search_type_matching_defns(self, name: str) -> typing.List[IsabelleLemma]:
        if name in IsabelleExecutor.keywords:
            return []
        if not self.global_lemmas:
            assert not self.current_state == 0, 'Search tool cannot be used in top level state (before imports)'
            self.global_lemmas = self.pisa_env.get_global_lemmas(self.get_state_str(self.current_state))
        return self.global_lemmas
    
    def get_all_type_matching_defns(self, name: str) -> typing.Generator[IsabelleLemma, None, None]:
        return self.search_type_matching_defns(name)

    def search_exact(self, name: str) -> typing.List[IsabelleLemma]:
        return self.search_type_matching_defns(name)

    def search_defn(self, name: str, match_until: typing.Tuple[str], max_search_res: typing.Optional[int] = None) -> typing.List[IsabelleLemma]:
        return self.search_type_matching_defns(name)
    
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
                if "qed" in stmt or "sorry" in stmt or "oops" in stmt:
                    return True
            except StopIteration:
                return False
    
    def rewind_proof_steps(self) -> str:
        raise NotImplementedError("rewind_proof_steps is not implemented")
    
    def run_till_next_lemma(self) -> typing.Tuple[bool, typing.Optional[str]]:
        # Run the file until the next lemma is found
        next_stmt = None
        in_proof_mode = self.is_in_proof_mode()
        if in_proof_mode or self.execution_complete:
            # If we are already in proof mode, then we have already found a lemma
            return False, next_stmt
        prev_stmt = self.current_stmt
        ran_last_cmd = self.run_next(proof_search_mode=False)
        next_stmt = self.current_stmt
        if not ran_last_cmd:
            return False, None
        assigned = False
        in_proof_mode = self.is_in_proof_mode()
        while ran_last_cmd and not in_proof_mode:
            if not assigned:
                prev_stmt = next_stmt
            ran_last_cmd = self.run_next(proof_search_mode=False)
            in_proof_mode = self.is_in_proof_mode()
            if not assigned:
                next_stmt = self.current_stmt
                if in_proof_mode:
                    assigned = True
        lemma_name = next_stmt if next_stmt.startswith("theorem") or next_stmt.startswith("lemma") else prev_stmt
        return in_proof_mode, lemma_name

    def run_till_next_lemma_return_exec_stmt(self) -> typing.Generator[str, None, None]:
        # Run the file until the next lemma is found
        next_stmt = None
        in_proof_mode = self.is_in_proof_mode()
        if in_proof_mode or self.execution_complete:
            # If we are already in proof mode, then we have already found a lemma
            yield from []
        else:
            ran_last_cmd = self.run_next(proof_search_mode=False)
            next_stmt = self.current_stmt
            if not ran_last_cmd:
                yield from []
            else:
                yield next_stmt
            in_proof_mode = self.is_in_proof_mode()
            while ran_last_cmd and not in_proof_mode:
                ran_last_cmd = self.run_next(proof_search_mode=False)
                next_stmt = self.current_stmt
                if ran_last_cmd:
                    yield next_stmt
                in_proof_mode = self.is_in_proof_mode()

    def run_to_finish_lemma_return_exec(self) -> typing.Generator[str, None, None]:
        # Run the file until the next lemma is found
        next_stmt = None
        in_proof_mode = self.is_in_proof_mode()
        if not in_proof_mode or self.execution_complete:
            # If we are already in proof mode, then we have already found a lemma
            yield from []
        else:
            ran_last_cmd = self.run_next(proof_search_mode=False)
            next_stmt = self.current_stmt
            if not ran_last_cmd:
                yield from []
            else:
                yield next_stmt
            in_proof_mode = self.is_in_proof_mode()
            while ran_last_cmd and in_proof_mode:
                ran_last_cmd = self.run_next(proof_search_mode=False)
                next_stmt = self.current_stmt
                if ran_last_cmd:
                    yield next_stmt
                in_proof_mode = self.is_in_proof_mode()

    def run_to_finish_lemma(self) -> bool:
        # Run the file and finish the current lemma
        in_proof_mode = self.is_in_proof_mode()
        if not in_proof_mode or self.execution_complete:
            # If we are not in proof mode, then we are not finishing a lemma
            return False
        ran_last_cmd = self.run_next(proof_search_mode=False)
        if not ran_last_cmd:
            return False
        in_proof_mode = self.is_in_proof_mode()
        while ran_last_cmd and in_proof_mode:
            ran_last_cmd = self.run_next(proof_search_mode=False)
            in_proof_mode = self.is_in_proof_mode()
        return not in_proof_mode

    def run_till_line_num(self, line_num: int):
        assert line_num >= self.line_num
        ran_last_cmd = True
        while ran_last_cmd and self.line_num < line_num:
            ran_last_cmd = self.run_next(proof_search_mode=False)
        return self.line_num
    
    def run_to_finish(self):
        ran_last_cmd = True
        while ran_last_cmd:
            ran_last_cmd = self.run_next(proof_search_mode=False)
        
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
        
    def _run_stmt_on_isabelle_server(self, stmt: str, proof_search_mode=True) -> str:
        begin_clause = []
        last_thm_details = []
        start_state = self.get_state_str(self.current_state)
        end_state = self.get_state_str(self.current_state + 1)
        found_lemma = False

        # Deal with multi-line statements: 
        #   1. theory initialization (theory... imports... begin)
        #   2. lemma declarations over multiple lines
        if self._top_level:
            self.buffer += stmt + '\n' # Add current line to buffer
            begin_clause = IsabelleExecutor.begin_theory_match.findall(self.buffer)
        elif not self._proof_running:
            self.buffer += stmt + '\n' # Add current line to buffer
            # If the action succeeds, we have a new lemma
            description = self.pisa_env.step(start_state, self.buffer, end_state)
            if not description.startswith('Step error:'):
                last_thm_details = IsabelleExecutor.theorem_match.findall(self.buffer)

        # Complete initialization transitions found! Exit top-level mode
        if begin_clause:
            # Perform initialization
            stmt = self.buffer
            self.buffer = ""
            self._top_level = False

            start_state = self.get_state_str(0)
            end_state = self.get_state_str(1)

        # Complete lemma found! Enter proof mode
        if last_thm_details:
            found_lemma = True
            # Extract lemma name and declaration
            stmt = self.buffer
            self.buffer = ""
            full_thm_stmt, _, _, _, thm_name, _, thm_value = last_thm_details[-1]
            full_thm_stmt, thm_name, thm_value = full_thm_stmt.strip(), thm_name.strip(), thm_value.strip()

            self.local_theorem_lemma_description[thm_name] = full_thm_stmt
            self.curr_lemma_name, self.curr_lemma = thm_name, thm_value
            self._proof_running = True # Set proof mode

        # In proof mode. Execute tactics
        if self._proof_running or begin_clause:
            stmt = f"{self.buffer}{stmt.strip()}"

            # Throw an error if "sorry" is used, which is not allowed
            # Note that this is a fairly simple/optimistic way of handling it
            if proof_search_mode and "sorry" in stmt:
                raise Exception('Error: expected tactic, got "sorry". Do not use "sorry" in your proof.')
 
            try:
                # Run statement. TODO: pass in timeout
                stmt = self._handle_sledgehammer(start_state, stmt, end_state, proof_search_mode)
                # Parse proof context
                local_hypotheses = self.pisa_env.get_local_lemmas(self.get_state_str(self.current_state + 1))
                proof_state = self.pisa_env.get_state(self.get_state_str(self.current_state + 1))
                self.proof_context = self._parse_proof_context(proof_state, local_hypotheses, found_lemma)
            except Exception as e:
                # If we're not in proof search mode, we assume the file compiles correctly
                # Then this error is likely the result of a tactic split between multiple lines
                # To fix, we'll simply fill a buffer until the tactic compiles correctly
                if not proof_search_mode:
                    self.buffer = stmt + '\n'
                raise

            self.current_state += 1
            self.line_num_to_state[self.line_num] = self.current_state
            self.buffer = ""
            # print(repr(stmt) + "\n -> \n" + repr(description))

            if not begin_clause:
                is_proof_done = self.pisa_env.is_finished(end_state)
                if is_proof_done: # Proof finished
                    self.buffer = ""
                    self._proof_running = False
                    self.curr_lemma_name, self.curr_lemma = None, ""
                    self.proof_context = None
            return stmt
        return "-"

    # PISA only supports sledgehammer as an atomic operation. So we must split any tactic which uses it
    def _handle_sledgehammer(self, start_state: str, step: str, end_state: str, proof_search_mode=True) -> str:
        if step is None or len(step) == 0:
            return None

        tactics = re.split(r'(sledgehammer)', step)
        tactics = list(filter(None, tactics))
        
        description = None
        stmt = ""
        for idx, tactic in enumerate(tactics):
            temp_start = end_state
            temp_end = end_state
            if idx == 0:
                temp_start = start_state
            
            if tactic == 'sledgehammer':
                if proof_search_mode and self.use_hammer == ProofAction.HammerMode.NONE:
                    raise Exception('Error: got "sledgehammer" query with hammer turned off. To use hammer, toggle "use_hammer"')
 
                # Attempt to solve proof with sledgehammer
                description = self._handle_auto_tactics(temp_start, temp_end)
                stmt += description # Replace with hammer-provided tactic, not 'sledgehammer' literally
            else:
                # Run tactic normally
                description = self._handle_reg_tactic(temp_start, tactic, temp_end, proof_search_mode)
                stmt += description # Replace with a potentially modified tactic

        return stmt
    
    def _handle_auto_tactics(self, start_state: str, end_state: str) -> str:
        # First we'll try easier tactics, e.g. "simp", "auto", "blast", etc.
        for tactic in IsabelleExecutor.auto_tactics:
            stmt = 'by ' + tactic
            description = self.pisa_env.step(start_state, stmt, end_state)
            if not description.startswith('Step error:'):
                return stmt + ' <auto tactic>'

        # If those fail, run sledgehammer (more powerful but slower)
        description = self.pisa_env.apply_hammer(start_state, end_state)
        if description.startswith('Step error:'):
            raise Exception(description)
        return description.split('<hammer>')[0] + '<hammer>'

    def _handle_reg_tactic(self, start_state: str, step: str, end_state: str, proof_search_mode=True) -> str:
        description = self.pisa_env.step(start_state, step, end_state)
        if not description.startswith('Step error:'):
            return step
        
        tactics = re.split(r'\susing\s|\sby\s', step, maxsplit=1)
        if len(tactics) > 1 and self.use_hammer == ProofAction.HammerMode.AUTO and proof_search_mode:
            # Try applying sledgehammer. We do some awkward parsing to apply it to the correct portion
            # This will not always work, but because this is a heuristic and not mission-critical, it is ok
            new_tactic = tactics[0] + ' sledgehammer'
            # Check if we've already tried this tactic
            if self.current_state in self._seldgehammer_cache and new_tactic in self._seldgehammer_cache[self.current_state]:
                raise Exception(description)
            try:
                step = self._handle_sledgehammer(start_state, new_tactic, end_state, proof_search_mode)
                return step
            except:
                # Don't throw an error here -- we want to throw the original error
                pass
            # Add the tactic to cache
            if self.current_state not in self._seldgehammer_cache:
                self._seldgehammer_cache[self.current_state] = set()
            self._seldgehammer_cache[self.current_state].add(new_tactic)
        raise Exception(description)

    def _parse_proof_context(self, proof_context_str: str, local_hypotheses: typing.List[IsabelleLemma], found_lemma: bool) -> ProofContext:
        if proof_context_str is None or len(proof_context_str) == 0:
            return None
        
        # Parse proof context and find 1. last proved statement (this) and 2. goals
        all_matches = self.proof_context_match.findall(proof_context_str)
        if len(all_matches) == 0:
            return None
        
        context_type, _, _, this_hyps_str, _, goals_str = all_matches[0]
        this_hyps_str, goals_str = this_hyps_str.strip(), goals_str.strip()
        if(goals_str == "No subgoals!"):
            return ProofContext.empty()

        if not found_lemma and not context_type == 'state':
            raise Exception(f'Error: please provide a full tactic. This step ends in "{context_type}" mode but it should end in "state" mode')

        hypotheses = []
        for hyp in local_hypotheses:
            # Replace assms_x with assms(x)
            hypotheses.append(IsabelleExecutor.assms_regex_match.sub(r'assms(\1):', hyp.dfn))

        goals_list = list(filter(None, goals_str.split("\n")))
        goals = []
        for i, goal_str in enumerate(goals_list):
            goal_str = re.sub("\d+.", "", goal_str, 1).strip() # Remove numbering
            if i == 0:
                goal = Obligation(hypotheses, goal_str)
            else:
                goal = Obligation([], goal_str)
            goals.append(goal)
        
        return ProofContext(goals, [], [], [])


class IsabelleStdInOutExecutor:
    def __init__(self):
        self.isabelle_stdin_reader = IsabelleStepByStepStdInReader()
        self.isabelle_exec : IsabelleExecutor = IsabelleExecutor(
            use_human_readable_proof_context=True, 
            proof_step_iter=self.isabelle_stdin_reader.instruction_step_generator())
    
    def __enter__(self):
        self.isabelle_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.isabelle_exec.__exit__(exc_type, exc_value, traceback)
    
    def run_in_loop(self):
        print("In> ", end="")
        while True:
            try:
                cmd_ran = self.isabelle_exec.run_next()
                if not cmd_ran:
                    break
                print(f"Isabelle> {cmd_ran}")
                print(f"{self.isabelle_exec.proof_context}")
                print("In> ", end="")
            except Exception as e:
                print(e)
                pass
            pass

class IsabelleCustomFileExec:
    def __init__(self, file_path: str, project_root: str):
        self.isabelle_exec : IsabelleExecutor = IsabelleExecutor(
            use_human_readable_proof_context=True, 
            main_file=file_path,
            project_root=project_root)
            
    def __enter__(self):
        self.isabelle_exec.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.isabelle_exec.__exit__(exc_type, exc_value, traceback)
    
    def run_in_loop(self):
        print("In> Press 'Enter' for running next line and 'c' + 'Enter' to cancel the last command and 're-run'.", end="")
        cmd_ran = None
        while True:
            try:
                opt = input()
                if opt == "c" and cmd_ran is not None:
                    if self.isabelle_exec.is_in_proof_mode():
                        print(f"Goals before cancelling")
                        print(self.isabelle_exec.proof_context.all_goals)
                    else:
                        print("No goals before cancelling")
                    print("cancel_last() not implemented")
                    if self.isabelle_exec.is_in_proof_mode():
                        print(f"Goals after cancelling")
                        print(self.isabelle_exec.proof_context.all_goals)
                    else:
                        print("No goals after cancelling")
                    print(f"Canceled last statement: {cmd_ran}")
                    print(f"Re-running: {cmd_ran}")
                    print("re-running not implemented")
                    print(f"Isabelle> Ran {cmd_ran} again")
                    continue
                cmd_ran = self.isabelle_exec.run_next(proof_search_mode=False)
                if not cmd_ran:
                    break
                print(f"Isabelle> {self.isabelle_exec.current_stmt}")
                print(f"Parsed Tactic> {cmd_ran}")
                print(f"{self.isabelle_exec.proof_context}")
                print("In> ", end="")
            except Exception as e:
                print(e)
                pass
            pass    

if __name__ == "__main__":
    logging.basicConfig(filename='isabelle_executor.log', filemode='w', level=logging.INFO)
    # with IsabelleStdInOutExecutor() as isabelle_exec:
        # isabelle_exec.run_in_loop()

    # os.chdir(root_dir)
    # with IsabelleCustomFileExec("data/benchmarks/miniF2F/isabelle/test/aime_1983_p1.thy", "data/benchmarks/miniF2F") as isabelle_exec:
    #     isabelle_exec.run_in_loop()

    os.chdir(root_dir)
    IsabelleExecutor.start_server(port=17000)
    try:
        with IsabelleCustomFileExec("data/test/SimpleAlgebra.thy", "data/test") as isabelle_exec:
            isabelle_exec.run_in_loop()
    finally:
        IsabelleExecutor.stop_server()