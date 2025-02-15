#!/usr/bin/env python3
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import contextlib
import subprocess
import re

from typing import Iterator, List, Optional

from .util import eprint, parseSexpOneLevel
from .contexts import (ScrapedTactic, TacticContext, Obligation,
                       ProofContext, SexpObligation)
from .lsp_backend import main as lsp_main
from .lsp_backend import CoqLSPyInstance
from .serapi_backend import CoqSeraPyInstance
from .coq_util import (kill_comments, preprocess_command, get_stem,
                       split_tactic, parse_hyps, kill_nested,
                       get_var_term_in_hyp, get_hyp_type,
                       get_vars_in_hyps, get_indexed_vars_in_hyps,
                       get_indexed_vars_dict, get_first_var_in_hyp,
                       tacticTakesHypArgs, tacticTakesBinderArgs,
                       tacticTakesIdentifierArg,
                       lemma_name_from_statement, get_words,
                       get_binder_var, normalizeNumericArgs,
                       parsePPSubgoal, summarizeContext, summarizeObligation,
                       isValidCommand, load_commands_preserve,
                       load_commands, read_commands,
                       get_module_from_filename, symbol_matches,
                       subgoalSurjective, contextSurjective,
                       lemmas_in_file, let_to_hyp, admit_proof_cmds,
                       set_switch, setup_opam_env,
                       module_prefix_from_stack, sm_prefix_from_stack,
                       possibly_starting_proof, ending_proof,
                       initial_sm_stack, update_sm_stack,
                       lemmas_defined_by_stmt)
from .coq_agent import TacticHistory, CoqAgent
from .coq_backend import (CoqBackend, CoqExn, BadResponse, AckError,
                          CompletedError, CoqTimeoutError,
                          UnrecognizedError, CoqAnomaly, CoqException,
                          ParseError, NoSuchGoalError, LexError, CoqOverflowError)

def set_parseSexpOneLevel_fn(newfn) -> None:
    global parseSexpOneLevel
    parseSexpOneLevel = newfn

def GetCoqAgent(prelude: str = ".", verbosity: int = 0, set_env: bool = True, use_human_readable_str: bool = False, env_string: str = None, timeout = 60) -> CoqAgent:
    if set_env:
        setup_opam_env(env_string)
    version_string = subprocess.run(["coqc", "--version"], stdout=subprocess.PIPE,
                                    text=True, check=True).stdout
    version_match = re.fullmatch(r"(?:The Coq Proof Assistant, version)? \d+\.(\d+).*", version_string,
                                 flags=re.DOTALL)
    assert version_match, version_string
    minor_version = int(version_match.group(1))
    assert minor_version >= 10, \
            "Versions of Coq before 8.10 are not supported! "\
            f"Currently installed coq is {version_string}"

    backend: CoqBackend
    try:
        if minor_version < 16:
            backend = CoqSeraPyInstance(["sertop", "--implicit"],
                                        timeout=timeout,
                                        set_env=set_env)
            backend.verbosity = verbosity
        else:
            backend = CoqLSPyInstance("coq-lsp", root_dir=prelude, set_env=set_env, timeout=timeout)
        agent = CoqAgent(backend, prelude, verbosity=verbosity, use_human_readable=use_human_readable_str)
    except CoqAnomaly:
        eprint("Anomaly during initialization! Something has gone horribly wrong.")
        raise
    return agent

@contextlib.contextmanager
def CoqContext(prelude: str = ".", verbosity: int = 0, set_env: bool = True) \
        -> Iterator[CoqAgent]:
    if set_env:
        setup_opam_env()
    version_string = subprocess.run(["coqc", "--version"], stdout=subprocess.PIPE,
                                    text=True, check=True).stdout
    version_match = re.fullmatch(r"(?:The Coq Proof Assistant, version)? \d+\.(\d+).*", version_string,
                                 flags=re.DOTALL)
    assert version_match, version_string
    minor_version = int(version_match.group(1))
    assert minor_version >= 10, \
            "Versions of Coq before 8.10 are not supported! "\
            f"Currently installed coq is {version_string}"

    backend: CoqBackend
    try:
        if minor_version < 16:
            backend = CoqSeraPyInstance(["sertop", "--implicit"],
                                        set_env=set_env)
            backend.verbosity = verbosity
        else:
            backend = CoqLSPyInstance("coq-lsp", root_dir=prelude, set_env=set_env)
        agent = CoqAgent(backend, prelude, verbosity=verbosity)
    except CoqAnomaly:
        eprint("Anomaly during initialization! Something has gone horribly wrong.")
        raise

    try:
        yield agent
    finally:
        agent.backend.close()

# Backwards Compatibility (to some extent)
def SerapiInstance(coq_command: List[str], module_name: Optional[str],
                   prelude: str, set_env: bool = True,
                   timeout: int = 30, use_hammer: bool = False,
                   log_outgoing_messages: Optional[str] = None,
                   use_human_readable_str : bool = False) -> CoqAgent:
    del timeout
    del use_hammer
    del log_outgoing_messages
    backend = CoqSeraPyInstance(coq_command, set_env=set_env)
    agent = CoqAgent(backend, prelude, use_human_readable=use_human_readable_str)
    if module_name and module_name not in ["Parameter", "Prop", "Type"]:
        agent.run_stmt(f"Module {module_name}.")
    return agent
@contextlib.contextmanager
def SerapiContext(coq_commands: List[str], module_name: Optional[str],
                  prelude: str, set_env: bool = True, use_hammer: bool = False,
                  log_outgoing_messages: Optional[str] = None) \
                  -> Iterator[CoqAgent]:
    del use_hammer
    del log_outgoing_messages
    try:
        backend = CoqSeraPyInstance(coq_commands, set_env=set_env)
        agent = CoqAgent(backend, prelude)
        if module_name and module_name not in ["Parameter", "Prop", "Type"]:
            agent.run_stmt(f"Module {module_name}.")
    except CoqAnomaly:
        eprint("Anomaly during initialization! Something has gone horribly wrong.")
        raise
    try:
        yield agent
    finally:
        agent.backend.close()
SerapiException = CoqException
def admit_proof(coq: CoqAgent, lemma_statement: str,
                ending_statement: str) -> List[str]:
    admit_cmds = admit_proof_cmds(lemma_statement, ending_statement)
    for cmd in admit_cmds:
        coq.run_stmt(cmd)
    return admit_cmds
