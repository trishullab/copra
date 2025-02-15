#!/usr/bin/env python3

import re
import os
import subprocess
import argparse
import sys
from re import Pattern, Match

from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

from sexpdata import Symbol, dumps
from tqdm import tqdm

from .contexts import ProofContext, Obligation, ScrapedTactic
from .util import split_by_char_outside_matching, unwrap, mybarfmt, eprint

def kill_comments(string: str) -> str:
    result = ""
    depth = 0
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if depth == 0:
                result += string[i]
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = False
        else:
            if string[i:i+2] == '(*':
                depth += 1
            if depth == 0:
                result += string[i]
            if string[i-1:i+1] == '*)' and depth > 0:
                depth -= 1
            if string[i] == '"' and string[i-1] != '\\':
                in_quote = True
    return result

def preprocess_command(cmd: str) -> List[str]:
    coq_import_match = re.fullmatch(r"\s*Require\s+Import\s+Coq\.([\w\.'])", cmd)
    if coq_import_match:
        return [f"Require Import {coq_import_match.group(1)}"]

    return [cmd]


def get_stem(tactic: str) -> str:
    return split_tactic(tactic)[0]


def split_tactic(tactic: str) -> Tuple[str, str]:
    tactic = kill_comments(tactic).strip()
    if not tactic:
        return ("", "")
    outer_parens_match = re.fullmatch(r"\((.*)\)\.", tactic)
    if outer_parens_match:
        return split_tactic(outer_parens_match.group(1) + ".")
    if re.match(r"^\s*[-+*\{\}]+\s*$", tactic):
        stripped = tactic.strip()
        return stripped[:-1], stripped[-1]
    if split_by_char_outside_matching(r"\(", r"\)", ";", tactic):
        return tactic, ""
    for prefix in ["try", "now", "repeat", "decide"]:
        prefix_match = re.match(rf"{prefix}\s+(.*)", tactic)
        if prefix_match:
            rest_stem, rest_rest = split_tactic(prefix_match.group(1))
            return prefix + " " + rest_stem, rest_rest
    for special_stem in ["rewrite <-", "rewrite !",
                         "intros until", "simpl in"]:
        special_match = re.match(rf"{special_stem}(:?(:?\s+(.*))|(\.))", tactic)
        if special_match:
            return special_stem, special_match.group(1)
    match = re.match(r"^\(?([\w']+)(\W+.*)?", tactic)
    if not match:
        return tactic, ""
    stem, rest = match.group(1, 2)
    if not rest:
        rest = ""
    return stem, rest


def parse_hyps(hyps_str: str) -> List[str]:
    lets_killed = kill_nested(r"\Wlet\s", r"\sin\s", hyps_str)
    funs_killed = kill_nested(r"\Wfun\s", "=>", lets_killed)
    foralls_killed = kill_nested(r"\Wforall\s", ",", funs_killed)
    fixs_killed = kill_nested(r"\Wfix\s", ":=", foralls_killed)
    structs_killed = kill_nested(r"\W\{\|\s", r"\|\}", fixs_killed)
    hyps_replaced = re.sub(":=.*?:(?!=)", ":", structs_killed, flags=re.DOTALL)
    var_terms = re.findall(r"(\S+(?:, \S+)*) (?::=.*?)?:(?!=)\s.*?",
                           hyps_replaced, flags=re.DOTALL)
    if len(var_terms) == 0:
        return []
    rest_hyps_str = hyps_str
    hyps_list = []
    # Assumes hypothesis are printed in reverse order, because for
    # whatever reason they seem to be.
    for next_term in reversed(var_terms[1:]):
        next_match = rest_hyps_str.rfind(" " + next_term + " :")
        hyp = rest_hyps_str[next_match:].strip()
        rest_hyps_str = rest_hyps_str[:next_match].strip()
        hyps_list.append(hyp)
    hyps_list.append(rest_hyps_str)
    for hyp in hyps_list:
        assert re.search(":(?!=)", hyp) is not None, \
            f"hyp: {hyp}, hyps_str: {hyps_str}\n"\
            f"hyps_list: {hyps_list}\nvar_terms: {var_terms}"
    return hyps_list


def kill_nested(start_string: str, end_string: str, hyps: str) \
        -> str:
    def searchpos(pattern: str, hyps: str, end: bool = False):
        match = re.search(pattern, hyps, flags=re.DOTALL)
        if match:
            if end:
                return match.end()
            return match.start()
        return float("Inf")
    next_forall_pos = searchpos(start_string, hyps)
    next_comma_pos = searchpos(end_string, hyps, end=True)
    forall_depth = 0
    last_forall_position = -1
    cur_position = 0
    while (next_forall_pos != float("Inf") or
           (next_comma_pos != float("Inf") and forall_depth > 0)):
        old_forall_depth = forall_depth
        if next_forall_pos < next_comma_pos:
            cur_position = next_forall_pos
            if forall_depth == 0:
                last_forall_position = next_forall_pos
            forall_depth += 1
        else:
            if forall_depth == 1:
                hyps = hyps[:last_forall_position] + hyps[next_comma_pos:]
                cur_position = last_forall_position
                last_forall_position = -1
            else:
                cur_position = next_comma_pos
            if forall_depth > 0:
                forall_depth -= 1

        new_next_forall_pos = \
            searchpos(start_string, hyps[cur_position+1:]) + cur_position + 1
        new_next_comma_pos = \
            searchpos(end_string, hyps[cur_position+1:], end=True) + \
            cur_position + 1
        assert new_next_forall_pos != next_forall_pos or \
            new_next_comma_pos != next_comma_pos or \
            forall_depth != old_forall_depth, \
            "old start pos was {}, new start pos is {}, old end pos was {},"\
            "new end pos is {}, cur_position is {}"\
            .format(next_forall_pos, new_next_forall_pos, next_comma_pos,
                    new_next_comma_pos, cur_position)
        next_forall_pos = new_next_forall_pos
        next_comma_pos = new_next_comma_pos
    return hyps


def get_var_term_in_hyp(hyp: str) -> str:
    return hyp.partition(":")[0].strip()


hypcolon_regex = re.compile(":(?!=)")


def get_hyp_type(hyp: str) -> str:
    splits = hypcolon_regex.split(hyp, maxsplit=1)
    if len(splits) == 1:
        return ""
    return splits[1].strip()


def get_vars_in_hyps(hyps: List[str]) -> List[str]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [name.strip() for term in var_terms
                 for name in term.split(",")]
    return var_names


def get_indexed_vars_in_hyps(hyps: List[str]) -> List[Tuple[str, int]]:
    var_terms = [get_var_term_in_hyp(hyp) for hyp in hyps]
    var_names = [(name.strip(), hyp_idx)
                 for hyp_idx, term in enumerate(var_terms)
                 for name in term.split(",")]
    return var_names


def get_indexed_vars_dict(hyps: List[str]) -> Dict[str, int]:
    result = {}
    for hyp_var, hyp_idx in get_indexed_vars_in_hyps(hyps):
        if hyp_var not in result:
            result[hyp_var] = hyp_idx
    return result


def get_first_var_in_hyp(hyp: str) -> str:
    return get_var_term_in_hyp(hyp).split(",")[0].strip()



def tacticTakesHypArgs(stem: str) -> bool:
    now_match = re.match(r"\s*now\s+(.*)", stem)
    if now_match:
        return tacticTakesHypArgs(now_match.group(1))
    try_match = re.match(r"\s*try\s+(.*)", stem)
    if try_match:
        return tacticTakesHypArgs(try_match.group(1))
    repeat_match = re.match(r"\s*repeat\s+(.*)", stem)
    if repeat_match:
        return tacticTakesHypArgs(repeat_match.group(1))
    return (
        stem in [
            "apply", "eapply", "eexploit", "exploit",
            "erewrite", "rewrite", "erewrite !", "rewrite !",
            "erewrite <-", "rewrite <-", "destruct", "elim",
            "eelim", "inversion", "monadInv", "pattern",
            "revert", "exact", "eexact", "simpl in",
            "fold", "generalize", "exists", "case",
            "inv", "subst", "specialize"]
    )


def tacticTakesBinderArgs(stem: str) -> bool:
    return stem == "induction"


def tacticTakesIdentifierArg(stem: str) -> bool:
    return stem == "unfold"

normal_lemma_starting_patterns = [
    r"(?:Program\s+)?(?:Polymorphic\s+)?Lemma",
    "Coercion",
    r"(?:Polymorphic\s+)?Theorem",
    "Remark",
    "Proposition",
    r"(?:Polymorphic\s+)?Definition",
    r"Program\s+Definition",
    "Example",
    "Fixpoint",
    "Corollary",
    "Let",
    r"(?<!Declare\s)(?:Polymorphic\s+)?Instance",
    "Function",
    "Property",
    "Fact",
    r"Equations(?:\??)"]
special_lemma_starting_patterns = [
    "Derive",
    "Goal",
    "Add Morphism",
    "Next Obligation",
    r"Obligation\s+\d+",
    "Add Parametric Morphism"]

lemma_starting_patterns = \
    normal_lemma_starting_patterns + special_lemma_starting_patterns


def lemma_name_from_statement(stmt: str) -> str:
    if ("Goal" in stmt or "Obligation" in stmt or "Morphism" in stmt):
        return ""
    stripped_stmt = kill_comments(stmt).strip()
    derive_match = re.fullmatch(
        r"\s*Derive\s+([\w'_]+)\s+SuchThat\s+(.*)\s+As\s+([\w']+)\.\s*",
        stripped_stmt, flags=re.DOTALL)
    if derive_match:
        return derive_match.group(3)
    program_match = re.match(
        r"\s*(?:#\[[^\]]*\]\s*)?Program(?:\s+Instance)?(?:\s+Definition)?\s+"
        r"([\w'\.]*)(.*)",
        stripped_stmt,
        flags=re.DOTALL)
    if program_match:
        return program_match.group(1)
    lemma_match = re.match(
        r"\s*(?:#\[[^\]]*\]\s*)?(?:(?:Local|Global)\s+)?(?:" +
        "|".join(normal_lemma_starting_patterns) +
        r")\s+([\w'\.]*)(.*)",
        stripped_stmt,
        flags=re.DOTALL)
    assert lemma_match, (stripped_stmt, stmt)
    lemma_name = lemma_match.group(1)
    assert ":" not in lemma_name, stripped_stmt
    return lemma_name

symbols_regexp = (r',|(?::>)|(?::(?!=))|(?::=)|\)|\(|;|@\{|~|\+{1,2}|\*{1,2}'
                  r'|&&|\|\||(?<!\\)/(?!\\)|/\\|\\/|(?<![<*+-/|&])=(?!>)|%|'
                  r'(?<!<)-(?!>)|<-|->|<=|>=|<>|\^|\[|\]|(?<!\|)\}|\{(?!\|)')


def get_words(string: str) -> List[str]:
    return [word for word in re.sub(
        r'(\.+|' + symbols_regexp + ')',
        r' \1 ',
        string).split()
            if word.strip() != '']


def get_binder_var(goal: str, binder_idx: int) -> Optional[str]:
    paren_depth = 0
    binders_passed = 0
    skip = False
    forall_match = re.match(r"forall\s+", goal.strip())
    if not forall_match:
        return None
    rest_goal = goal[forall_match.end():]
    for w in get_words(rest_goal):
        if w == "(":
            paren_depth += 1
        elif w == ")":
            paren_depth -= 1
            if paren_depth == 1 or paren_depth == 0:
                skip = False
        elif (paren_depth == 1 or paren_depth == 0) and not skip:
            if w == ":":
                skip = True
            else:
                binders_passed += 1
                if binders_passed == binder_idx:
                    return w
    return None


def normalizeNumericArgs(datum: ScrapedTactic) -> ScrapedTactic:
    numerical_induction_match = re.match(
        r"\s*(induction|destruct)\s+(\d+)\s*\.",
        kill_comments(datum.tactic).strip())
    if numerical_induction_match:
        stem = numerical_induction_match.group(1)
        binder_idx = int(numerical_induction_match.group(2))
        binder_var = get_binder_var(datum.context.fg_goals[0].goal, binder_idx)
        if binder_var:
            newtac = stem + " " + binder_var + "."
            return ScrapedTactic(datum.prev_tactics,
                                 datum.relevant_lemmas,
                                 datum.context, newtac)
    return datum


def parsePPSubgoal(substr: str) -> Obligation:
    split = re.split("\n====+\n", substr)
    assert len(split) == 2, substr
    hypsstr, goal = split
    return Obligation(parse_hyps(hypsstr), goal)

def summarizeObligation(obl: Obligation) -> str:
    hyps_str = ",".join(get_first_var_in_hyp(hyp)
                        for hyp in obl.hypotheses)
    goal_str = re.sub("\n", "\\n", obl.goal)[:100]
    return f"{hyps_str} -> {goal_str}"

def summarizeContext(context: ProofContext,
                     include_background: bool = False,
                     include_all: bool = False) -> None:
    eprint("Foreground:")
    for i, subgoal in enumerate(context.fg_goals):
        eprint(f"S{i}: {summarizeObligation(subgoal)}")
    if not include_background and not include_all:
        return
    if len(context.bg_goals) > 0:
        eprint("Background:")
    for i, subgoal in enumerate(context.bg_goals):
        eprint(f"S{i}: {summarizeObligation(subgoal)}")
    if not include_all:
        return
    if len(context.shelved_goals) > 0:
        eprint("Shelved:")
    for i, subgoal in enumerate(context.shelved_goals):
        eprint(f"S{i}: {summarizeObligation(subgoal)}")
    if len(context.given_up_goals) > 0:
        eprint("Given Up:")
    for i, subgoal in enumerate(context.given_up_goals):
        eprint(f"S{i}: {summarizeObligation(subgoal)}")


def isValidCommand(command: str) -> bool:
    command = kill_comments(command)
    goal_selector_match = re.fullmatch(r"\s*\d+\s*:(.*)", command,
                                       flags=re.DOTALL)
    if goal_selector_match:
        return isValidCommand(goal_selector_match.group(1))
    return ((command.strip()[-1] == "."
             and not re.match(r"\s*{", command))
            or re.fullmatch(r"\s*[-+*{}]*\s*", command) is not None) \
        and (command.count('(') == command.count(')'))


def load_commands_preserve(args: argparse.Namespace, file_idx: int,
                           filename: str) -> List[str]:
    try:
        should_show = args.progress
    except AttributeError:
        should_show = False
    try:
        should_show = should_show or args.read_progress
    except AttributeError:
        pass

    try:
        command_limit = args.command_limit
    except AttributeError:
        command_limit = None
    try:
        text_encoding = args.text_encoding
    except AttributeError:
        text_encoding = 'utf-8'
    return load_commands(filename, max_commands=command_limit,
                         progress_bar=should_show,
                         progress_bar_offset=file_idx * 2,
                         encoding=text_encoding)


def load_commands(filename: str,
                  max_commands: Optional[int] = None,
                  progress_bar: bool = False,
                  progress_bar_offset: Optional[int] = None,
                  encoding: str = 'utf-8') -> List[str]:
    with open(filename, 'r', encoding=encoding) as fin:
        contents = fin.read()
    return read_commands(contents,
                         max_commands=max_commands,
                         progress_bar=progress_bar,
                         progress_bar_offset=progress_bar_offset)


def read_commands(contents: str,
                  max_commands: Optional[int] = None,
                  progress_bar: bool = False,
                  progress_bar_offset: Optional[int] = None) -> List[str]:
    result: List[str] = []
    cur_command = ""
    comment_depth = 0
    in_quote = False
    curPos = 0

    def search_pat(pat: Pattern) -> Tuple[Optional[Match], int]:
        match = pat.search(contents, curPos)
        return match, match.end() if match else len(contents) + 1

    with tqdm(total=len(contents)+1, file=sys.stdout,
              disable=(not progress_bar),
              position=progress_bar_offset,
              desc="Reading file", leave=False,
              dynamic_ncols=True, bar_format=mybarfmt) as pbar:
        while curPos < len(contents) and (max_commands is None or
                                          len(result) < max_commands):
            _, next_quote = search_pat(re.compile(r"\""))
            _, next_open_comment = search_pat(re.compile(r"\(\*"))
            _, next_close_comment = search_pat(re.compile(r"\*\)"))
            _, next_bracket = search_pat(re.compile(r"[\{\}]"))
            next_bullet_match, next_bullet = search_pat(
                re.compile(r"[\+\-\*]+(?![\)\+\-\*])"))
            _, next_period = search_pat(
                re.compile(r"(?<!\.)\.($|\s)|\.\.\.($|\s)"))
            nextPos = min(next_quote,
                          next_open_comment, next_close_comment,
                          next_bracket,
                          next_bullet, next_period)
            assert curPos < nextPos
            next_chunk = contents[curPos:nextPos]
            cur_command += next_chunk
            pbar.update(nextPos - curPos)
            if nextPos == next_quote:
                if comment_depth == 0:
                    in_quote = not in_quote
            elif nextPos == next_open_comment:
                if not in_quote:
                    comment_depth += 1
            elif nextPos == next_close_comment:
                if not in_quote and comment_depth > 0:
                    comment_depth -= 1
            elif nextPos == next_bracket:
                if not in_quote and comment_depth == 0 and \
                   re.match(r"\s*(?:\d+\s*:)?\s*$",
                            kill_comments(cur_command[:-1])):
                    result.append(cur_command)
                    cur_command = ""
            elif nextPos == next_bullet:
                assert next_bullet_match
                match_length = next_bullet_match.end() - \
                    next_bullet_match.start()
                if not in_quote and comment_depth == 0 and \
                   re.match(r"\s*$",
                            kill_comments(cur_command[:-match_length])):
                    result.append(cur_command)
                    cur_command = ""
                assert next_bullet_match.end() >= nextPos
            elif nextPos == next_period:
                if not in_quote and comment_depth == 0:
                    result.append(cur_command)
                    cur_command = ""
            curPos = nextPos
    assert kill_comments(cur_command).strip() == "", \
      "Couldn't parse command list! Are you sure you didn't forget an ending period?" + \
      (contents if len(contents) < 64 else \
       "[too long to print]")
    return result

def get_module_from_filename(filename: Union[Path, str]) -> str:
    return Path(filename).stem


def symbol_matches(full_symbol: str, shorthand_symbol: str) -> bool:
    if full_symbol == shorthand_symbol:
        return True
    return full_symbol.split(".")[-1] == shorthand_symbol


def subgoalSurjective(newsub: Obligation, oldsub: Obligation) -> bool:
    oldhyp_terms = [get_hyp_type(hyp) for hyp in oldsub.hypotheses]
    for newhyp_term in [get_hyp_type(hyp) for hyp in newsub.hypotheses]:
        if newhyp_term not in oldhyp_terms:
            return False
    return newsub.goal == oldsub.goal


def contextSurjective(newcontext: ProofContext, oldcontext: ProofContext):
    for oldsub in oldcontext.all_goals:
        if not any((subgoalSurjective(newsub, oldsub)
                    for newsub in newcontext.all_goals)):
            return False
    return len(newcontext.all_goals) >= len(oldcontext.all_goals)


def lemmas_in_file(filename: str, cmds: List[str],
                   include_proof_relevant: bool = False,
                   disambiguate_goal_stmts: bool = False) \
        -> List[Tuple[str, str]]:
    lemmas: Dict[Tuple[int, str], Optional[str]] = {}
    proof_relevant = False
    in_proof = False
    save_name = None
    for cmd_idx, cmd in reversed(list(enumerate(cmds))):
        if in_proof and possibly_starting_proof(cmd):
            in_proof = False
            proof_relevant = proof_relevant or \
                cmd.strip().startswith("Derive") or \
                cmd.strip().startswith("Equations")
            if not proof_relevant or include_proof_relevant:
                lemmas[(cmd_idx,cmd)] = save_name
        if ending_proof(cmd):
            in_proof = True
            proof_relevant = cmd.strip().rstrip(".") == "Defined"
            named_ending_match = re.match("(?:Save|Defined)\s+(\w+)\.", cmd.strip())
            if named_ending_match:
                save_name = named_ending_match.group(1)
            else:
                save_name = None
    sm_stack = initial_sm_stack(filename)
    full_lemmas = []
    obl_num = 0
    unnamed_goal_num = 0
    last_program_statement = ""
    for cmd_idx, cmd in enumerate(cmds):
        scmd = kill_comments(cmd).strip()
        sm_stack = update_sm_stack(sm_stack, cmd)
        goal_match = re.match(r"\s*Goal\s+(.*)\.$", scmd)
        if re.match(r"\s*Next\s+Obligation\s*\.\s*",
                    scmd):
            assert last_program_statement != ""
            unique_lemma_statement = f"{last_program_statement} Obligation {obl_num}."
            obl_num += 1
        elif goal_match and disambiguate_goal_stmts:
            save_name = lemmas.get((cmd_idx, cmd), None)
            if save_name:
                unique_lemma_statement = f"Theorem {save_name}: {goal_match.group(1)}."
            else:
                if unnamed_goal_num == 0:
                    postfix = ""
                else:
                    postfix = str(unnamed_goal_num-1)
                unique_lemma_statement = \
                    f"Theorem Unnamed_thm{postfix}: {goal_match.group(1)}."
                unnamed_goal_num += 1
        else:
            unique_lemma_statement = cmd
        if re.match(r"\s*(?:(?:Local|Global)\s+)?Program\s+.*", scmd):
            last_program_statement = cmd
            obl_num = 0
        if (cmd_idx, cmd) in lemmas:
            full_lemmas.append((sm_prefix_from_stack(
                sm_stack), unique_lemma_statement))
    return full_lemmas


def let_to_hyp(let_cmd: str) -> str:
    let_match = re.match(r"\s*Let(?:\s+Fixpoint)?\s+(.*)\.\s*$",
                         let_cmd,
                         flags=re.DOTALL)
    assert let_match, "Command passed in isn't a Let!"
    split = split_by_char_outside_matching(r"\(", r"\)", ":=",
                                           let_match.group(1))
    if split:
        name_and_type, _body = split
    else:
        name_and_type = let_match.group(1)

    name_and_prebinders, ty = \
        unwrap(split_by_char_outside_matching(r"\(", r"\)", ":",
                                              name_and_type))
    prebinders_match = re.match(
        r"\s*([\w']*)(.*)",
        name_and_prebinders,
        flags=re.DOTALL)
    assert prebinders_match, \
        f"{name_and_prebinders} doesn't match prebinders pattern"
    name = prebinders_match.group(1)
    prebinders = prebinders_match.group(2)
    if prebinders.strip() != "":
        prebinders = f"forall {prebinders},"

    return f"{name} : {prebinders} {ty[1:]}."


def admit_proof_cmds(lemma_statement: str, ending_statement: str) -> List[str]:
    lemma_statement = kill_comments(lemma_statement)
    let_match = re.fullmatch(r"\s*Let(?:\s+Fixpoint)?\s+(.*)\.\s*$",
                            lemma_statement,
                            flags=re.DOTALL)
    if let_match and ":=" not in lemma_statement:
        admitted_defn = f"Hypothesis {let_to_hyp(lemma_statement)}"
        return ["Abort.", admitted_defn]
    save_match = re.fullmatch(r"\s*Save\s+(.*)\.\s*$",
                              kill_comments(ending_statement),
                              flags=re.DOTALL)
    if save_match:
        goal_match = re.fullmatch(r"\s*Goal\s+(.*)\.\s*$",
                                  lemma_statement, flags=re.DOTALL)
        assert goal_match, f"Didn't start with 'Goal'! lemma_statement is {lemma_statement}"

        admitted_defn = f"Axiom {save_match.group(1)} : {goal_match.group(1)}."
        return ["Abort.", admitted_defn]
    return ["Admitted."]


def set_switch(switch: str) -> None:
    env_string = subprocess.run(f"opam env --switch={switch} --set-switch",
                                shell=True, stdout=subprocess.PIPE, text=True,
                                check=True).stdout

    _setup_opam_env_from_str(env_string)

def setup_opam_env(env_string: str = None) -> None:
    if env_string is None:
        env_string = subprocess.run("opam env", shell=True, stdout=subprocess.PIPE,
                                    check=True, text=True).stdout
    _setup_opam_env_from_str(env_string)


def _setup_opam_env_from_str(env_string: str) -> None:
    for env_line in env_string.splitlines():
        linematch = re.fullmatch(r"(\w*)='([^;]*)'; export (\w*);", env_line)
        assert linematch, env_line
        envvar = linematch.group(1)
        assert envvar == linematch.group(3)
        envval = linematch.group(2)
        os.environ[envvar] = envval

def module_prefix_from_stack(sm_stack: List[Tuple[str, bool]]) -> str:
    return "".join([sm[0] + "." for sm in sm_stack if not sm[1]])

def sm_prefix_from_stack(sm_stack: List[Tuple[str, bool]]) -> str:
    return "".join([sm[0] + "." for sm in sm_stack])

def possibly_starting_proof(command: str) -> bool:
    stripped_command = kill_comments(command).strip()
    pattern = r"(?:#\[(?:.*)\]\s+)?(?:(?:Local|Global)\s+)?(" + "|".join(lemma_starting_patterns) + r")\s*"
    return bool(re.match(pattern,
                         stripped_command, flags=re.DOTALL))


def ending_proof(command: str) -> bool:
    stripped_command = kill_comments(command).strip()
    return (re.match(r"(?:Time\s+)?Qed\s*\.", stripped_command) or
            re.match(r"Defined\s*(?:\S*)?\.", stripped_command) or
            re.match(r"Admitted\s*\.", stripped_command) or
            stripped_command == "Abort." or
            "Save" in stripped_command or
            (re.match(r"\s*Proof\s+\S+\s*", stripped_command) is not None and
             re.match(r"\s*Proof\s+with", stripped_command) is None and
             re.match(r"\s*Proof\s+using", stripped_command) is None))


def initial_sm_stack(filename: str) -> List[Tuple[str, bool]]:
    return [(get_module_from_filename(filename), False)]

def cancel_update_sm_stack(sm_stack: List[Tuple[str, bool]],
                           cmd: str, cmds_before: List[str]) -> List[Tuple[str, bool]]:
    new_stack = list(sm_stack)
    stripped_cmd = kill_comments(cmd).strip()
    module_start_match = re.match(
        r"Module\s+(?:(?:Import|Export)\s+)?(?:Type\s+)?([\w']*)", stripped_cmd)
    if stripped_cmd.count(":=") > stripped_cmd.count("with"):
        module_start_match = None
    section_start_match = re.match(r"Section\s+([\w']*)(?!.*:=)",
                                   stripped_cmd)
    end_match = re.match(r"End\s+([\w']*)\.", stripped_cmd)
    if module_start_match:
        if new_stack and new_stack[-1][0] == module_start_match.group(1):
            new_stack.pop()
        else:
            assert False, \
                f"Unrecognized cancelled Module \"{cmd}\", " \
                f"top of module stack is {new_stack[-1]}"
    elif section_start_match:
        if new_stack and new_stack[-1][0] == section_start_match.group(1):
            new_stack.pop()
        else:
            assert False, \
                f"Unrecognized cancelled Section \"{cmd}\", " \
                f"top of module stack is {new_stack[-1]}"
    elif end_match:
        new_stack = stack_from_commands(sm_stack[0][0] + ".v", cmds_before)
    return new_stack

def update_sm_stack(sm_stack: List[Tuple[str, bool]],
                    cmd: str) -> List[Tuple[str, bool]]:
    new_stack = list(sm_stack)
    stripped_cmd = kill_comments(cmd).strip()
    module_start_match = re.match(
        r"Module\s+(?:(?:Import|Export)\s+)?(?:Type\s+)?([\w']*)", stripped_cmd)
    if stripped_cmd.count(":=") > stripped_cmd.count("with"):
        module_start_match = None
    section_start_match = re.match(r"Section\s+([\w']*)(?!.*:=)",
                                   stripped_cmd)
    end_match = re.match(r"End\s+([\w']*)\.", stripped_cmd)
    reset_match = re.match(r"Reset\s+([\w']*)\.", stripped_cmd)
    if module_start_match:
        new_stack.append((module_start_match.group(1), False))
    elif section_start_match:
        new_stack.append((section_start_match.group(1), True))
    elif end_match:
        if new_stack and new_stack[-1][0] == end_match.group(1):
            new_stack.pop()
        else:
            assert False, \
                f"Unrecognized End \"{cmd}\", " \
                f"module stack is {new_stack}"
    elif reset_match:
        if new_stack and any((item[0] == reset_match.group(1)
                              for item in new_stack)):
            while new_stack[-1][0] != reset_match.group(1):
                new_stack.pop()
            new_stack.pop()
    return new_stack

def stack_from_commands(filename: str, cmds: List[str]) -> List[Tuple[str, bool]]:
    stack = initial_sm_stack(filename)
    for cmd in cmds:
        stack = update_sm_stack(stack, cmd)
    return stack

def update_local_lemmas(local_lemmas: List[Tuple[List[Tuple[str, bool]], str, bool]],
                        sm_stack: List[Tuple[str, bool]], cmd: str) \
        -> List[Tuple[List[Tuple[str, bool]], str, bool]]:
    new_local_lemmas = list(local_lemmas)
    lemmas = lemmas_defined_by_stmt(cmd)
    is_section = "Let" in cmd
    for lemma in lemmas:
        new_local_lemmas.append((sm_stack, lemma, is_section))
    reset_match = re.match(r"Reset\s+(.*)\.", cmd)
    if reset_match:
        reseted_lemma_name = module_prefix + reset_match.group(1)
        for (lemma_sm_stack, lemma, is_section) in list(new_local_lemmas):
            if lemma == ":":
                continue
            lemma_match = re.match(r"\s*([\w'\.]+)\s*:", lemma)
            assert lemma_match, f"{lemma} doesnt match!"
            lemma_name = lemma_match.group(1)
            if lemma_name == reseted_lemma_name:
                new_local_lemmas.remove((lemma_sm_stack, lemma, is_section))
    abort_match = re.match(r"\s*Abort", cmd)
    if abort_match:
        new_local_lemmas.pop()
    end_match = re.match(r"End\s+(.*)\.", cmd)
    if end_match:
        new_local_lemmas = [(lemma_sm_stack, lemma, is_section) for (lemma_sm_stack, lemma, is_section)
                            in new_local_lemmas if not is_section]
    return new_local_lemmas

def lemmas_from_cmds(filename: str, cmds: List[str]) -> List[Tuple[str, bool]]:
    stack = initial_sm_stack(filename)
    lemmas: List[Tuple[List[Tuple[str, bool]], str, bool]] = []
    for cmd in cmds:
        stack = update_sm_stack(stack, cmd)
        lemmas = update_local_lemmas(lemmas, stack, cmd)
    return lemmas


def lemmas_defined_by_stmt(cmd: str) -> List[str]:
    cmd = kill_comments(cmd)
    normal_lemma_match = re.match(
        r"\s*(?:(?:Local|Global)\s+)?(?:" +
        "|".join(normal_lemma_starting_patterns) +
        r")\s+([\w']*)(.*)",
        cmd,
        flags=re.DOTALL)

    if normal_lemma_match:
        lemma_name = normal_lemma_match.group(1)
        binders, body = unwrap(split_by_char_outside_matching(
            r"\(", r"\)", ":", normal_lemma_match.group(2)))
        if binders.strip():
            lemma_statement = (lemma_name +
                               " : forall " + binders + ", " + body[1:])
        else:
            lemma_statement = lemma_name + " " + body
        return [lemma_statement]

    goal_match = re.match(r"\s*(?:Goal)\s+(.*)", cmd, flags=re.DOTALL)

    if goal_match:
        return [": " + goal_match.group(1)]

    morphism_match = re.match(
        r"\s*Add\s+(?:Parametric\s+)?Morphism.*"
        r"with signature(.*)\s+as\s+(\w*)\.",
        cmd, flags=re.DOTALL)
    if morphism_match:
        return [morphism_match.group(2) + " : " + morphism_match.group(1)]

    proposition_match = re.match(r".*Inductive\s*\w+\s*:.*Prop\s*:=(.*)",
                                 cmd, flags=re.DOTALL)
    if proposition_match:
        case_matches = re.finditer(r"\|\s*(\w+\s*:[^|]*)",
                                   proposition_match.group(1))
        constructor_lemmas = [case_match.group(1)
                              for case_match in
                              case_matches]
        return constructor_lemmas
    obligation_match = re.match(".*Obligation", cmd, flags=re.DOTALL)
    if obligation_match:
        return [":"]

    return []

def raise_(ex):
    raise ex
