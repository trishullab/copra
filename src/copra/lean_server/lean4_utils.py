#!/usr/bin/env python3

import os
import re
import typing
from copra.lean_server.lean_context import Obligation, ProofContext

class Lean4Utils:
    lean_internal_lib_cmd = "elan which lean"
    theorem_lemma_search_regex = re.compile(r"(theorem|lemma) ([\w+|\d+]*) ([\S|\s]*?):=")
    proof_context_separator = "⊢"
    proof_context_regex = r"((\d+) goals)*((case [\S]+(\n|\n\n))*[\s|\S]*?)\n\n"
    goal_regex = rf"([\s|\S]*?){proof_context_separator}([\s|\S]*)"

    def remove_comments(text: str) -> str:
        # Remove comments
        #1. First remove all nested comments
        #2. Then remove all single line comments
        # Comments are of the form:
        # 1. /- ... -/
        # 2. -- ...
        # Let's do 1
        # First, let's find all the comments
        new_text = []
        idx = 0
        while idx < len(text):
            if idx < len(text) - 1 and text[idx] == '/' and text[idx+1] == '-':
                # We found a comment
                # Find the end of the comment
                end_of_comment_idx = idx + 2
                while end_of_comment_idx < len(text) and \
                    not (text[end_of_comment_idx] == '-' and \
                    end_of_comment_idx + 1 < len(text) and \
                    text[end_of_comment_idx + 1] == '/'):
                    end_of_comment_idx += 1
                if end_of_comment_idx >= len(text):
                    # Unfinished comment
                    new_text.extend(text[idx:end_of_comment_idx])
                    idx = end_of_comment_idx
                else:
                    # Remove the comment
                    idx = end_of_comment_idx + 2
            if idx < len(text):
                new_text.append(text[idx])
                idx += 1
        text = "".join(new_text)
        new_text = []
        # Now let's do 2
        idx = 0
        while idx < len(text):
            if idx < len(text) - 1 and text[idx] == '-' and text[idx+1] == '-':
                # We found a comment
                # Find the end of the comment
                end_of_comment_idx = idx + 2
                while end_of_comment_idx < len(text) and text[end_of_comment_idx] != '\n':
                    end_of_comment_idx += 1
                if end_of_comment_idx >= len(text):
                    # Unfinished comment
                    new_text.extend(text[idx:end_of_comment_idx])
                # Remove the comment
                idx = end_of_comment_idx
            if idx < len(text):
                new_text.append(text[idx])
                idx += 1
        text = "".join(new_text)
        return text
    
    def get_lean_root_path() -> str:
        lean_exe = os.popen(Lean4Utils.lean_internal_lib_cmd).read().strip()
        lean_bin_path = os.path.dirname(lean_exe)
        lean_root_path = os.path.dirname(lean_bin_path)
        return lean_root_path

    def get_lean_lib_path() -> str:
        lean_root_path = Lean4Utils.get_lean_root_path()
        lean_lib_path = os.path.join(lean_root_path, "lib", "lean", "library")
        return lean_lib_path
    
    def find_theorems_with_namespaces(text: str) -> typing.List[typing.Tuple[str, str, str]]:
        idx = 0
        theorems = []
        lines = text.split('\n')
        current_namespace = None
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("namespace"):
                current_namespace = line[len("namespace"):].strip()
                # Find the end of the namespace
                end_of_namespace_idx = idx + 1
                end_line = lines[end_of_namespace_idx]
                while end_line is not None and not end_line.startswith("end") and not end_line.endswith(current_namespace):
                    end_of_namespace_idx += 1
                    end_line = lines[end_of_namespace_idx] if end_of_namespace_idx < len(lines) else None
                if end_line is not None:
                    namespace_content = " ".join(lines[idx:end_of_namespace_idx+1])
                    for name, dfn in Lean4Utils.find_theorems(namespace_content):
                        theorems.append((current_namespace, name, dfn))
                idx = end_of_namespace_idx + 1
            else:
                idx += 1
        return theorems
    
    def find_theorems(text: str) -> typing.List[typing.Tuple[str, str]]:
        matches = Lean4Utils.theorem_lemma_search_regex.findall(text)
        theorems = []
        for match in matches:
            name = str(match[1]).strip()
            dfn = str(match[2]).strip()
            name = name.strip(':')
            dfn = dfn.strip(':')
            theorems.append((name, dfn))
        return theorems

    def parse_proof_context_human_readable(proof_context_str: str) -> ProofContext:
        if len(proof_context_str) == 0 and Lean4Utils.proof_context_separator not in proof_context_str:
            return None
        if proof_context_str == "no goals":
            return ProofContext.empty()
        proof_context_str = proof_context_str.strip()
        proof_context_str += "\n\n"
        all_matches = re.findall(Lean4Utils.proof_context_regex, proof_context_str, re.MULTILINE)
        goal_strs = []
        for match in all_matches:
            goal_str = match[2]
            goal_str = goal_str.strip()
            goal_strs.append(goal_str)
        goals = []
        # try:
        for goal_str in goal_strs:
            goal = Lean4Utils.parse_goal(goal_str)
            goals.append(goal)
        # except Exception as e:
        #     print(f"proof_context_str:\n {proof_context_str}")
        #     print(f"goal_strs:\n {goal_strs}")
        #     raise
        return ProofContext(goals, [], [], [])

    def parse_proof_context_human_readable_as_goals(proof_context_str: str) -> typing.List[Obligation]:
        if len(proof_context_str) == 0 and Lean4Utils.proof_context_separator not in proof_context_str:
            return None
        if proof_context_str == "no goals":
            return ProofContext.empty()
        proof_context_str = proof_context_str.strip()
        proof_context_str += "\n\n"
        all_matches = re.findall(Lean4Utils.proof_context_regex, proof_context_str, re.MULTILINE)
        goal_strs = []
        for match in all_matches:
            goal_str = match[2]
            goal_str = goal_str.strip()
            goal_strs.append(goal_str)
        goals = []
        # try:
        for goal_str in goal_strs:
            goal = Lean4Utils.parse_goal(goal_str)
            goals.append(goal)
        # except Exception as e:
        #     print(f"proof_context_str:\n {proof_context_str}")
        #     print(f"goal_strs:\n {goal_strs}")
        #     raise
        return goals

    def parse_goal(goal_str: str):
        goal_str = goal_str.strip()
        goal = ""
        hyps_goals = re.findall(Lean4Utils.goal_regex, goal_str, re.MULTILINE)
        assert len(hyps_goals) <= 1, f"Found more than one goal in the goal string: {goal_str}"
        if len(hyps_goals) == 1:
            hypotheses_str, goal = hyps_goals[0]
            hypotheses_str = hypotheses_str.strip()
            goal = goal.strip()
            hypotheses = [hyp.rstrip(',') for hyp in hypotheses_str.split("\n")]
            goal = Obligation(hypotheses, goal)
            return goal
        else:
            # Everyline except the last one is a hypothesis
            hypotheses = goal_str.split("\n")
            goal = hypotheses[-1]
            hypotheses = hypotheses[:-1]
            goal = Obligation(hypotheses, goal)
            return goal

if __name__ == '__main__':
    goals = "case intro.inl\n\u03b1 : Type u_1\n\u03b2 : Type u_2\n\u03b3 : Type u_3\nr r' : \u03b1 \u2192 \u03b1 \u2192 Prop\ninst\u271d\u00b9 : LinearOrder \u03b2\nh : WellFounded fun x x_1 => x < x_1\ninst\u271d : PartialOrder \u03b3\nf g : \u03b2 \u2192 \u03b3\nhf : StrictMono f\nhg : StrictMono g\nhfg : range f = range g\nb : \u03b2\nH : \u2200 a < b, f a = g a\nc : \u03b2\nhc : f c = g b\nhcb : c < b\n\u22a2 f b \u2264 g b\n\ncase intro.inr\n\u03b1 : Type u_1\n\u03b2 : Type u_2\n\u03b3 : Type u_3\nr r' : \u03b1 \u2192 \u03b1 \u2192 Prop\ninst\u271d\u00b9 : LinearOrder \u03b2\nh : WellFounded fun x x_1 => x < x_1\ninst\u271d : PartialOrder \u03b3\nf g : \u03b2 \u2192 \u03b3\nhf : StrictMono f\nhg : StrictMono g\nhfg : range f = range g\nb : \u03b2\nH : \u2200 a < b, f a = g a\nc : \u03b2\nhc : f c = g b\nhbc : b \u2264 c\n\u22a2 f b \u2264 g b"
    goals = """case inl

⊢ Λ 1 ≠ 0 ↔ IsPrimePow 1

case inr
n : ℕ
hn : n ≠ 1
⊢ Λ n ≠ 0 ↔ IsPrimePow n
"""
    print("Testing Lean4Utils.parse_proof_context_human_readable")
    print(f"Goals:\n{goals}")
    proof_context = Lean4Utils.parse_proof_context_human_readable(goals)
    print(proof_context)
    text = """
    -- This is a comment
    /- This is a nested comment -/
    theorem foo : 1 = 1 := rfl
    lemma bar : 2 = 2 := rfl
    """
    print("Testing Lean4Utils")
    print("-"*20)
    print("Testing Lean4Utils.remove_comments")
    print("Before:")
    print(text)
    print("After:")
    print(Lean4Utils.remove_comments(text))
    print("-"*20)
    print("Testing Lean3Utils.get_lean_root_path")
    print(Lean4Utils.get_lean_root_path())
    print("-"*20)
    print("Testing Lean3Utils.get_lean_lib_path")
    print(Lean4Utils.get_lean_lib_path())
    print("-"*20)
    print("Testing Lean3Utils.find_theorems")
    lib_path = Lean4Utils.get_lean_lib_path()
    init_data_name = "init/data/array/basic.lean"
    init_data_path = os.path.join(lib_path, init_data_name)
    with open(init_data_path, 'r') as f:
        text = f.read()
    print(f"Testing Lean3Utils.find_theorems on {init_data_name}")
    for name, dfn in Lean4Utils.find_theorems(text):
        print(f"{name}: {dfn}")
    print("-"*20)
    print("Testing Lean3Utils.find_theorems_with_namespaces")
    for namespace, name, dfn in Lean4Utils.find_theorems_with_namespaces(text):
        print(f"[{namespace}]::{name}: {dfn}")
    print("-"*20)
    # print(text)

    # print(Lean3Utils.find_theorems(text))
    # print(Lean3Utils.find_theorems_with_namespaces(text))