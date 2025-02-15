#!/usr/bin/env python3

import os
import re
import typing

class Lean3Utils:
    lean_internal_lib_cmd = "elan which lean"
    theorem_lemma_search_regex = re.compile(r"(theorem|lemma) ([\w+|\d+]*) ([\S|\s]*?):=")
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
        lean_exe = os.popen(Lean3Utils.lean_internal_lib_cmd).read().strip()
        lean_bin_path = os.path.dirname(lean_exe)
        lean_root_path = os.path.dirname(lean_bin_path)
        return lean_root_path

    def get_lean_lib_path() -> str:
        lean_root_path = Lean3Utils.get_lean_root_path()
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
                    for name, dfn in Lean3Utils.find_theorems(namespace_content):
                        theorems.append((current_namespace, name, dfn))
                idx = end_of_namespace_idx + 1
            else:
                idx += 1
        return theorems
    
    def find_theorems(text: str) -> typing.List[typing.Tuple[str, str]]:
        matches = Lean3Utils.theorem_lemma_search_regex.findall(text)
        theorems = []
        for match in matches:
            name = str(match[1]).strip()
            dfn = str(match[2]).strip()
            name = name.strip(':')
            dfn = dfn.strip(':')
            theorems.append((name, dfn))
        return theorems

if __name__ == '__main__':
    text = """
    -- This is a comment
    /- This is a nested comment -/
    theorem foo : 1 = 1 := rfl
    lemma bar : 2 = 2 := rfl
    """
    print("Testing Lean3Utils")
    print("-"*20)
    print("Testing Lean3Utils.remove_comments")
    print("Before:")
    print(text)
    print("After:")
    print(Lean3Utils.remove_comments(text))
    print("-"*20)
    print("Testing Lean3Utils.get_lean_root_path")
    print(Lean3Utils.get_lean_root_path())
    print("-"*20)
    print("Testing Lean3Utils.get_lean_lib_path")
    print(Lean3Utils.get_lean_lib_path())
    print("-"*20)
    print("Testing Lean3Utils.find_theorems")
    lib_path = Lean3Utils.get_lean_lib_path()
    init_data_name = "init/data/array/basic.lean"
    init_data_path = os.path.join(lib_path, init_data_name)
    with open(init_data_path, 'r') as f:
        text = f.read()
    print(f"Testing Lean3Utils.find_theorems on {init_data_name}")
    for name, dfn in Lean3Utils.find_theorems(text):
        print(f"{name}: {dfn}")
    print("-"*20)
    print("Testing Lean3Utils.find_theorems_with_namespaces")
    for namespace, name, dfn in Lean3Utils.find_theorems_with_namespaces(text):
        print(f"[{namespace}]::{name}: {dfn}")
    print("-"*20)
    # print(text)

    # print(Lean3Utils.find_theorems(text))
    # print(Lean3Utils.find_theorems_with_namespaces(text))