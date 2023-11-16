#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os

class Lean3Utils:
    lean_internal_lib_cmd = "elan which lean"
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