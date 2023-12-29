#!/usr/bin/env python3

import os
import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing

class IsabelleLineByLineReader:
    def __init__(self, file_name: str = None, file_content: str = None):
        assert file_name is not None or file_content is not None, "Either file_name or file_content must be provided"
        assert file_name is None or file_content is None, "Only one of file_name or file_content must be provided"
        self.file_name : str = file_name
        self.file_content : str = file_content
        if self.file_name is not None:
            with open(file_name, 'r') as fd:
                self.file_content : str = fd.read()
        self.remove_comments()
    
    def remove_comments(self):
        CODE_MODE = 0
        COMMENT_MODE = 1
        idx = 0
        mode = CODE_MODE
        code_without_comments = []
        comment_stack = []
        while idx < len(self.file_content):
            ch = self.file_content[idx]
            if mode == CODE_MODE and ch == '(' and idx + 1 < len(self.file_content) and self.file_content[idx + 1] == '*':
                mode = COMMENT_MODE
                idx += 2
                comment_stack.append("(*")
            elif mode == COMMENT_MODE and ch == '(' and idx + 1 < len(self.file_content) and self.file_content[idx + 1] == '*':
                mode = COMMENT_MODE
                idx += 2
                comment_stack.append("(*") # Start of nested comment
            elif mode == COMMENT_MODE and ch == '*' and idx + 1 < len(self.file_content) and self.file_content[idx + 1] == ')':
                comment_stack.pop()
                if len(comment_stack) == 0:
                    mode = CODE_MODE
                else:
                    assert comment_stack[-1] == "(*", "Comment stack should only contain '(*'"
                    mode = COMMENT_MODE # There is some nested comment still open
                idx += 2
            elif mode == COMMENT_MODE:
                mode = COMMENT_MODE
                idx += 1
            elif mode == CODE_MODE:
                code_without_comments.append(ch)
                idx += 1
            else:
                raise Exception("Unknown mode")
        self.file_content = "".join(code_without_comments)

    def instruction_step_generator(self) -> typing.Iterator[str]:
        lines = self.file_content.split('\n')
        for line in lines:
            if not len(line.strip()) == 0:
                yield line.strip()

class IsabelleStepByStepStdInReader:
    def __init__(self):
        pass

    def instruction_step_generator(self) -> typing.Iterator[str]:
        while True:
            try:
                line = input()
                if line == "(* exit *)":
                    break
                yield line.strip() # Keep going line by line
            except EOFError:
                break

if __name__ == "__main__":
    file_content = """
(*
  Authors: Wenda Li
*)

theory aime_1983_p1 imports Complex_Main
begin

theorem aime_1983_p1:
  fixes x y z w :: nat
  assumes ht : "1 < x \<and> 1 < y \<and> 1 < z"
    and hw : "0 \<le> w"
    and h0 : "ln w / ln x = 24"
    and h1 : "ln w / ln y = 40"
    and h2 : "ln w / ln (x * y * z) = 12"
  shows "ln w / ln z = 60"
proof -
  define xx yy zz ww where "xx=ln x" and "yy = ln y" 
      and "zz = ln z" and "ww = ln w"
  have "xx= ww/24" using h0 ht unfolding xx_def ww_def 
    by (auto simp:field_simps)
  moreover have "yy = ww / 40" using ht h1 unfolding yy_def ww_def
    by (auto simp:field_simps)
  moreover have "xx+yy+zz > 0" 
    unfolding xx_def yy_def zz_def using ht 
    by (metis ln_gt_zero of_nat_1 of_nat_less_iff pos_add_strict)
  then have "ww = 12*(xx+yy+zz)"using ht h2
    unfolding xx_def yy_def zz_def ww_def
    by (auto simp:field_simps ln_mult)
  ultimately have "ww = 12*(ww/24 + ww/40 + zz)"
    by blast
  then have "ww=60*zz"
    by (auto simp:field_simps)
  then show ?thesis unfolding ww_def zz_def using ht by auto
qed

end
"""
    isabelle_reader = IsabelleLineByLineReader(file_content=file_content)
    assert isabelle_reader.file_content is not None
    for idx, instruction in enumerate(isabelle_reader.instruction_step_generator()):
        print(f"[{idx}]: {instruction}")