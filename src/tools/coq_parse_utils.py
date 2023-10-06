#!/usr/bin/env python3

import os
import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing

class CoqLineByLineReader:
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
        CODE_MODE = 0
        CODE_QT_MODE = 1
        all_code = self.file_content
        idx = 0
        mode = CODE_MODE
        instr_start = 0
        special_context_starting_chars = ["+", "-", "*", "{", "}"]
        while idx < len(all_code):
            ch = all_code[idx]
            if mode == CODE_MODE and ch == '.' and idx + 1 < len(all_code) and len(all_code[idx + 1].strip()) == 0:
                code_to_yield = all_code[instr_start: idx + 1].strip()
                contexts = []
                context = ""
                while any([code_to_yield.startswith(ctx) for ctx in special_context_starting_chars]):
                    context += code_to_yield[0]
                    if len(context) >= 2 and context[-2] != context[-1]:
                        # Split the context
                        contexts.append(context[:-1])
                        context = context[-1]
                    code_to_yield = code_to_yield[1:].strip()
                    if len(code_to_yield) == 0:
                        break
                if len(context) > 0:
                    contexts.append(context) # Add the last context if it was missed
                for ctx in contexts:
                    yield ctx.strip()
                code_to_yield = code_to_yield.strip()
                if len(code_to_yield) > 0:
                    yield code_to_yield
                idx += 1
                mode = CODE_MODE
                instr_start = idx
            elif mode == CODE_MODE and ch == '.' and idx + 1 == len(all_code):
                code_to_yield = all_code[instr_start: idx + 1].strip()
                contexts = []
                context = ""
                while any([code_to_yield.startswith(ctx) for ctx in special_context_starting_chars]):
                    context += code_to_yield[0]
                    if len(context) >= 2 and context[-2] != context[-1]:
                        # Split the context
                        contexts.append(context[:-1])
                        context = context[-1]
                    code_to_yield = code_to_yield[1:].strip()
                    if len(code_to_yield) == 0:
                        break
                code_to_yield = code_to_yield.strip()
                if len(context) > 0:
                    contexts.append(context) # Add the last context if it was missed
                for ctx in contexts:
                    yield ctx.strip()
                if len(code_to_yield) > 0:
                    yield code_to_yield
                idx += 1
                mode = CODE_MODE
                instr_start = idx
            elif mode == CODE_MODE and ch == "\"":
                idx += 1
                mode = CODE_QT_MODE
            elif mode == CODE_QT_MODE and ch == "\"" and idx + 1 < len(all_code) and all_code[idx + 1] == "\"":
                idx += 1
                mode = CODE_QT_MODE
            elif mode == CODE_QT_MODE and ch == "\"":
                idx += 1
                mode = CODE_MODE
            elif mode == CODE_MODE:
                mode = CODE_MODE
                idx += 1
            elif mode == CODE_QT_MODE:
                mode = CODE_QT_MODE
                idx += 1
                # instruction start doesn't change
            else:
                raise Exception("This case is not possible")
        if instr_start < len(all_code):
            yield all_code[instr_start:].strip()

class CoqStepByStepStdInReader:
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
{reflexivity. }
"""
    coq_reader = CoqLineByLineReader(file_content=file_content)
    assert coq_reader.file_content is not None
    for idx, instruction in enumerate(coq_reader.instruction_step_generator()):
        print(f"[{idx}]: {instruction}")