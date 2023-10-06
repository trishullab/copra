#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import typing
from subprocess import Popen, PIPE, STDOUT
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Message:
    level: str
    file_name: str
    line_num: int
    column_num: int
    text: str

@dataclass_json
@dataclass
class LeanCmdServerResponse:
    state: typing.Optional[str] = None
    messages: typing.List[Message] = field(default_factory=list)

EmptyResponse = LeanCmdServerResponse()

class LeanCmdServer:
    has_state_message = 'tactic failed, there are unsolved goals\nstate:'
    def __init__(self, memory_in_mibs = 40000, cwd = '.', debug=False):
        assert cwd is not None, "cwd must be provided"
        assert os.path.isdir(cwd), "cwd must be a valid directory"
        self.memory_in_mibs = memory_in_mibs
        self.debug = debug
        self.cwd = cwd
        self.process = None

    def run(self, filepath: str, timeout_in_secs: float = 120.0):
        full_path = os.path.join(self.cwd, filepath)
        assert os.path.isfile(full_path), f"filepath must be a valid file: {filepath}"
        lean_cmd = f'lean --memory={self.memory_in_mibs} {filepath}'
        self.process = Popen(
            lean_cmd, 
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = self.cwd, 
            bufsize = 1, 
            universal_newlines = True)
        # Start the process, and wait for it to finish
        self.process.wait(timeout=timeout_in_secs)
        # Get the output
        output = self.process.stdout.read()
        # Kill the process if it is still running
        self.process.kill()
        # Return the output
        if len(output) == 0:
            return EmptyResponse
        else:
            return self.parse_output(full_path, output) 
    
    def parse_output(self, full_path: str, output: str):
        # AbsFilePath:Line:Column: [waring|error]: Message
        # First get absolute path from full path
        abs_path = os.path.abspath(full_path) + ':'
        messages = output.split(abs_path)
        messages = [msg for msg in messages if len(msg) > 0] # Remove empty strings
        final_messages : typing.List[Message] = []
        state = None
        msg_unparsed = []
        for msg in messages:
            # Get rid of line number and column number
            try:
                line_num_str, col_num_str, level_str, text = msg.split(':', 3)
                line_num_str = line_num_str.strip()
                col_num_str = col_num_str.strip()
                level_str = level_str.strip()
                text = text.strip()
                line_num = int(line_num_str)
                col_num = int(col_num_str)
                level = level_str.lower()
                if level == 'error' and text.startswith(LeanCmdServer.has_state_message):
                    state = text[len(LeanCmdServer.has_state_message):]
                else:
                    final_messages.append(Message(level, full_path, line_num, col_num, text))
            except:
                msg_unparsed.append(msg)
                pass
        if len(final_messages) > 0:
            # Sort messages by line number
            final_messages.sort(key=lambda msg: msg.line_num)
        last_line_num = 0 if len(final_messages) == 0 else final_messages[-1].line_num
        # Now add the unparsed messages
        for msg in msg_unparsed:
            final_messages.append(Message('info', full_path, last_line_num, 0, msg))
        # re-sort
        final_messages.sort(key=lambda msg: msg.line_num)
        return LeanCmdServerResponse(state, final_messages)

if __name__ == "__main__":
    os.chdir(root_dir)
    cwd = 'data/test/lean_proj'
    path = 'src/simple.lean'
    server = LeanCmdServer(cwd=cwd)
    output = server.run(path)
    print(output)