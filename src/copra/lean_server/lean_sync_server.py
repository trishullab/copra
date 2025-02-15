#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
from subprocess import Popen, PIPE, STDOUT
from threading import Event
import threading
import time
from typing import Optional, List

from src.lean_server.commands import (SyncRequest, InfoRequest,
                                  Request, CommandResponse, Message, Task,
                                  InfoResponse, AllMessagesResponse, CurrentTasksResponse, ErrorResponse,
                                  OkResponse, SyncResponse)

class SyncLeanServer:
    def __init__(self, lean_cmd='lean --server --memory=40000', cwd = '.', debug=False, debug_bytes=False):
        assert lean_cmd is not None, "lean_cmd must be provided"
        assert cwd is not None, "cwd must be provided"
        assert os.path.isdir(cwd), "cwd must be a valid directory"
        self.lean_cmd = lean_cmd
        self.seq_num = 0
        self.debug = debug
        self.debug_bytes = debug_bytes
        self.response_events = {}  # Key: seq_num, Value: threading.Event
        self.responses = {}  # Key: seq_num, Value: response
        self.is_fully_ready = Event()
        self.process = None
        self.messages : List[Message] = []
        self.cwd = cwd
        self.current_tasks : List[Task] = []
        self._exit_receiver = False
        self._lock = threading.Lock() # This is needed for safety of dictionary access   

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.kill()

    def start(self):
        self.process = Popen(
            self.lean_cmd, 
            shell = True, 
            stdin = PIPE, 
            stdout = PIPE, 
            stderr = STDOUT,
            cwd = self.cwd, 
            bufsize = 1, 
            universal_newlines = True)
        self.thread = threading.Thread(target=self.receiver)
        self.thread.start()

    def send(self, request: Request) -> Optional[CommandResponse]:
        self.seq_num += 1
        request.seq_num = self.seq_num

        if self.debug:
            print(f'Sending {request}')

        json_request = request.to_json() + '\n'
        if self.debug_bytes:
            print(f'Sending {json_request.encode()}')

        # First create the event
        response_event = Event()
        with self._lock:
            self.response_events[self.seq_num] = response_event

        # Then send the request
        self.process.stdin.write(json_request)
        self.process.stdin.flush()

        # Wait for the response
        response_event.wait()

        # Get the response
        with self._lock:
            self.response_events.pop(self.seq_num)
            response = self.responses.pop(self.seq_num)

        if isinstance(response, OkResponse):
            return response.to_command_response(request.command)
        elif isinstance(response, ErrorResponse):
            raise ChildProcessError(f'Lean server error while executing "{request.command}":\n{response}')

    def receiver(self):
        while not self._exit_receiver:
            line = self.process.stdout.readline().strip()
            if line:
                if self.debug_bytes:
                    print(f'Received {line}')
                response = CommandResponse.parse_response(line)
                if self.debug:
                    print(f'Received {response}')

                if isinstance(response, CurrentTasksResponse):
                    self.current_tasks = response.tasks
                    if not response.is_running:
                        self.is_fully_ready.set()
                elif isinstance(response, AllMessagesResponse):
                    self.messages = response.msgs  # Storing messages
                elif isinstance(response, OkResponse) or isinstance(response, ErrorResponse):
                    seq_num = response.seq_num
                    with self._lock:
                        # seq_num will be in the dictionary because we created the event before sending the request
                        self.responses[seq_num] = response
                        self.response_events[seq_num].set()
            else:
                # sleep for a bit to avoid busy waiting
                time.sleep(0.02)

    def full_sync(self, filename, content=None):
        request = SyncRequest(filename, content)
        response = self.send(request)

        if isinstance(response, SyncResponse) and response.message == "file invalidated":
            self.is_fully_ready.clear()
            self.is_fully_ready.wait()

    def state(self, filename, line, col) -> str:
        request = InfoRequest(filename, line, col)
        response = self.send(request)

        if isinstance(response, InfoResponse) and response.record:
            return response.record.state or ''
        else:
            return ''

    def kill(self):
        # Ensure that the process is killed
        self.process.kill()
        try:
            self.process.stdin.close()
        except:
            pass
        self._exit_receiver = True
        self.thread.join(1.0)

if __name__ == "__main__":
    from pathlib import Path
    os.chdir(root_dir)
    # cwd = 'data/benchmarks/miniF2F'
    cwd = 'data/test/lean_proj'
    # path = f'{cwd}/lean/src/test.lean'
    path = f'{cwd}/src/simple.lean'
    lines = Path(path).read_text().split('\n')
    with SyncLeanServer(cwd=cwd) as server:
        path = 'temp124321432.lean'
        server.full_sync(path, content='')

        theorem_started = False
        for i, line in enumerate(lines):
            content = '\n'.join(lines[:i+1])
            if 'theorem' in line:
                theorem_started = True
            if theorem_started and 'end' in line:
                theorem_started = False
            if theorem_started:
                content += '\nend'
            server.full_sync(path, content)
            if len(server.messages) > 0:
                for idx, message in enumerate(server.messages):
                    print(f"Message [{idx + 1}]: {message.text}")
            before = server.state(path, i+1, 0)
            after = server.state(path, i+1, len(line))
            print(f'Line {i+1}: {line}')
            if before or after:
                print(f'State before:\n{before}\n')
                print(f'State after:\n{after}\n')
