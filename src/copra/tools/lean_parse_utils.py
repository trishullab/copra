#!/usr/bin/env python3

import typing
from copra.lean_server.lean_utils import Lean3Utils

class LeanLineByLineReader(object):
    def __init__(self, file_name: str = None, file_content: str = None, remove_comments: bool = False, no_strip: bool = False):
        assert file_name is not None or file_content is not None, "Either file_name or file_content must be provided"
        assert file_name is None or file_content is None, "Only one of file_name or file_content must be provided"
        self.file_name : str = file_name
        self.file_content : str = file_content
        self.no_strip = no_strip
        if self.file_name is not None:
            with open(file_name, 'r') as fd:
                self.file_content : str = fd.read()
        if remove_comments:
            self.file_content = Lean3Utils.remove_comments(self.file_content)
    
    def instruction_step_generator(self) -> typing.Iterator[str]:
        lines = self.file_content.split('\n')
        for line in lines:
            if not self.no_strip:
                line = line.strip()
            else:
                line = line
            yield line