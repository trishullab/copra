#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing

class LeanLineByLineReader(object):
    def __init__(self, file_name: str = None, file_content: str = None):
        assert file_name is not None or file_content is not None, "Either file_name or file_content must be provided"
        assert file_name is None or file_content is None, "Only one of file_name or file_content must be provided"
        self.file_name : str = file_name
        self.file_content : str = file_content
        if self.file_name is not None:
            with open(file_name, 'r') as fd:
                self.file_content : str = fd.read()
    
    def instruction_step_generator(self) -> typing.Iterator[str]:
        lines = self.file_content.split('\n')
        for line in lines:
            yield line.strip()