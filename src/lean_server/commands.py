"""
Basic tools to convert back and forth between json expected or sent by the Lean
server and python objets. Mostly based on the TypeScript version.

Anything whose name contains Request is meant to be sent to the Lean server after
conversion by its to_json method.

Anything whose name contains Response is meant to be built from some json sent
by the Lean server by the parse_response function at the bottom of this file.

Everything else in this file are intermediate objects that will be contained in
response objects.
"""
#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from dataclasses import dataclass
from typing import Optional, List, NewType, ClassVar, Union, Type
from enum import Enum
import json


def dict_to_dataclass(cls, dic: dict):
    dic = {k: dic[k] for k in cls.__dataclass_fields__ if k in dic}
    return cls(**dic)


class Request:
    command: ClassVar[str]
    expect_response: ClassVar[bool]

    def __post_init__(self):
        self.seq_num = 0

    def to_json(self) -> str:
        dic = self.__dict__.copy()
        dic['command'] = self.command
        return json.dumps(dic)


class Response:
    response: ClassVar[str]

    @staticmethod
    def parse_response(data: str) -> Union['AllMessagesResponse', 'CurrentTasksResponse', 'OkResponse', 'ErrorResponse']:
        dic = json.loads(data)
        response = dic.pop('response')

        for cls in [AllMessagesResponse, CurrentTasksResponse, OkResponse, ErrorResponse]:
            if response == cls.response:  # type: ignore
                return cls.from_dict(dic)  # type: ignore
        raise ValueError("Couldn't parse response string.")



Severity = Enum('Severity', 'information warning error')


@dataclass
class Message:
    file_name: str
    severity: Severity
    caption: str
    text: str
    pos_line: int
    pos_col: int
    end_pos_line: Optional[int] = None
    end_pos_col: Optional[int] = None

    @classmethod
    def from_dict(cls, dic):
        dic['severity'] = getattr(Severity, dic['severity'])
        return dict_to_dataclass(cls, dic)


@dataclass
class AllMessagesResponse(Response):
    response = 'all_messages'
    msgs: List[Message]

    @classmethod
    def from_dict(cls, dic):
        return cls([Message.from_dict(msg) for msg in dic['msgs']])


@dataclass
class Task:
    file_name: str
    pos_line: int
    pos_col: int
    end_pos_line: int
    end_pos_col: int
    desc: str


@dataclass
class CurrentTasksResponse(Response):
    response = 'current_tasks'
    is_running: bool
    tasks: List[Task]
    cur_task: Optional[Task] = None

    @classmethod
    def from_dict(cls, dic):
        dic['tasks'] = [dict_to_dataclass(Task, task) for task in dic.pop('tasks')]
        return dict_to_dataclass(cls, dic)


@dataclass
class ErrorResponse(Response):
    response = 'error'
    message: str
    seq_num: Optional[int] = None

    @classmethod
    def from_dict(cls, dic):
        return dict_to_dataclass(cls, dic)

@dataclass
class CommandResponse(Response):
    """
    Parent class for all 'ok' responses directly tied to a specific request.
    """
    command: ClassVar[str]
    response = 'ok'
    seq_num: int

    @classmethod
    def from_dict(cls, dic):
        return dict_to_dataclass(cls, dic)


@dataclass
class OkResponse(Response):
    """
    Intermediate representation of a CommandResponse which can be constructed from the
    JSON alone.  It can later be converted to a CommandResponse.
    """
    response = 'ok'
    seq_num: int
    data: dict

    @classmethod
    def from_dict(cls, dic) -> 'OkResponse':
        return OkResponse(seq_num=dic['seq_num'], data=dic)

    def to_command_response(self, command: str) -> CommandResponse:
        response_types: List[Type[CommandResponse]] = [
            CompleteResponse, InfoResponse, HoleCommandsResponse, SyncResponse,
            SearchResponse, AllHoleCommandsResponse, HoleResponse, RoiResponse
        ]
        for cls in response_types:
            if cls.command == command:
                self.data['seq_num'] = self.seq_num
                return cls.from_dict(self.data)
        raise ValueError("Couldn't parse response string.")


@dataclass
class SyncRequest(Request):
    command = 'sync'
    expect_response = True
    file_name: str
    content: Optional[str] = None

    def to_json(self):
        dic = self.__dict__.copy()
        dic['command'] = 'sync'
        if dic['content'] is None:
            dic.pop('content')
        return json.dumps(dic)


@dataclass
class SyncResponse(CommandResponse):
    command = 'sync'
    message: Optional[str] = None


@dataclass
class CompleteRequest(Request):
    command = 'complete'
    expect_response = True
    file_name: str
    line: int
    column: int
    skip_completions: bool = False


@dataclass
class Source:
    line: Optional[int] = None
    column: Optional[int] = None
    file: Optional[str] = None


@dataclass
class CompletionCandidate:
    text: str
    type_: Optional[str] = None
    tactic_params: Optional[str] = None
    doc: Optional[str] = None
    source: Optional[Source] = None

    @classmethod
    def from_dict(cls, dic):
        dic['type_'] = dic.pop('type')
        if 'source' in dic:
            dic['source'] = dict_to_dataclass(Source, dic.pop('source'))
        return dict_to_dataclass(cls, dic)


@dataclass
class CompleteResponse(CommandResponse):
    command = 'complete'
    prefix: Optional[str] = None
    completions: Optional[List[CompletionCandidate]] = None

    @classmethod
    def from_dict(cls, dic):
        if 'completions' in dic:
            dic['completions'] = [CompletionCandidate.from_dict(cdt)
                                  for cdt in dic.pop('completions')]
        return dict_to_dataclass(cls, dic)


@dataclass
class InfoRequest(Request):
    command = 'info'
    expect_response = True
    file_name: str
    line: int
    column: int


GoalState = NewType('GoalState', str)


@dataclass
class InfoRecord:
    full_id: Optional[str] = None
    text: Optional[str] = None
    type_: Optional[str] = None
    doc: Optional[str] = None
    source: Optional[Source] = None
    state: Optional[GoalState] = None
    tactic_param_idx: Optional[int] = None
    tactic_params: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, dic):
        if 'full-id' in dic:
            dic['full_id'] = dic.pop('full-id')
        if 'type' in dic:
            dic['type_'] = dic.pop('type')
        if 'source' in dic:
            dic['source'] = dict_to_dataclass(Source, dic.pop('source'))
        return dict_to_dataclass(cls, dic)


@dataclass
class InfoResponse(CommandResponse):
    command = 'info'
    record: Optional[InfoRecord] = None

    @classmethod
    def from_dict(cls, dic):
        if 'record' in dic:
            dic['record'] = InfoRecord.from_dict(dic.pop('record'))
        return dict_to_dataclass(cls, dic)


@dataclass
class SearchRequest(Request):
    command = 'search'
    expect_response = True
    query: str


@dataclass
class SearchItem:
    text: str
    type_: str
    source: Optional[Source] = None
    doc: Optional[str] = None

    @classmethod
    def from_dict(cls, dic):
        dic['type_'] = dic.pop('type')
        if 'source' in dic:
            dic['source'] = dict_to_dataclass(Source, dic.pop('source'))
        return dict_to_dataclass(cls, dic)


@dataclass
class SearchResponse(CommandResponse):
    command = 'search'
    results: List[SearchItem]

    @classmethod
    def from_dict(cls, dic):
        dic['results'] = [SearchItem.from_dict(si)
                          for si in dic.pop('results')]
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleCommandsRequest(Request):
    command = 'hole_commands'
    expect_response = True
    file_name: str
    line: int
    column: int


@dataclass
class HoleCommandAction:
    name: str
    description: str


@dataclass
class Position:
    line: int
    column: int


@dataclass
class HoleCommands:
    file: str
    start: Position
    end: Position
    results: List[HoleCommandAction]

    @classmethod
    def from_dict(cls, dic):
        dic['results'] = [dict_to_dataclass(HoleCommandAction, hc)
                          for hc in dic.pop('results')]
        dic['start'] = dict_to_dataclass(Position, dic.pop('start'))
        dic['end'] = dict_to_dataclass(Position, dic.pop('end'))
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleCommandsResponse(CommandResponse):
    command = 'hole_commands'
    message: Optional[str] = None
    file: Optional[str] = None
    start: Optional[Position] = None
    end: Optional[Position] = None
    results: Optional[List[HoleCommandAction]] = None

    @classmethod
    def from_dict(cls, dic):
        if 'results' in dic:
            dic['results'] = [dict_to_dataclass(HoleCommandAction, hc)
                              for hc in dic.pop('results')]
            dic['start'] = dict_to_dataclass(Position, dic.pop('start'))
            dic['end'] = dict_to_dataclass(Position, dic.pop('end'))

        return dict_to_dataclass(cls, dic)


@dataclass
class AllHoleCommandsRequest(Request):
    command = 'all_hole_commands'
    expect_response = True
    file_name: str


@dataclass
class AllHoleCommandsResponse(CommandResponse):
    command = 'all_hole_commands'
    holes: List[HoleCommands]

    @classmethod
    def from_dict(cls, dic):
        dic['holes'] = [HoleCommands.from_dict(hole)
                          for hole in dic.pop('holes')]
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleRequest(Request):
    command = 'hole'
    expect_response = True
    file_name: str
    line: int
    column: int
    action: str


@dataclass
class HoleReplacementAlternative:
    code: str
    description: str


@dataclass
class HoleReplacements:
    file: str
    start: Position
    end: Position
    alternatives: List[HoleReplacementAlternative]

    @classmethod
    def from_dict(cls, dic):
        dic['alternatives'] = [dict_to_dataclass(HoleReplacementAlternative, alt)
                               for alt in dic.pop('alternatives')]
        dic['start'] = dict_to_dataclass(Position, dic.pop('start'))
        dic['end'] = dict_to_dataclass(Position, dic.pop('end'))
        return dict_to_dataclass(cls, dic)


@dataclass
class HoleResponse(CommandResponse):
    command = 'hole'
    replacements: Optional[HoleReplacements] = None
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, dic):
        if 'replacements' in dic:
            dic['replacements'] = HoleReplacements.from_dict(
                    dic.pop('replacements'))
        return dict_to_dataclass(cls, dic)


CheckingMode = Enum('CheckingMode',
    'nothing visible-lines visible-lines-and-above visible-files open-files')


@dataclass
class RoiRange:
    begin_line: int
    end_line: int


@dataclass
class FileRoi:
    file_name: str
    ranges: List[RoiRange]

    def to_dict(self):
        return {'file_name': self.file_name,
                'ranges': [rr.__dict__ for rr in self.ranges] }


@dataclass
class RoiRequest(Request):
    command = 'roi'
    expect_response = True
    mode: CheckingMode
    files: List[FileRoi]

    def to_json(self) -> str:
        dic = self.__dict__.copy()
        dic['command'] = 'roi'
        dic['mode'] = dic['mode'].name
        dic['files'] = [fileroi.to_dict() for fileroi in dic['files']]

        return json.dumps(dic)


@dataclass
class RoiResponse(CommandResponse):
    command = 'roi'


@dataclass
class SleepRequest(Request):
    command = 'sleep'
    expect_response = False


@dataclass
class LongSleepRequest(Request):
    command = 'long_sleep'
    expect_response = False


