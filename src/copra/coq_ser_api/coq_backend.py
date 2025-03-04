#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Optional, List

from .contexts import ProofContext, SexpObligation

class CoqBackend(ABC):
    verbosity: int

    @abstractmethod
    def addStmt(self, stmt: str, timeout:Optional[int] = None,
                force_update_nonfg_goals: bool = False) -> None:
        pass
    @abstractmethod
    def addStmt_noupdate(self, stmt: str, timeout:Optional[int] = None) -> None:
        pass
    @abstractmethod
    def updateState(self) -> None:
        pass
    @abstractmethod
    def cancelLastStmt(self, cancelled: str, force_update_nonfg_goals: bool = False) -> None:
        pass
    @abstractmethod
    def cancelLastStmt_noupdate(self, cancelled: str) -> None:
        pass

    @abstractmethod
    def getProofContext(self) -> Optional[ProofContext]:
        pass
    @abstractmethod
    def getSexpProofContext(self) -> List[SexpObligation]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def isInProof(self) -> bool:
        pass

    @abstractmethod
    def queryVernac(self, vernac: str) -> List[str]:
        pass
    @abstractmethod
    def interrupt(self) -> None:
        pass

    def enterDirectory(self, root_dir: str) -> None:
        pass

    @abstractmethod
    def setFilename(self, filename: str) -> None:
        pass

    @abstractmethod
    def resetCommandState(self) -> None:
        pass

    @abstractmethod
    def backToState(self, state_num: int) -> None:
        pass

    @abstractmethod
    def backToState_noupdate(self, state_num: int) -> None:
        pass


# Some Exceptions to throw when various responses come back from coq
@dataclass
class CoqException(Exception):
    msg: str


@dataclass
class AckError(CoqException):
    pass


@dataclass
class CompletedError(CoqException):
    pass


@dataclass
class CoqExn(CoqException):
    pass


@dataclass
class BadResponse(CoqException):
    pass


@dataclass
class NotInProof(CoqException):
    pass


@dataclass
class ParseError(CoqException):
    pass


@dataclass
class LexError(CoqException):
    pass


@dataclass
class CoqTimeoutError(CoqException):
    pass


@dataclass
class CoqOverflowError(CoqException):
    pass


@dataclass
class UnrecognizedError(CoqException):
    pass


@dataclass
class NoSuchGoalError(CoqException):
    pass


@dataclass
class CoqAnomaly(CoqException):
    pass
