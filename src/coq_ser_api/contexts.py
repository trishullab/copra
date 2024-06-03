#!/usr/bin/env python3.7
##########################################################################
#
#    This file is part of Proverbot9001.
#
#    Proverbot9001 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Proverbot9001 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Proverbot9001.  If not, see <https://www.gnu.org/licenses/>.
#
#    Copyright 2019 Alex Sanchez-Stern and Yousef Alhessi
#
##########################################################################

import json
import hashlib
from typing import (List, TextIO, Optional, NamedTuple, Union, Dict,
                    Any, Type, TYPE_CHECKING, Sequence)

if TYPE_CHECKING:
    from sexpdata import Sexp

class SexpObligation(NamedTuple):
    hypotheses: List['Sexp']
    goal: 'Sexp'

class Obligation:
    hypotheses: Sequence[str]
    goal: str

    def __init__(self, hypotheses: Sequence[str], goal: str) -> None:
        self.hypotheses = tuple(hypotheses)
        self.goal = goal

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Obligation):
            return False
        if self.goal != other.goal:
            return False
        if self.hypotheses != other.hypotheses:
            return False
        return True

    def __hash__(self) -> int:
        return int.from_bytes(hashlib.md5(json.dumps(
          (self.hypotheses, self.goal),
           sort_keys=True).encode('utf-8')).digest(), byteorder='big')

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"hypotheses": list(self.hypotheses),
                "goal": self.goal}

    @classmethod
    def from_structeq(cls, obj: Any) -> 'Obligation':
        return Obligation(tuple(obj.hypotheses), obj.goal)
    
    def __str__(self) -> str:
        return f"Obligation(goal={self.goal}, hypotheses={self.hypotheses})"


class ProofContext(NamedTuple):
    fg_goals: List[Obligation]
    bg_goals: List[Obligation]
    shelved_goals: List[Obligation]
    given_up_goals: List[Obligation]

    @classmethod
    def empty(cls: Type['ProofContext']):
        return ProofContext([], [], [], [])

    @classmethod
    def from_dict(cls, data):
        fg_goals = list(map(Obligation.from_dict, data["fg_goals"]))
        bg_goals = list(map(Obligation.from_dict, data["bg_goals"]))
        shelved_goals = list(map(Obligation.from_dict, data["shelved_goals"]))
        given_up_goals = list(map(Obligation.from_dict,
                                  data["given_up_goals"]))
        return cls(fg_goals, bg_goals, shelved_goals, given_up_goals)

    def to_dict(self) -> Dict[str, Any]:
        return {"fg_goals": list(map(Obligation.to_dict, self.fg_goals)),
                "bg_goals": list(map(Obligation.to_dict, self.bg_goals)),
                "shelved_goals": list(map(Obligation.to_dict,
                                          self.shelved_goals)),
                "given_up_goals": list(map(Obligation.to_dict,
                                           self.given_up_goals))}

    @property
    def all_goals(self) -> List[Obligation]:
        return self.fg_goals + self.bg_goals + \
            self.shelved_goals + self.given_up_goals

    @property
    def focused_goal(self) -> str:
        if self.fg_goals:
            return self.fg_goals[0].goal
        else:
            return ""

    @property
    def focused_hyps(self) -> List[str]:
        if self.fg_goals:
            return list(self.fg_goals[0].hypotheses)
        else:
            return []

    @classmethod
    def from_structeq(cls, obj: Any) -> 'ProofContext':
        return ProofContext([Obligation.from_structeq(fg) for fg in obj.fg_goals],
                            [Obligation.from_structeq(bg) for bg in obj.bg_goals],
                            [Obligation.from_structeq(sg) for sg in obj.shelved_goals],
                            [Obligation.from_structeq(gg) for gg in obj.given_up_goals])
    
    def __str__(self) -> str:
        return f"ProofContext(fg_goals={self.fg_goals}, bg_goals={self.bg_goals}, "\
            f"shelved_goals={self.shelved_goals}, given_up_goals={self.given_up_goals})"


def assert_proof_context_matches(context1: ProofContext, context2: ProofContext) -> None:
    def assert_obligation_matches(label: str, obl1: Obligation, obl2: Obligation) -> None:
        assert obl1.goal == obl2.goal, f"{label}: Goals {obl1.goal} and {obl2.goal} don't match"
        for idx, (hyp1, hyp2) in enumerate(zip(obl1.hypotheses, obl2.hypotheses)):
            assert hyp1 == hyp2, f"{label}: Hypotheses at index {idx} don't match! "\
                f"{hyp1} vs {hyp2}"
    assert len(context1.fg_goals) == len(context2.fg_goals), \
        "Number of foreground goals doesn't match! "\
        f"First context has {len(context1.fg_goals)} goals, "\
        f"but second context has {len(context2.fg_goals)} goals."
    for idx, (fg_goal1, fg_goal2) in enumerate(zip(context1.fg_goals,
                                                   context2.fg_goals)):
        assert_obligation_matches(f"Item {idx} of foreground goals", fg_goal1, fg_goal2)
    assert len(context1.bg_goals) == len(context2.bg_goals), \
        "Number of background goals doesn't match! "\
        f"First context has {len(context1.fg_goals)} goals, "\
        f"but second context has {len(context2.fg_goals)} goals."
    for idx, (bg_goal1, bg_goal2) in enumerate(zip(context1.bg_goals,
                                                   context2.bg_goals)):
        assert_obligation_matches(f"Item {idx} of background goals", bg_goal1, bg_goal2)
    assert len(context1.shelved_goals) == len(context2.shelved_goals), \
        "Number of shelved goals doesn't match! "\
        f"First context has {len(context1.fg_goals)} goals, "\
        f"but second context has {len(context2.fg_goals)} goals."
    for idx, (shelved_goal1, shelved_goal2) in enumerate(zip(context1.shelved_goals,
                                                             context2.shelved_goals)):
        assert_obligation_matches(f"Item {idx} of shelved goals",
                                  shelved_goal1, shelved_goal2)
    assert len(context1.given_up_goals) == len(context2.given_up_goals), \
        "Number of background goals doesn't match! "\
        f"First context has {len(context1.fg_goals)} goals, "\
        f"but second context has {len(context2.fg_goals)} goals."
    for idx, (given_up_goal1, given_up_goal2) in enumerate(zip(context1.given_up_goals,
                                                               context2.given_up_goals)):
        assert_obligation_matches(f"Item {idx} of given up goals",
                                  shelved_goal1, shelved_goal2)

def ident_in_context(ident: str, context: ProofContext) -> bool:
    def ident_in_obl(obligation: Obligation) -> bool:
        if ident in obligation.goal:
            return True
        return any(ident in hyp for hyp in obligation.hypotheses)
    return any(ident_in_obl(obl) for obl in
               context.all_goals)


class ScrapedTactic(NamedTuple):
    relevant_lemmas: List[str]
    prev_tactics: List[str]
    context: ProofContext
    tactic: str

    def to_dict(self) -> Dict[str, Any]:
        return {"relevant_lemmas": self.relevant_lemmas,
                "prev_tactics": self.prev_tactics,
                "context": self.context.to_dict(),
                "tactic": self.tactic}

    @classmethod
    def from_structeq(cls, obj: Any) -> 'ScrapedTactic':
        return ScrapedTactic(obj.relevant_lemmas, obj.prev_tactics,
                             ProofContext.from_structeq(obj.context),
                             obj.tactic)


class TacticContext:
    relevant_lemmas: Sequence[str]
    prev_tactics: Sequence[str]
    hypotheses: Sequence[str]
    goal: str

    def __init__(self, relevant_lemmas: Sequence[str], prev_tactics: Sequence[str],
                 hypotheses: Sequence[str], goal: str) -> None:
        self.relevant_lemmas = tuple(relevant_lemmas)
        self.prev_tactics = tuple(prev_tactics)
        self.hypotheses = tuple(hypotheses)
        self.goal = goal
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TacticContext):
            return False
        if self.goal != other.goal:
            return False
        if self.hypotheses != other.hypotheses:
            return False
        if self.relevant_lemmas != other.relevant_lemmas:
            return False
        if self.prev_tactics != other.prev_tactics:
            return False
        return True
    def __hash__(self) -> int:
        return hash((self.relevant_lemmas, self.prev_tactics, self.hypotheses, self.goal))


class FullContext(NamedTuple):
    relevant_lemmas: List[str]
    prev_tactics: List[str]
    obligations: ProofContext

    def as_tcontext(self) -> TacticContext:
        return TacticContext(self.relevant_lemmas,
                             self.prev_tactics,
                             self.obligations.focused_hyps,
                             self.obligations.focused_goal)


def truncate_tactic_context(context: TacticContext,
                            max_term_length: int):
    def truncate_hyp(hyp: str) -> str:
        var_term = hyp.split(":")[0].strip()
        hyp_type = hyp.split(":", 1)[1].strip()
        return f"{var_term} : {hyp_type}"
    return TacticContext(
        [truncate_hyp(lemma) for lemma
         in context.relevant_lemmas],
        context.prev_tactics,
        [truncate_hyp(hyp) for hyp
         in context.hypotheses],
        context.goal[:max_term_length])


ScrapedCommand = Union[ScrapedTactic, str]


def strip_scraped_output(scraped: ScrapedTactic) -> TacticContext:
    relevant_lemmas, prev_tactics, context, tactic = scraped
    if context and context.fg_goals:
        return TacticContext(relevant_lemmas, prev_tactics,
                             context.fg_goals[0].hypotheses,
                             context.fg_goals[0].goal)
    else:
        return TacticContext(relevant_lemmas, prev_tactics,
                             [], "")


def read_tuple(f_handle: TextIO) -> Optional[ScrapedCommand]:
    line = f_handle.readline()
    if line.strip() == "":
        return None
    obj = json.loads(line)
    if isinstance(obj, str):
        return obj
    else:
        return ScrapedTactic(obj["relevant_lemmas"],
                             obj["prev_tactics"],
                             ProofContext.from_dict(obj["context"]),
                             obj["tactic"])


def read_tactic_tuple(f_handle: TextIO) -> Optional[ScrapedTactic]:
    next_tuple = read_tuple(f_handle)
    while(isinstance(next_tuple, str)):
        next_tuple = read_tuple(f_handle)
    return next_tuple
