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
from typing import List, TextIO, Optional, NamedTuple, Union, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from sexpdata import Sexp

class SexpObligation(NamedTuple):
    hypotheses: List['Sexp']
    goal: 'Sexp'

class Obligation(NamedTuple):
    hypotheses: List[str]
    goal: str

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"hypotheses": self.hypotheses,
                "goal": self.goal}


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
            return self.fg_goals[0].hypotheses
        else:
            return []


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


class TacticContext(NamedTuple):
    relevant_lemmas: List[str]
    prev_tactics: List[str]
    hypotheses: List[str]
    goal: str


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
