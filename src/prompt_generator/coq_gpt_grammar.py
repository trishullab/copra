#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.prompt_generator.interpreter import Grammar

class CoqGPTRequestGrammar(Grammar):
    grammar = """
Prog: 
  RunTacticRequest
| GetThmsRequest
| GetDfnsRequest
| String Prog;
GetThmsRequest:
    GetThms End;
GetDfnsRequest:
    GetDfns End;
RunTacticRequest:
    RunTactic StpRequests End;
StpRequests:
  Stp String 
| Stp String StpRequests;

terminals
Stp: "[STP]";
End: "[END]";
RunTactic: "[RUN TACTIC]";
GetThms: "[GET THMS]";
GetDfns: "[GET DFNS]";
String:;
"""
    keywords = ["[STP]", "[END]", "[RUN TACTIC]", "[GET THMS]", "[GET DFNS]"]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in CoqGPTRequestGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1

    def __init__(self):
        recognizers = {
            'String': CoqGPTRequestGrammar.before_keyword
        }
        super(CoqGPTRequestGrammar, self).__init__(CoqGPTRequestGrammar.grammar, CoqGPTRequestGrammar.keywords, recognizers=recognizers)
    pass

#   RunTacticResponse
# | GetThmsResponse
# | GetDfnsResponse
class CoqGPTResponseGrammar(Grammar):
    grammar = """
Prog:
  GoalsResponse
| RunTacticResponse
| GetThmsResponses
| GetDfnsResponses
| String Prog;
RunTacticResponse:
  RunTacticResult Success End
| RunTacticResult Error String End;
GetThmsResponses:
  GetThmsResult ThmsResponses End;
ThmsResponses:
  Thms int ThmList
| Thms int ThmList ThmsResponses;
ThmList:
  Thm String
| Thm String ThmList;
GetDfnsResponses:
  GetDfnsResult DfnsResponses End;
DfnsResponses:
  Dfns int DfnList
| Dfns int DfnList DfnsResponses;
DfnList:
  Dfn String
| Dfn String DfnList;
GoalsResponse:
  Goals GoalResponses Stps;
GoalResponses:
  GoalResponse
| GoalResponse GoalResponses;
GoalResponse:
  Goal int String Hyps int
|  Goal int String Hyps int HypResponses;
HypResponses:
  HypResponse
| HypResponse HypResponses;
HypResponse:
  Hyp String;


terminals
int: /\d+/;
Goals: "[GLS]";
Goal: "[GL]";
Hyps: "[HYPS]";
Hyp: "[HYP]";
Stps: "[STPS]";
Stp: "[STP]";
Dfns: "[DFNS]";
Dfn: "[DFN]";
Thms: "[THMS]";
Thm: "[THM]";
Error: "[ERROR]";
Success: "[SUCCESS]";
End: "[END]";
RunTacticResult: "[RUN TACTIC RESULT]";
GetThmsResult: "[GET THMS RESULT]";
GetDfnsResult: "[GET DFNS RESULT]";
String:;
"""
    keywords = ["[GLS]", "[GL]", "[HYPS]", "[HYP]", "[STPS]", "[STP]", "[DFNS]", "[DFN]", "[THMS]", "[THM]", "[ERROR]", "[SUCCESS]", "[END]", "[RUN TACTIC RESULT]", "[GET THMS RESULT]", "[GET DFNS RESULT]"]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in CoqGPTResponseGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1

    def __init__(self):
        recognizers = {
            'String': CoqGPTResponseGrammar.before_keyword
        }
        super(CoqGPTResponseGrammar, self).__init__(CoqGPTResponseGrammar.grammar, CoqGPTResponseGrammar.keywords, recognizers=recognizers)

if __name__ == "__main__":
    grammar = CoqGPTRequestGrammar()
    result = grammar.compile(
"""
Please run the tactics below.
[RUN TACTIC] [STP] reflexivity.
[STP]reflexivity.
rewrite <- plus_n_O.[END]"""
    )
    print(result)
    result = grammar.compile(
"""
[GET THMS]
[END]"""
    )
    print(result)
    result = grammar.compile(
"""
[GET DFNS]
[END]"""
    )
    print(result)
    grammar = CoqGPTResponseGrammar()
    result = grammar.compile("""
New goals to prove:
[GLS]
[GL] 1
n + 0 = n
[HYPS] 1
[HYP] n: nat
[GL] 2
n + 1 = n + 1
[HYPS] 1
[HYP] n: nat
[HYP] nat: Set

[STPS]
""")
    print(result)
    result = grammar.compile("""
[RUN TACTIC RESULT][SUCCESS]
[END]
""")
    print(result)
    result = grammar.compile("""
[RUN TACTIC RESULT][ERROR]
In environment
n : nat
Unable to unify "n" with "n + 0".
[END]
""")
    print(result)

    result = grammar.compile("""
[GET THMS RESULT]
[THMS] 1
[THM]plus_n_O : forall n  nat, n = n + 0
[THM]plus_O_n : forall n  nat, 0 + n = n
[THM]mult_n_O : forall n  nat, 0 = n * 0
[THM]plus_n_Sm : forall n m  nat, S (n
[THMS] 2
[THM]plus_n_O : forall n  nat, n = n + 0
[THM]plus_O_n : forall n  nat, 0 + n = n
[THM]mult_n_O : forall n  nat, 0 = n * 0
[THM]plus_n_Sm : forall n m  nat, S (n
[END]
""")
    print(result)

    result = grammar.compile("""
[GET DFNS RESULT]
[DFNS] 1
[DFN]nat: Set
[DFNS] 2
[DFN]nat: Set
[DFN]nat: Set
[END]
""")
    print(result)