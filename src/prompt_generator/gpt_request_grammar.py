#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.prompt_generator.interpreter import Grammar
from src.tools.training_data_format import TrainingDataFormat
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class CoqGptRequestActions(object):
    RUN_TACTIC = "[RUN TACTIC]"
    GET_DFNS_THMS = "[GET DEFINITIONS AND THEOREMS]"

@dataclass_json
@dataclass
class CoqGptRequest(object):
    action : str = CoqGptRequestActions.RUN_TACTIC
    args: typing.List[str] = field(default_factory=list)

class CoqGPTRequestGrammar(Grammar):
    grammar = """
Prog: 
  RunTacticRequest
| GetDfnsThmsRequest
| String Prog;
GetDfnsThmsRequest:
    GetDfnsThms End;
RunTacticRequest:
    RunTactic StpRequests End;
StpRequests:
  String;

terminals
Stp: "[STEP]";
End: "[END]";
RunTactic: "[RUN TACTIC]";
GetDfnsThms: "[GET DEFINITIONS AND THEOREMS]";
String:;
"""
    keywords = ["[STEP]", "[END]", "[RUN TACTIC]", "[GET DEFINITIONS]"]

    end = "[END]"

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

    def _parse_expr(self, nonTerminal, nodes, context):
        if nonTerminal == "GetDfnsThmsRequest":
            context.action = CoqGptRequestActions.GET_DFNS_THMS
            context.args = [CoqGptRequestActions.GET_DFNS_THMS[1:-1]]
        elif nonTerminal == "RunTacticRequest":
            assert len(nodes) >= 2
            context.action = CoqGptRequestActions.RUN_TACTIC
            context.args.reverse()
        elif nonTerminal == "StpRequests":
            assert len(nodes) >= 1
            str_node = str(nodes[0]).strip()
            if len(str_node) > 0:
                context.args.append(str_node)
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
        return context
        
    def get_action(self, inp=None):
        context = CoqGptRequest()
        actions = {
            "Prog": lambda _, nodes: context,
            "GetDfnsThmsRequest": lambda _, nodes: self._parse_expr('GetDfnsThmsRequest', nodes, context),
            "RunTacticRequest": lambda _, nodes: self._parse_expr('RunTacticRequest', nodes, context),
            "StpRequests": lambda _, nodes: self._parse_expr('StpRequests', nodes, context),
            "String": lambda _, nodes: str(nodes) # Since this is always a string
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, CoqGptRequest), f"Result must be a CoqGptRequest. Got {type(result)}"
        return result
    
    def generate_message_from_gpt_request(self, coq_gpt_request: CoqGptRequest) -> str:
        if coq_gpt_request.action == CoqGptRequestActions.RUN_TACTIC:
            args = '\n'.join(coq_gpt_request.args)
            return f"{CoqGptRequestActions.RUN_TACTIC}\n{args}\n{CoqGPTRequestGrammar.end}"
        elif coq_gpt_request.action == CoqGptRequestActions.GET_DFNS_THMS:
            return f"{CoqGptRequestActions.GET_DFNS_THMS}\n{CoqGPTRequestGrammar.end}"
        else:
            raise Exception(f"Invalid action {coq_gpt_request.action}")

    def get_openai_request(self, message_response: str) -> typing.Tuple[CoqGptRequest, str]:
        message, finish_reason = message_response
        if finish_reason != "stop":            
            # do a greedy correction to ensure that the message is parsable
            idx = len(message)
            exceptions = []
            message_seems_fixable = True
            try:
                # trim any unwanted keywords at the end
                idx = message.rfind('[')
                if idx < 0:
                    raise Exception("No opening bracket found, message is not parsable")
                close_idx = message.rfind(']', idx, len(message))
                if close_idx < 0:
                    message = message[:idx]
                else:
                    idx = len(message)
            except Exception:
                message_seems_fixable = False
                pass
            if message_seems_fixable:    
                attempt = 0
                while idx >= 0:
                    try:
                        parsable_message = message[:idx] + f"\n{CoqGPTRequestGrammar.end}"
                        self.compile(parsable_message)
                        break
                    except Exception as e:
                        exceptions.append(e)
                        idx = message.rfind('[', 0, idx)
                    attempt += 1
                if idx >= 0:
                    message = parsable_message
                else:
                    raise exceptions[0]
                result : CoqGptRequest = self.run(message, None)
                if result.action == CoqGptRequestActions.RUN_TACTIC and len(result.args) > 1:
                    result.args = result.args[:-1] # remove the last tactic as it can be incomplete
            else:
                message += CoqGPTRequestGrammar.end
                result : CoqGptRequest = self.run(message, None)
        else:
            message += CoqGPTRequestGrammar.end
            result : CoqGptRequest = self.run(message, None)            
        message = self.generate_message_from_gpt_request(result)
        return (result, message)
    
    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(CoqGPTRequestGrammar.end), "Message must end with end token"
            result : CoqGptRequest = self.run(message, None)
            results.extend(result.args)
        return results 

if __name__ == "__main__":
    code = """
Please run the tactics below.
[RUN TACTIC] reflexivity.
reflexivity.
rewrite <- plus_n_O.[END]"""
    grammar = CoqGPTRequestGrammar()
    result = grammar.compile(code)
    print(result)
    run_result = grammar.run(code, None)
    print(run_result)
    result = grammar.compile(
"""
[GET DEFINITIONS AND THEOREMS]
[END]"""
    )
