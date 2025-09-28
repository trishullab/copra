#!/usr/bin/env python3

import typing
import re
from copra.prompt_generator.interpreter import Grammar
from itp_interface.tools.training_data_format import TrainingDataFormat
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

    run_tactic_regex = r"\[RUN TACTIC\]([\s|\S]*?)\[END\]"
    run_tactic_match = re.compile(run_tactic_regex)

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

    def __init__(self, enable_defensive_parsing: bool = False):
        recognizers = {
            'String': CoqGPTRequestGrammar.before_keyword
        }
        self.enable_defensive_parsing = enable_defensive_parsing
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
            str_node = str(nodes[0])
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
        defensive_parsing = finish_reason != "stop" or self.enable_defensive_parsing
        if defensive_parsing:     
            return self.defensive_parsing(message)
        else:
            return self.normal_parsing(message)
    
    def normal_parsing(self, message: str):
        # Check if the message has the regex for run tactic
        if not message.endswith(CoqGPTRequestGrammar.end):
            message += CoqGPTRequestGrammar.end
        match = CoqGPTRequestGrammar.run_tactic_match.search(message)
        if match:
            message = match.group(0)
        else:
            raise Exception(f"Invalid message {message}, cannot find run tactic")
        message = message.strip()
        message = message.rstrip(CoqGPTRequestGrammar.end)
        action = None
        if message.startswith(CoqGptRequestActions.RUN_TACTIC):
            message = message[len(CoqGptRequestActions.RUN_TACTIC):]
            action = CoqGptRequestActions.RUN_TACTIC
        elif message.startswith(CoqGptRequestActions.GET_DFNS_THMS):
            raise Exception("GET_DFNS_THMS is not supported in normal parsing")
        else:
            raise Exception(f"Invalid message {message}")
        # Remove any newlines at the beginning or end
        message = message.strip('\n')

        # VALIDATION: Reject tactics containing ✝ symbol
        if '✝' in message:
            raise Exception(f"Invalid tactic: tactics cannot contain the ✝ symbol. Found in: {message}")

        result = CoqGptRequest(action=action, args=[message])
        # result : CoqGptRequest = self.run(message, None)
        message = self.generate_message_from_gpt_request(result)
        return (result, message)

    def defensive_parsing(self, message):
        start_idx = 0
        end_idx = len(message)
        # Generate all possible sub-strings such that the start_idx is less than end_idx
        idxs = [(s_idx, e_idx) for s_idx in range(start_idx, end_idx) for e_idx in range(end_idx, s_idx, -1)]
        message_temp = message
        message_parsed = False
        for s_idx, e_idx in idxs:
            # This type of robust parsing can be needed in case of some LLMs which
            # don't follow the specified format
            try:
                message_temp = message[s_idx:e_idx]
                if message_temp.endswith(CoqGPTRequestGrammar.end):
                    # Just in case the LLM doesn't remove the stop token
                    message_temp = message_temp.strip(CoqGPTRequestGrammar.end)
                message_temp += f"\n{CoqGPTRequestGrammar.end}"
                result : CoqGptRequest = self.run(message_temp, None)

                # VALIDATION: Reject tactics containing ✝ symbol
                if result.action == CoqGptRequestActions.RUN_TACTIC:
                    for arg in result.args:
                        if '✝' in arg:
                            raise Exception(f"Invalid tactic: tactics cannot contain the ✝ symbol. Found in: {arg}")

                message_temp = self.generate_message_from_gpt_request(result)
                message_parsed = True
            except:
                message_parsed = False
            if message_parsed:
                break
        if not message_parsed:
            message_temp = message[start_idx:end_idx]
            message_temp += f"\n{CoqGPTRequestGrammar.end}"
            result : CoqGptRequest = self.run(message_temp, None)

            # VALIDATION: Reject tactics containing ✝ symbol
            if result.action == CoqGptRequestActions.RUN_TACTIC:
                for arg in result.args:
                    if '✝' in arg:
                        raise Exception(f"Invalid tactic: tactics cannot contain the ✝ symbol. Found in: {arg}")

            message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)

    def attempt_parsing(self, message):
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

            # VALIDATION: Reject tactics containing ✝ symbol
            if result.action == CoqGptRequestActions.RUN_TACTIC:
                for arg in result.args:
                    if '✝' in arg:
                        raise Exception(f"Invalid tactic: tactics cannot contain the ✝ symbol. Found in: {arg}")
        else:
            message += CoqGPTRequestGrammar.end
            result : CoqGptRequest = self.run(message, None)

            # VALIDATION: Reject tactics containing ✝ symbol
            if result.action == CoqGptRequestActions.RUN_TACTIC:
                for arg in result.args:
                    if '✝' in arg:
                        raise Exception(f"Invalid tactic: tactics cannot contain the ✝ symbol. Found in: {arg}")

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
    code = """
[RUN TACTIC]
    ring_nf
[END]"""
    result = grammar.compile(code)
    print(result)
    run_result = grammar.normal_parsing(code)
    print(run_result)
    result = grammar.compile(
"""
[GET DEFINITIONS AND THEOREMS]
[END]"""
    )
