#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
from src.prompt_generator.interpreter import Grammar

class AgentGrammar(Grammar):
    grammar = """
Prog:
  ConvStart Convs1 ConvEnd
| ConvStart Convs2 ConvEnd;
Convs1:
  Conv1
| Conv1 Conv2
| Conv1 Conv2 Convs1;
Convs2:
  Conv2
| Conv2 Conv1
| Conv2 Conv1 Convs2;
Conv1:
    '`{0}`' String;
Conv2:
    '`{1}`' String;
terminals
ConvStart: "`conv start`";
ConvEnd: "`conv end`";
String:;
"""

    def get_string_parser(keywords):
        def string_parser(keywords, text, pos):
            last = pos
            while last < len(text):
                while last < len(text) and text[last] != '`':
                    last += 1
                if last < len(text):
                    for keyword in keywords:
                        if text[last:].startswith(keyword):
                            return text[pos:last]
                    last += 1
        return lambda text, pos: string_parser(keywords, text, pos)

    def __init__(self, user_name: str = 'example_user', agent_name: str = 'example_assistant'):
        self.keywords = [f"`{user_name}`", f"`{agent_name}`", "`conv start`", "`conv end`"]
        self.user_name = user_name
        self.agent_name = agent_name
        recognizers = {
            'String': AgentGrammar.get_string_parser(self.keywords)
        }
        super(AgentGrammar, self).__init__(AgentGrammar.grammar.format(user_name, agent_name), self.keywords, recognizers=recognizers)
    pass

    def _parse_expr(self, nonTerminal, nodes, role, context):
        if nonTerminal == "Conv1" or nonTerminal == "Conv2":
            assert len(nodes) == 2
            context.append({
                'role': role,
                'name': str(nodes[0]).strip('`'),
                'content' : nodes[1].strip()
            })
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
    
    def get_action(self, inp):
        assert isinstance(inp, str), f"Input must be a string. Got {type(inp)}"
        assert inp in {"system", "user"}, f"Input must be either 'system' or 'user'. Got {inp}"
        role = inp
        context = []
        actions = {
            "Prog": lambda _, nodes: context,
            "Conv1": lambda _, nodes: self._parse_expr('Conv1', nodes, role, context),
            "Conv2": lambda _, nodes: self._parse_expr('Conv2', nodes, role, context),
            "String": lambda _, nodes: str(nodes) # Since this is always a string
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, list), f"Result must be a list. Got {type(result)}"
        return result
    
    def get_openai_sample_messages(self, file_path: str, role: str = "system"):
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        with open(file_path, "r") as f:
            conv = f.read()
        result = self.run(conv, role)
        return result

if __name__ == "__main__":
    os.chdir(root_dir)
    grammar = AgentGrammar("example_user", "example_assistant")
    result = grammar.get_openai_sample_messages("data/prompts/conversation/coq-proof-agent-example.md", "system")
    print(result)