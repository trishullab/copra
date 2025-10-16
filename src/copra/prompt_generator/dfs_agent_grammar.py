#!/usr/bin/env python3

import os
from copra.prompt_generator.interpreter import Grammar
from copra.prompt_generator.grammar_utils import StringParser


class DfsAgentGrammar(Grammar):
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
    Conv1Term String;
Conv2:
    Conv2Term String;

terminals
Conv1Term: "`{0}`";
Conv2Term: "`{1}`";
ConvStart: "`conv start`";
ConvEnd: "`conv end`";
String:;
"""

    @staticmethod
    def get_string_parser(keywords):
        """Create a picklable string parser for the given keywords."""
        return StringParser(keywords)

    def __init__(self, user_name: str = 'example_user', agent_name: str = 'example_assistant'):
        self.keywords = [f"`{user_name}`", f"`{agent_name}`", "`conv start`", "`conv end`"]
        self.user_name = user_name
        self.agent_name = agent_name
        recognizers = {
            'String': DfsAgentGrammar.get_string_parser(self.keywords)
        }
        super(DfsAgentGrammar, self).__init__(DfsAgentGrammar.grammar.format(user_name, agent_name), self.keywords, recognizers=recognizers)
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
    
    def get_openai_conv_messages(self, file_path: str, role: str = "system"):
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        with open(file_path, "r") as f:
            conv = f.read()
        result = self.run(conv, role)
        return result

    def get_openai_main_message(self, file_path: str, role: str = "system"):
        with open(file_path) as f:
            main_content = f.read()
        return self.get_openai_main_message_from_string(main_content, role)

    def get_openai_main_message_from_string(self, content: str, role: str = "system", name: str = None):
        if name is None:
            return {
                'role': role,
                'content': content
            }
        else:
            return {
                'role': role,
                'name': name,
                'content': content
            }

    def parse_openai_messages(self, messages: list, role: str = "assistant"):
        assert isinstance(messages, list), f"Messages must be a list. Got {type(messages)}"
        result = [(message["content"], message["finish_reason"]) for message in messages if message['role'] == role]
        return result

if __name__ == "__main__":
    grammar = DfsAgentGrammar("example_user", "example_assistant")
    conv_file = "data/prompts/conversation/coq-proof-agent-example-long-conv-dfs.md"
    assert os.path.exists(conv_file), f"File {conv_file} does not exist"
    result = grammar.get_openai_conv_messages(conv_file, "system")
    print(result)