#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import parglare as Parglare

class Grammar:
    def __init__(self, grammar, keywords, recognizers=None, debug=False):
        assert isinstance(grammar, str)
        self.grammar = grammar
        self.keywords = keywords
        self.recognizers = recognizers
        self.debug = debug

    def get_action(self,inp):
        raise Exception("override this")
    
    def interpret_result(self, result):
        # Do nothing
        return result

    def _get_parser(self, actions={}):
        if self.recognizers is not None:
            g = Parglare.Grammar.from_string(self.grammar, recognizers=self.recognizers, debug=self.debug)
        else:
            g = Parglare.Grammar.from_string(self.grammar)
        parser = Parglare.Parser(g, debug=self.debug, actions=actions)
        return parser

    def compile(self, code):
        parser = self._get_parser()
        result = parser.parse(code)
        return result

    def run(self, code, inp):
        actions = self.get_action(inp)
        parser = self._get_parser(actions)
        result = parser.parse(code)
        return self.interpret_result(result)
    
    def get_syntax_tree(self, code):
        g = Parglare.Grammar.from_string(self.grammar)
        parser = Parglare.Parser(g, debug=False, build_tree=True)
        tree = parser.parse(code)
        return tree

    def get_success_program_idx(self, programs):
        for idx, soln in enumerate(programs):
            try:
                examples = soln["Examples"]
                code = soln["Program"]
                all_example_match = False
                for ex in examples:
                    inp = ex["In"]
                    expected_out = ex["Out"]
                    actual_out = self.run(code, inp)
                    if str(actual_out) == str(expected_out):
                        all_example_match = True
                    else:
                        break
                if all_example_match:
                    yield idx
            except:
                pass
    
    def get_success_programs_unique(self, programs):
        prog_set = set()
        for idx in self.get_success_program_idx(programs):
            prog = programs[idx]
            if prog["Program"] not in prog_set:
                prog_set.add(prog["Program"])
                yield prog
        prog_set.clear()
    
    def check_preconditions(self, precondition: str):
        return "[]"

    def check_postconditions(self, postcondition: str, inp: str, output: str):
        pass
