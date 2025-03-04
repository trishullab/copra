#!/usr/bin/env python3

import typing
from itp_interface.rl.proof_action import ProofAction
from copra.prompt_generator.interpreter import Grammar
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class InformalFewShotGptLeanKeywords(object):
    THEOREM = "[THEOREM]"
    PROOF = "[PROOF]"
    END = "[END]"

class InformalFewShotGptIsabelleKeywords(object):
    THEOREM = "[THEOREM]"
    PROOF = "[PROOF]"
    END = "[END]"

@dataclass_json
@dataclass
class InformalFewShotGptResponse(object):
    theorem: str

class InformalFewShotGptResponseGrammar(Grammar):
    grammar = f"""
Prog:
  Theorem String End;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.LEAN):
        self.language = language
        if language == ProofAction.Language.LEAN:
            self.keywords = [InformalFewShotGptLeanKeywords.THEOREM, InformalFewShotGptLeanKeywords.END]
        elif language == ProofAction.Language.ISABELLE:
            self.keywords = [InformalFewShotGptIsabelleKeywords.THEOREM, InformalFewShotGptIsabelleKeywords.END]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
End: "{InformalFewShotGptLeanKeywords.END}";
Theorem: "{InformalFewShotGptLeanKeywords.THEOREM}";
String:;
"""
        elif language == ProofAction.Language.ISABELLE:
            terminals = f"""
terminals
End: "{InformalFewShotGptIsabelleKeywords.END}";
Theorem: "{InformalFewShotGptIsabelleKeywords.THEOREM}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.LEAN:
            self.END = InformalFewShotGptLeanKeywords.END
            self.THEOREM = InformalFewShotGptLeanKeywords.THEOREM
        elif language == ProofAction.Language.ISABELLE:
            self.END = InformalFewShotGptIsabelleKeywords.END
            self.THEOREM = InformalFewShotGptIsabelleKeywords.THEOREM
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = InformalFewShotGptResponseGrammar.grammar + terminals
        super(InformalFewShotGptResponseGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: InformalFewShotGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: int = 4) -> str:
        assert coq_gpt_response.theorem is not None
        char_cnt = max_token_cnt * characters_per_token if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            self.THEOREM : [],
            self.END : []
        }
        lines_order = [
            self.THEOREM,
            self.END
        ]
        priority_order_lo_hi = [
            self.THEOREM,
            self.END
        ]
        
        new_line = f"{self.THEOREM}\n{coq_gpt_response.theorem}"
        lines_map[self.THEOREM] = [new_line]

        keywords = [keyword for keyword in lines_map.keys()]
        # Convert all the lines under each keyword to a single string
        for keyword in keywords:
            lines_map[keyword] = "\n".join(lines_map[keyword])
        # Frame the first prompt version without any token limit
        text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0])
        
        # Trim the lines based on the max_token_cnt
        if char_cnt is not None and len(text) > char_cnt:
            _idx = 0
            diff = len(text) - char_cnt
            while _idx < len(priority_order_lo_hi) and diff > 0:
                trim_part = priority_order_lo_hi[_idx]
                if trim_part in lines_map:
                    lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0])
                diff = len(text) - char_cnt
                _idx += 1
        text += f"\n\n{self.END}"
        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None and len(text) > char_cnt:
            text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text


class InformalFewShotGptRequestGrammar(Grammar):
    grammar = f"""
Prog: 
    Proof String End;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.LEAN, enable_defensive_parsing: bool = False):
        self.language = language
        if language == ProofAction.Language.LEAN:
            self.keywords = [InformalFewShotGptLeanKeywords.PROOF, InformalFewShotGptLeanKeywords.END]
        elif language == ProofAction.Language.ISABELLE:
            self.keywords = [InformalFewShotGptIsabelleKeywords.PROOF, InformalFewShotGptIsabelleKeywords.END]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if self.language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
Proof: "{InformalFewShotGptLeanKeywords.PROOF}";
End: "{InformalFewShotGptLeanKeywords.END}";
String:;
"""
        elif self.language == ProofAction.Language.ISABELLE:
            terminals = f"""
terminals
Proof: "{InformalFewShotGptIsabelleKeywords.PROOF}";
End: "{InformalFewShotGptIsabelleKeywords.END}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.LEAN:
            self.PROOF = InformalFewShotGptLeanKeywords.PROOF
            self.END = InformalFewShotGptLeanKeywords.END
        elif language == ProofAction.Language.ISABELLE:
            self.PROOF = InformalFewShotGptIsabelleKeywords.PROOF
            self.END = InformalFewShotGptIsabelleKeywords.END
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = InformalFewShotGptRequestGrammar.grammar + terminals
        self.enable_defensive_parsing = enable_defensive_parsing
        super(InformalFewShotGptRequestGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)

    def _parse_expr(self, nonTerminal: str, nodes) -> ProofAction:
        if nonTerminal == "Prog":
            assert len(nodes) >= 3
            proof = str(nodes[1]).strip()
            return ProofAction(action_type=ProofAction.ActionType.INFORMAL, language=self.language, proof=proof)
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
        
    def get_action(self, inp=None):
        actions = {
            "Prog": lambda _, nodes: self._parse_expr('Prog', nodes)
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, ProofAction), f"Result must be a ProofAction. Got {type(result)}"
        return result
    
    def generate_message_from_gpt_request(self, coq_gpt_request: ProofAction) -> str:
        assert coq_gpt_request.action_type == ProofAction.ActionType.INFORMAL, f"action_type must be {ProofAction.ActionType.INFORMAL}, not {coq_gpt_request.action_type}"
        return f"{self.PROOF}\n{coq_gpt_request.kwargs['proof']}\n{self.END}"

    def get_openai_request(self, message_response: str) -> typing.Tuple[ProofAction, str]:
        message, _ = message_response
        if self.enable_defensive_parsing:
            return self.defensive_parsing(message)
        else:
            return self.normal_parsing(message)
    
    def normal_parsing(self, message):
        message_temp = message
        message_temp += f"\n{self.END}"
        result : ProofAction = self.run(message_temp, None)            
        message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)

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
                if message_temp.endswith(self.END):
                    # Just in case the LLM doesn't remove the stop token
                    message_temp = message_temp.strip(self.END)
                message_temp += f"\n{self.END}"
                result : ProofAction = self.run(message_temp, None)            
                message_temp = self.generate_message_from_gpt_request(result)
                message_parsed = True
            except:
                message_parsed = False
            if message_parsed:
                break
        if not message_parsed:
            message_temp = message[start_idx:end_idx]
            message_temp += f"\n{self.END}"
            result : ProofAction = self.run(message_temp, None)            
            message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)
    
    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(self.END), "Message must end with end token"
            result : ProofAction = self.run(message, None)
            results.extend(result.args)
        return results 

if __name__ == "__main__":
    code = """
[PROOF]
Suppose a and b are even numbers. 
Then there exist integers m and n such that a = 2 * m and b = 2 * n. 
Then a + b = 2 * m + 2 * n = 2 * (m + n). Since m + n is an integer, a + b is even.
[END]"""
    grammar = InformalFewShotGptRequestGrammar()
    result = grammar.compile(code)
    print(result)
    run_result = grammar.run(code, None)
    print(run_result)

    response_grammar = InformalFewShotGptResponseGrammar()
    response = InformalFewShotGptResponse(theorem="Sum of two even numbers is even.")
    response_text = response_grammar.format_as_per_grammar(response, k=3)
    print('-' * 80)
    print(response_text)
    print('-' * 80)