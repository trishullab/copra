#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.rl.proof_action import ProofAction
from src.prompt_generator.interpreter import Grammar
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class FewShotGptKeywords(object):
    PROOF = "Proof."
    QED = "Qed."
    THEOREM = "[THEOREM]"
    DEFINITION = "[DEFINITION]"
    DEFINITIONS = "[DEFINITIONS]"
    LEMMAS = "[LEMMAS]"
    LEMMA = "[LEMMA]"
    END = "[END]"

@dataclass_json
@dataclass
class FewShotGptRequest(object):
    action : ProofAction
    proof_string : str

@dataclass_json
@dataclass
class FewShotGptResponse(object):
    theorem: str
    defintions: typing.List[str] = field(default_factory=list)
    lemmas: typing.List[str] = field(default_factory=list)

class FewShotGptResponseGrammar(Grammar):
    grammar = f"""
Prog:
  Thm String End
| Thm String DfnsResponses LmsResponses End;
DfnsResponses:
    Dfns DfnResponses
|   EMPTY;
DfnResponses:
    DfnResponse
|   DfnResponse DfnResponses;
DfnResponse:
    Dfn String;
LmsResponses:
    Lms LmResponses
|   EMPTY;
LmResponses:
    LmResponse
|   LmResponse LmResponses;
LmResponse:
    Lm String;

terminals
End: "{FewShotGptKeywords.END}";
Thm: "{FewShotGptKeywords.THEOREM}";
Dfn: "{FewShotGptKeywords.DEFINITION}";
Dfns: "{FewShotGptKeywords.DEFINITIONS}";
Lm: "{FewShotGptKeywords.LEMMA}";
Lms: "{FewShotGptKeywords.LEMMAS}";
String:;
"""

    keywords = [FewShotGptKeywords.THEOREM, FewShotGptKeywords.DEFINITION, FewShotGptKeywords.DEFINITIONS, FewShotGptKeywords.LEMMA, FewShotGptKeywords.LEMMAS, FewShotGptKeywords.END]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
            for keyword in FewShotGptResponseGrammar.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self):
        recognizers = {
            'String': FewShotGptResponseGrammar.before_keyword
        }
        super(FewShotGptResponseGrammar, self).__init__(FewShotGptResponseGrammar.grammar, FewShotGptResponseGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: FewShotGptResponse, k: typing.Optional[int] = None) -> str:
        text = ""
        assert coq_gpt_response.theorem is not None
        text = f"{FewShotGptKeywords.THEOREM}\n{coq_gpt_response.theorem}\n"
        lines = []
        if len(coq_gpt_response.defintions) > 0:
            lines.append(f"\n{FewShotGptKeywords.DEFINITIONS}")
        for idx, dfn in enumerate(coq_gpt_response.defintions):
            if k is not None and idx >= k:
                break
            lines.append(f"{FewShotGptKeywords.DEFINITION} {dfn}")
        if len(coq_gpt_response.lemmas) > 0:
            lines.append(f"\n{FewShotGptKeywords.LEMMAS}")
        for idx, lm in enumerate(coq_gpt_response.lemmas):
            if k is not None and idx >= k:
                break
            lines.append(f"{FewShotGptKeywords.LEMMA} {lm}")
        if len(lines) > 0:
            gls_args = '\n'.join(lines)
            text += gls_args
        text += f"\n\n{FewShotGptKeywords.END}"
        self.compile(text)
        return text

class FewShotGptRequestGrammar(Grammar):
    grammar = f"""
Prog: 
    Proof String Qed;

terminals
Proof: "{FewShotGptKeywords.PROOF}";
Qed: "{FewShotGptKeywords.QED}";
String:;
"""
    keywords = [FewShotGptKeywords.PROOF, FewShotGptKeywords.QED]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
            for keyword in FewShotGptRequestGrammar.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self):
        recognizers = {
            'String': FewShotGptRequestGrammar.before_keyword
        }
        super(FewShotGptRequestGrammar, self).__init__(FewShotGptRequestGrammar.grammar, FewShotGptRequestGrammar.keywords, recognizers=recognizers)

    def _parse_expr(self, nonTerminal: str, nodes) -> FewShotGptRequest:
        if nonTerminal == "Prog":
            assert len(nodes) >= 3
            actions = str(nodes[1]).strip()
            proof_action = ProofAction(ProofAction.ActionType.RUN_TACTIC, tactics=[actions])
            return FewShotGptRequest(action=proof_action, proof_string=actions)
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
        
    def get_action(self, inp=None):
        actions = {
            "Prog": lambda _, nodes: self._parse_expr('Prog', nodes)
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, FewShotGptRequest), f"Result must be a FewShotGptRequest. Got {type(result)}"
        return result
    
    def generate_message_from_gpt_request(self, coq_gpt_request: FewShotGptRequest) -> str:
        return f"{FewShotGptKeywords.PROOF}\n{coq_gpt_request.proof_string}\n{FewShotGptKeywords.QED}"

    def get_openai_request(self, message_response: str) -> typing.Tuple[FewShotGptRequest, str]:
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
                        parsable_message = message[:idx] + f"\n{FewShotGptKeywords.QED}"
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
                result : FewShotGptRequest = self.run(message, None)
            else:
                message += FewShotGptKeywords.QED
                result : FewShotGptRequest = self.run(message, None)
        else:
            message += FewShotGptKeywords.QED
            result : FewShotGptRequest = self.run(message, None)            
        message = self.generate_message_from_gpt_request(result)
        return (result, message)
    
    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(FewShotGptKeywords.QED), "Message must end with end token"
            result : FewShotGptRequest = self.run(message, None)
            results.extend(result.args)
        return results 

if __name__ == "__main__":
    code = """
Proof.
    intros.
    induction a.
    simpl.
    reflexivity.
    simpl.
    reflexivity.
Qed."""
    grammar = FewShotGptRequestGrammar()
    result = grammar.compile(code)
    print(result)
    run_result = grammar.run(code, None)
    print(run_result)

    response_grammar = FewShotGptResponseGrammar()
    response = FewShotGptResponse(theorem="algb_nat_zero : forall a, 0 + a = a.", 
        defintions=["nat : Set", "0 : nat", "S : nat -> nat", "plus : nat -> nat -> nat"], 
        lemmas=[
            "plus_O_n: forall n : nat, 0 + n = n",
            "plus_n_O: forall n : nat, n = n + 0",
            "plus_n_Sm: forall n m : nat, S (n + m) = n + S m",
            "plus_Sn_m: forall n m : nat, S n + m = S (n + m)"
        ])
    response_text = response_grammar.format_as_per_grammar(response, k=3)
    print('-' * 80)
    print(response_text)
    print('-' * 80)
