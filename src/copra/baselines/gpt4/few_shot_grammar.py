#!/usr/bin/env python3

import typing
from itp_interface.rl.proof_action import ProofAction
from copra.prompt_generator.interpreter import Grammar
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class FewShotGptCoqKeywords(object):
    PROOF = "Proof."
    QED = "Qed."
    THEOREM = "[THEOREM]"
    DEFINITION = "[DEFINITION]"
    DEFINITIONS = "[DEFINITIONS]"
    LEMMAS = "[LEMMAS]"
    LEMMA = "[LEMMA]"
    END = "[END]"

class FewShotGptLeanKeywords(object):
    PROOF = "[PROOF]"
    QED = "[END]"
    THEOREM = "[THEOREM]"
    DEFINITION = "[DEFINITION]"
    DEFINITIONS = "[DEFINITIONS]"
    LEMMAS = "[LEMMAS]"
    LEMMA = "[LEMMA]"
    END = "[END]"
    INFORMAL_THEOREM = "[INFORMAL-THEOREM]"
    INFORMAL_PROOF = "[INFORMAL-PROOF]"

class FewShotGptIsabelleKeywords(object):
    PROOF = "[PROOF]"
    QED = "[END]"
    THEOREM = "[THEOREM]"
    DEFINITION = "[DEFINITION]"
    DEFINITIONS = "[DEFINITIONS]"
    LEMMAS = "[LEMMAS]"
    LEMMA = "[LEMMA]"
    END = "[END]"
    INFORMAL_THEOREM = "[INFORMAL-THEOREM]"
    INFORMAL_PROOF = "[INFORMAL-PROOF]"

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
    informal_theorem: typing.Optional[str] = None
    informal_proof: typing.Optional[str] = None

class FewShotGptResponseGrammar(Grammar):
    grammar = f"""
Prog:
  Thm String End
| Thm String DfnsResponses LmsResponses End
| Thm String InfThm String InfPrf String End;
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
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.COQ):
        self.language = language
        if language == ProofAction.Language.COQ:
            self.keywords = [FewShotGptCoqKeywords.THEOREM, FewShotGptCoqKeywords.DEFINITION, FewShotGptCoqKeywords.DEFINITIONS, FewShotGptCoqKeywords.LEMMA, FewShotGptCoqKeywords.LEMMAS, FewShotGptCoqKeywords.END]
        elif language == ProofAction.Language.LEAN:
            self.keywords = [FewShotGptLeanKeywords.THEOREM, FewShotGptLeanKeywords.DEFINITION, FewShotGptLeanKeywords.DEFINITIONS, FewShotGptLeanKeywords.LEMMA, FewShotGptLeanKeywords.LEMMAS, FewShotGptLeanKeywords.END, FewShotGptLeanKeywords.INFORMAL_THEOREM, FewShotGptLeanKeywords.INFORMAL_PROOF]
        elif language == ProofAction.Language.LEAN4:
            self.keywords = [FewShotGptLeanKeywords.THEOREM, FewShotGptLeanKeywords.DEFINITION, FewShotGptLeanKeywords.DEFINITIONS, FewShotGptLeanKeywords.LEMMA, FewShotGptLeanKeywords.LEMMAS, FewShotGptLeanKeywords.END]
        elif language == ProofAction.Language.ISABELLE:
            self.keywords = [FewShotGptIsabelleKeywords.THEOREM, FewShotGptIsabelleKeywords.DEFINITION, FewShotGptIsabelleKeywords.DEFINITIONS, FewShotGptIsabelleKeywords.LEMMA, FewShotGptIsabelleKeywords.LEMMAS, FewShotGptIsabelleKeywords.END, FewShotGptIsabelleKeywords.INFORMAL_THEOREM, FewShotGptIsabelleKeywords.INFORMAL_PROOF]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if language == ProofAction.Language.COQ:
            terminals = f"""
terminals
End: "{FewShotGptCoqKeywords.END}";
Thm: "{FewShotGptCoqKeywords.THEOREM}";
Dfn: "{FewShotGptCoqKeywords.DEFINITION}";
Dfns: "{FewShotGptCoqKeywords.DEFINITIONS}";
Lm: "{FewShotGptCoqKeywords.LEMMA}";
Lms: "{FewShotGptCoqKeywords.LEMMAS}";
String:;
"""
        elif language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
End: "{FewShotGptLeanKeywords.END}";
Thm: "{FewShotGptLeanKeywords.THEOREM}";
Dfn: "{FewShotGptLeanKeywords.DEFINITION}";
Dfns: "{FewShotGptLeanKeywords.DEFINITIONS}";
Lm: "{FewShotGptLeanKeywords.LEMMA}";
Lms: "{FewShotGptLeanKeywords.LEMMAS}"
InfThm: "{FewShotGptLeanKeywords.INFORMAL_THEOREM}"
InfPrf: "{FewShotGptLeanKeywords.INFORMAL_PROOF}";
String:;
"""
        elif language == ProofAction.Language.LEAN4:
            terminals = f"""
terminals
End: "{FewShotGptLeanKeywords.END}";
Thm: "{FewShotGptLeanKeywords.THEOREM}";
Dfn: "{FewShotGptLeanKeywords.DEFINITION}";
Dfns: "{FewShotGptLeanKeywords.DEFINITIONS}";
Lm: "{FewShotGptLeanKeywords.LEMMA}";
Lms: "{FewShotGptLeanKeywords.LEMMAS}"
InfThm: "{FewShotGptLeanKeywords.INFORMAL_THEOREM}"
InfPrf: "{FewShotGptLeanKeywords.INFORMAL_PROOF}";
String:;
"""
        elif language == ProofAction.Language.ISABELLE:
            terminals = f"""
terminals
End: "{FewShotGptIsabelleKeywords.END}";
Thm: "{FewShotGptIsabelleKeywords.THEOREM}";
Dfn: "{FewShotGptIsabelleKeywords.DEFINITION}";
Dfns: "{FewShotGptIsabelleKeywords.DEFINITIONS}";
Lm: "{FewShotGptIsabelleKeywords.LEMMA}";
Lms: "{FewShotGptIsabelleKeywords.LEMMAS}";
InfThm: "{FewShotGptIsabelleKeywords.INFORMAL_THEOREM}"
InfPrf: "{FewShotGptIsabelleKeywords.INFORMAL_PROOF}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.COQ:
            self.END = FewShotGptCoqKeywords.END
            self.THEOREM = FewShotGptCoqKeywords.THEOREM
            self.DEFINITION = FewShotGptCoqKeywords.DEFINITION
            self.DEFINITIONS = FewShotGptCoqKeywords.DEFINITIONS
            self.LEMMA = FewShotGptCoqKeywords.LEMMA
            self.LEMMAS = FewShotGptCoqKeywords.LEMMAS
        elif language == ProofAction.Language.LEAN:
            self.END = FewShotGptLeanKeywords.END
            self.THEOREM = FewShotGptLeanKeywords.THEOREM
            self.DEFINITION = FewShotGptLeanKeywords.DEFINITION
            self.DEFINITIONS = FewShotGptLeanKeywords.DEFINITIONS
            self.LEMMA = FewShotGptLeanKeywords.LEMMA
            self.LEMMAS = FewShotGptLeanKeywords.LEMMAS
            self.INFORMAL_THEOREM = FewShotGptLeanKeywords.INFORMAL_THEOREM
            self.INFORMAL_PROOF = FewShotGptLeanKeywords.INFORMAL_PROOF
        elif language == ProofAction.Language.LEAN4:
            self.END = FewShotGptLeanKeywords.END
            self.THEOREM = FewShotGptLeanKeywords.THEOREM
            self.DEFINITION = FewShotGptLeanKeywords.DEFINITION
            self.DEFINITIONS = FewShotGptLeanKeywords.DEFINITIONS
            self.LEMMA = FewShotGptLeanKeywords.LEMMA
            self.LEMMAS = FewShotGptLeanKeywords.LEMMAS
            self.INFORMAL_THEOREM = FewShotGptLeanKeywords.INFORMAL_THEOREM
            self.INFORMAL_PROOF = FewShotGptLeanKeywords.INFORMAL_PROOF
        elif language == ProofAction.Language.ISABELLE:
            self.END = FewShotGptIsabelleKeywords.END
            self.THEOREM = FewShotGptIsabelleKeywords.THEOREM
            self.DEFINITION = FewShotGptIsabelleKeywords.DEFINITION
            self.DEFINITIONS = FewShotGptIsabelleKeywords.DEFINITIONS
            self.LEMMA = FewShotGptIsabelleKeywords.LEMMA
            self.LEMMAS = FewShotGptIsabelleKeywords.LEMMAS
            self.INFORMAL_THEOREM = FewShotGptIsabelleKeywords.INFORMAL_THEOREM
            self.INFORMAL_PROOF = FewShotGptIsabelleKeywords.INFORMAL_PROOF
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = FewShotGptResponseGrammar.grammar + terminals
        super(FewShotGptResponseGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: FewShotGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: int = 4) -> str:
        assert coq_gpt_response.theorem is not None
        char_cnt = max_token_cnt * characters_per_token if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            self.THEOREM : [],
            self.DEFINITIONS : [],
            self.LEMMAS : [],
        }
        lines_order = [
            self.THEOREM,
            self.DEFINITIONS,
            self.LEMMAS
        ]
        priority_order_lo_hi = [
            self.LEMMAS,
            self.DEFINITIONS,
            self.THEOREM
        ]
        if self.language == ProofAction.Language.LEAN or self.language == ProofAction.Language.ISABELLE or self.language == ProofAction.Language.LEAN4:
            lines_map[self.INFORMAL_THEOREM] = []
            lines_map[self.INFORMAL_PROOF] = []
            lines_order = [
                self.THEOREM,
                self.INFORMAL_THEOREM,
                self.INFORMAL_PROOF,
                self.DEFINITIONS,
                self.LEMMAS
            ]
            priority_order_lo_hi = [
                self.LEMMAS,
                self.DEFINITIONS,
                self.INFORMAL_PROOF,
                self.INFORMAL_THEOREM,
                self.THEOREM
            ]
        new_line = f"{self.THEOREM}\n{coq_gpt_response.theorem}"
        lines_map[self.THEOREM] = [new_line]

        if len(coq_gpt_response.defintions) > 0:
            lines_map[self.DEFINITIONS] = [f"\n{self.DEFINITIONS}"]
        for idx, dfn in enumerate(coq_gpt_response.defintions):
            if k is not None and idx >= k:
                break
            lines_map[self.DEFINITIONS].append(f"{self.DEFINITION} {dfn}")

        if len(coq_gpt_response.lemmas) > 0:
            lines_map[self.LEMMAS] = [f"\n{self.LEMMAS}"]
        for idx, lm in enumerate(coq_gpt_response.lemmas):
            if k is not None and idx >= k:
                break
            lines_map[self.LEMMAS].append(f"{self.LEMMA} {lm}")
        
        if self.language == ProofAction.Language.LEAN or self.language == ProofAction.Language.ISABELLE or self.language == ProofAction.Language.LEAN4:
            if coq_gpt_response.informal_theorem is not None:
                lines_map[self.INFORMAL_THEOREM] = ["\n" + self.INFORMAL_THEOREM]
                lines_map[self.INFORMAL_THEOREM].append(coq_gpt_response.informal_theorem)
            if coq_gpt_response.informal_proof is not None:
                lines_map[self.INFORMAL_PROOF] = [self.INFORMAL_PROOF]
                lines_map[self.INFORMAL_PROOF].append(coq_gpt_response.informal_proof)

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


class FewShotGptRequestGrammar(Grammar):
    grammar = f"""
Prog: 
    Proof String Qed;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.COQ, enable_defensive_parsing: bool = False):
        self.language = language
        if language == ProofAction.Language.COQ:
            self.keywords = [FewShotGptCoqKeywords.PROOF, FewShotGptCoqKeywords.QED]
        elif language == ProofAction.Language.LEAN:
            self.keywords = [FewShotGptLeanKeywords.PROOF, FewShotGptLeanKeywords.QED]
        elif language == ProofAction.Language.LEAN4:
            self.keywords = [FewShotGptLeanKeywords.PROOF, FewShotGptLeanKeywords.END]
        elif language == ProofAction.Language.ISABELLE:
            self.keywords = [FewShotGptIsabelleKeywords.PROOF, FewShotGptIsabelleKeywords.QED]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if self.language == ProofAction.Language.COQ:
            terminals = f"""
terminals
Proof: "{FewShotGptCoqKeywords.PROOF}";
Qed: "{FewShotGptCoqKeywords.QED}";
String:;
"""
        elif self.language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
Proof: "{FewShotGptLeanKeywords.PROOF}";
Qed: "{FewShotGptLeanKeywords.QED}";
String:;
"""
        elif self.language == ProofAction.Language.LEAN4:
            terminals = f"""
terminals
Proof: "{FewShotGptLeanKeywords.PROOF}";
Qed: "{FewShotGptLeanKeywords.QED}";
String:;
"""
        elif self.language == ProofAction.Language.ISABELLE:
            terminals = f"""
terminals
Proof: "{FewShotGptIsabelleKeywords.PROOF}";
Qed: "{FewShotGptIsabelleKeywords.QED}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.COQ:
            self.PROOF = FewShotGptCoqKeywords.PROOF
            self.QED = FewShotGptCoqKeywords.QED
        elif language == ProofAction.Language.LEAN:
            self.PROOF = FewShotGptLeanKeywords.PROOF
            self.QED = FewShotGptLeanKeywords.QED
        elif language == ProofAction.Language.LEAN4:
            self.PROOF = FewShotGptLeanKeywords.PROOF
            self.QED = FewShotGptLeanKeywords.END
        elif language == ProofAction.Language.ISABELLE:
            self.PROOF = FewShotGptIsabelleKeywords.PROOF
            self.QED = FewShotGptIsabelleKeywords.QED
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = FewShotGptRequestGrammar.grammar + terminals
        self.enable_defensive_parsing = enable_defensive_parsing
        super(FewShotGptRequestGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)

    def _parse_expr(self, nonTerminal: str, nodes) -> FewShotGptRequest:
        if nonTerminal == "Prog":
            assert len(nodes) >= 3
            if self.language == ProofAction.Language.COQ:
                actions = str(nodes[1]).strip() + f"\n{self.QED}"
            elif self.language == ProofAction.Language.LEAN:
                actions = str(nodes[1]).strip()
                if actions.startswith("begin"):
                    actions = actions[len("begin"):]
                if not actions.endswith("end"):
                    actions += "end"
            elif self.language == ProofAction.Language.LEAN4:
                actions = str(nodes[1]).strip()
                if actions.startswith("by"):
                    actions = actions[len("by"):]
            elif self.language == ProofAction.Language.ISABELLE:
                actions = str(nodes[1]).strip().strip("proof -").strip()
                if not actions.endswith("qed"):
                    actions += "qed"
            else:
                raise NotImplementedError(f"language {self.language} not supported")
            proof_action = ProofAction(ProofAction.ActionType.RUN_TACTIC, self.language, tactics=[actions])
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
        return f"{self.PROOF}\n{coq_gpt_request.proof_string}"

    def get_openai_request(self, message_response: str) -> typing.Tuple[FewShotGptRequest, str]:
        message, _ = message_response
        if self.enable_defensive_parsing:
            return self.defensive_parsing(message)
        else:
            return self.normal_parsing(message)
        
    def normal_parsing(self, message):
        message_temp = message
        message_temp += f"\n{self.QED}"
        result : FewShotGptRequest = self.run(message_temp, None)            
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
                if message_temp.endswith(self.QED):
                    # Just in case the LLM doesn't remove the stop token
                    message_temp = message_temp.strip(self.QED)
                message_temp += f"\n{self.QED}"
                result : FewShotGptRequest = self.run(message_temp, None)            
                message_temp = self.generate_message_from_gpt_request(result)
                message_parsed = True
            except:
                message_parsed = False
            if message_parsed:
                break
        if not message_parsed:
            message_temp = message[start_idx:end_idx]
            message_temp += f"\n{self.QED}"
            result : FewShotGptRequest = self.run(message_temp, None)            
            message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)
    
    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(self.QED), "Message must end with end token"
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

    code = """
[PROOF]
begin
  rw [h₂, h₃] at h₁,
  rw h₁,
  norm_num
end
[END]
"""
    grammar = FewShotGptRequestGrammar(language=ProofAction.Language.LEAN)
    result = grammar.compile(code)
    print(result)
    run_result = grammar.run(code, None)
    print(run_result)

    code = """
[PROOF]
proof -
  have "f = 11 - 3*z" using h0 by (simp add:algebra_simps)
  then have "3*((11 - 3*z) - 1) - 5*z = -68"
    using h1 by simp
  then have "z=7" by (simp add:algebra_simps)
  moreover then have "f=-10" using h0
    apply (simp add:field_simps )
    by (metis add_diff_cancel diff_0)
  ultimately show ?thesis by simp
qed
[END]
"""
    grammar = FewShotGptRequestGrammar(language=ProofAction.Language.ISABELLE)
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

    response_grammar = FewShotGptResponseGrammar(language=ProofAction.Language.LEAN)
    response = FewShotGptResponse(theorem="""
some_random_theorem
  (b a x : ℝ)
  (h₀ : 0 < b ∧ 0 < a ∧ 0 < v)
  (h₁ : x = 1 / 2 * (b * a))
  (h₂ : b = 10)
  (h₃ : a = 5) :
  x = 25 :=
""", 
        defintions=[], 
        lemmas=[])
    response_text = response_grammar.format_as_per_grammar(response, k=3)
    print('-' * 80)
    print(response_text)
    print('-' * 80)

    response = FewShotGptResponse(theorem="""
some_random_theorem
(a : ℕ) : a + 0 = a :=
""", 
        defintions=[], 
        lemmas=[],
        informal_theorem="""For any natural number a, prove that a + 0 = a.""",
        informal_proof="""Let a be an arbitrary natural number.
We must show that a + 0 = a. We will prove this by induction on a.
First, suppose a = 0. We must show that 0 + 0 = 0. This follows directly from the definition of addition.
Next, by induction hypothesis, suppose that a + 0 = a. We must show that (a + 1) + 0 = a + 1. 
By associativity, (a + 1) + 0 = a + (1 + 0).
We know that 1 + 0 = 1. Therefore, (a + 1) + a = a + 1, as required.
""")
    response_text = response_grammar.format_as_per_grammar(response, k=3)
    print('-' * 80)
    print(response_text)
    print('-' * 80)

    response_grammar = FewShotGptResponseGrammar(language=ProofAction.Language.ISABELLE)
    response = FewShotGptResponse(theorem="""
theorem aime_1983_p1:
  fixes x y z w :: nat
  assumes ht : "1 < x \<and> 1 < y \<and> 1 < z"
    and hw : "0 \<le> w"
    and h0 : "ln w / ln x = 24"
    and h1 : "ln w / ln y = 40"
    and h2 : "ln w / ln (x * y * z) = 12"
  shows "ln w / ln z = 60"
""", 
        defintions=[], 
        lemmas=[])
    response_text = response_grammar.format_as_per_grammar(response, k=3)
    print('-' * 80)
    print(response_text)
    print('-' * 80)