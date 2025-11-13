import typing
from enum import Enum
from copra.prompt_generator.dfs_gpt_response_grammar import CoqGptResponse, CoqGptResponseActions
from copra.prompt_generator.interpreter import Grammar

class SimpleResponseGrammar(Grammar):
    """
    A simplified response grammar for parsing GPT responses in a DFS proof search context.
    This grammar is designed to handle simpler response formats while maintaining
    compatibility with the existing DFS framework.
    """
    class Keywords(Enum):
        GOALS_TO_PROVE = "Goals to prove:"
        PROOF_SEPERATOR = "⊢"
        WRONG_SYMBOL = "[❌]"
        STEPS = "[STEPS]"
        INCORRECT_STEPS = "[INCORRECT STEPS]"
        ERROR = "[ERROR]"
        ERROR_MESSAGE = "[ERROR MESSAGE]"
        SUCCESS = "[SUCCESS]"
        END = "[END]"
        DESCRIPTION = "[DESCRIPTION]"
        LAST_STEP = "[LAST STEP]"
        INFORMAL_PROOF = "[INFORMAL-PROOF]"
        INFORMAL_THEOREM = "[INFORMAL-THEOREM]"

        def __str__(self) -> str:
            return self.value

    grammar = f"""
Prog:
  GoalsResponse
| ErrorResponse
| String Prog
| Prog String Prog;
ErrorResponse:
   Error String End
|  Error ErrorString End;
GoalsResponse:
 Goals GoalResponses StepsResponses IncorrectStepsResponses LastResponse End;
GoalResponses:
  String
| EMPTY;
IncorrectStepsResponses:
    IncrctStps WrongResponses
|   EMPTY;
LastResponse:
    LastStep String ErrorMessage String
|   LastStep String Success
|   EMPTY;
StepsResponses:
    Stps String
|   EMPTY;
WrongResponses:
   WrongResponse WrongResponses
|  EMPTY;
WrongResponse:
    WrongSymbol String;

terminals
int: /\d+/;
Goals: "{Keywords.GOALS_TO_PROVE.value}";
Stps: "[STEPS]";
IncrctStps: "[INCORRECT STEPS]";
Error: "[ERROR]";
ErrorMessage: "[ERROR MESSAGE]";
Success: "[SUCCESS]";
LastStep: "[LAST STEP]";
End: "[END]";
WrongSymbol: "{Keywords.WRONG_SYMBOL.value}";
String:;
ErrorString:;
"""
    proof_separator = Keywords.PROOF_SEPERATOR.value
    keywords = [keyword.value for keyword in Keywords]

    @staticmethod
    def before_keyword(text: str, pos: int):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in SimpleResponseGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1
    
    @staticmethod
    def error_string(text: str, pos: int):
        last = pos
        while last < len(text):
            while last < len(text) and text[last] != '[':
                last += 1
            if last < len(text) and text[last:].startswith("[END]") and text[last:].endswith("[END]"):
                return text[pos:last]
            last += 1

    def __init__(self):
        recognizers = {
            'String': self.before_keyword,
            'ErrorString': self.error_string
        }
        super(SimpleResponseGrammar, self).__init__(SimpleResponseGrammar.grammar, SimpleResponseGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: CoqGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        if coq_gpt_response.action == CoqGptResponseActions.ERROR:
            text = f"{SimpleResponseGrammar.Keywords.ERROR}\n{coq_gpt_response.message}\n{SimpleResponseGrammar.Keywords.END}"
        elif coq_gpt_response.action == CoqGptResponseActions.GOALS:
            lines_map = {
                SimpleResponseGrammar.Keywords.GOALS_TO_PROVE : [],
                SimpleResponseGrammar.Keywords.INFORMAL_THEOREM : [],
                SimpleResponseGrammar.Keywords.INFORMAL_PROOF : [],
                SimpleResponseGrammar.Keywords.STEPS : [],
                SimpleResponseGrammar.Keywords.INCORRECT_STEPS : [],
                SimpleResponseGrammar.Keywords.LAST_STEP : [],
                SimpleResponseGrammar.Keywords.SUCCESS : [],
                SimpleResponseGrammar.Keywords.ERROR_MESSAGE : []
            }
            lines_order = [
                SimpleResponseGrammar.Keywords.GOALS_TO_PROVE,
                SimpleResponseGrammar.Keywords.INFORMAL_THEOREM,
                SimpleResponseGrammar.Keywords.INFORMAL_PROOF,
                SimpleResponseGrammar.Keywords.STEPS,
                SimpleResponseGrammar.Keywords.INCORRECT_STEPS,
                SimpleResponseGrammar.Keywords.LAST_STEP,
                SimpleResponseGrammar.Keywords.SUCCESS,
                SimpleResponseGrammar.Keywords.ERROR_MESSAGE
            ]
            priority_order_lo_hi = [
                SimpleResponseGrammar.Keywords.STEPS,
                SimpleResponseGrammar.Keywords.INFORMAL_PROOF,
                SimpleResponseGrammar.Keywords.INFORMAL_THEOREM,
                SimpleResponseGrammar.Keywords.GOALS_TO_PROVE, # trim down the goals
                SimpleResponseGrammar.Keywords.INCORRECT_STEPS,
                SimpleResponseGrammar.Keywords.ERROR_MESSAGE,
                SimpleResponseGrammar.Keywords.LAST_STEP,
                SimpleResponseGrammar.Keywords.SUCCESS,
            ]
            assert coq_gpt_response.training_data_format is not None
            lines_map[SimpleResponseGrammar.Keywords.GOALS_TO_PROVE] = [SimpleResponseGrammar.Keywords.GOALS_TO_PROVE.value]
            if coq_gpt_response.training_data_format.goal_description is not None:
                new_line = f"{SimpleResponseGrammar.Keywords.DESCRIPTION}\n{coq_gpt_response.training_data_format.goal_description}\n"
                lines_map[SimpleResponseGrammar.Keywords.GOALS_TO_PROVE].append(new_line)
            for i, goal in enumerate(coq_gpt_response.training_data_format.start_goals):
                if len(goal.hypotheses) > 0:
                    for hyp in goal.hypotheses:
                        lines_map[SimpleResponseGrammar.Keywords.GOALS_TO_PROVE].append(hyp)
                new_line = SimpleResponseGrammar.Keywords.PROOF_SEPERATOR.value + " " + str(goal.goal)
                lines_map[SimpleResponseGrammar.Keywords.GOALS_TO_PROVE].append(new_line)
                if len(goal.relevant_defns) > 0 and (k is None or k > 0):
                    dfns = goal.relevant_defns
                    if k is not None:
                        dfns = dfns[:k]
                    dfns = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[dfn.lemma_idx]) for dfn in dfns]
                    if len(dfns) > 0:
                        dfns = ["\nSome relevant definitions are:"] + dfns
                    lines_map[SimpleResponseGrammar.Keywords.GOALS_TO_PROVE].extend(dfns)
                if len(goal.possible_useful_theorems_external) + len(goal.possible_useful_theorems_local) > 0 and (k is None or k > 0):
                    thms = goal.possible_useful_theorems_local + goal.possible_useful_theorems_external
                    if k is not None:
                        thms = thms[:k]
                    thms = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) for thm in thms]
                    if len(thms) > 0:
                        thms = ["\nSome relevant theorems are:"] + thms
                    lines_map[SimpleResponseGrammar.Keywords.GOALS_TO_PROVE].extend(thms)
            if len(coq_gpt_response.steps) > 0:
                new_line = f"\n{SimpleResponseGrammar.Keywords.STEPS}"
                lines_map[SimpleResponseGrammar.Keywords.STEPS] = [new_line]
                steps = [step.rstrip() for step in coq_gpt_response.steps]
                lines_map[SimpleResponseGrammar.Keywords.STEPS].extend(steps)
            if len(coq_gpt_response.incorrect_steps) > 0:
                new_line = f"\n{SimpleResponseGrammar.Keywords.INCORRECT_STEPS}"
                lines_map[SimpleResponseGrammar.Keywords.INCORRECT_STEPS] = [new_line]
                incorrect_steps = [f"{SimpleResponseGrammar.Keywords.WRONG_SYMBOL} {step.rstrip()}" for step in coq_gpt_response.incorrect_steps]
                lines_map[SimpleResponseGrammar.Keywords.INCORRECT_STEPS].extend(incorrect_steps)
            if coq_gpt_response.last_step is not None:
                new_line = f"\n{SimpleResponseGrammar.Keywords.LAST_STEP}"
                lines_map[SimpleResponseGrammar.Keywords.LAST_STEP] = [new_line]
                lines_map[SimpleResponseGrammar.Keywords.LAST_STEP].append(coq_gpt_response.last_step)
                if coq_gpt_response.success:
                    new_line = f"\n{SimpleResponseGrammar.Keywords.SUCCESS}"
                    lines_map[SimpleResponseGrammar.Keywords.SUCCESS] = [new_line]
            if coq_gpt_response.error_message is not None:
                # assert coq_gpt_response.last_step is not None
                # assert not coq_gpt_response.success
                new_line = f"\n{SimpleResponseGrammar.Keywords.ERROR_MESSAGE}"
                lines_map[SimpleResponseGrammar.Keywords.ERROR_MESSAGE] = [new_line]
                lines_map[SimpleResponseGrammar.Keywords.ERROR_MESSAGE].append(coq_gpt_response.error_message)
            if coq_gpt_response.informal_theorem is not None:
                new_line = f"\n{SimpleResponseGrammar.Keywords.INFORMAL_THEOREM}"
                lines_map[SimpleResponseGrammar.Keywords.INFORMAL_THEOREM] = [new_line]
                lines_map[SimpleResponseGrammar.Keywords.INFORMAL_THEOREM].append(coq_gpt_response.informal_theorem)
            if coq_gpt_response.informal_proof is not None:
                new_line = f"\n{SimpleResponseGrammar.Keywords.INFORMAL_PROOF}"
                lines_map[SimpleResponseGrammar.Keywords.INFORMAL_PROOF] = [new_line]
                lines_map[SimpleResponseGrammar.Keywords.INFORMAL_PROOF].append(coq_gpt_response.informal_proof)
            keywords = [keyword for keyword in lines_map.keys()]
            # Convert all the lines under each keyword to a single string
            for keyword in keywords:
                lines_map[keyword] = "\n".join(lines_map[keyword])
            # Frame the first prompt version without any token limit
            text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{SimpleResponseGrammar.Keywords.END}"
            
            # Trim the lines based on the max_token_cnt
            if char_cnt is not None and len(text) > char_cnt:
                _idx = 0
                diff = len(text) - char_cnt
                while _idx < len(priority_order_lo_hi) and diff > 0:
                    trim_part = priority_order_lo_hi[_idx]
                    if trim_part in lines_map:
                        if trim_part == SimpleResponseGrammar.Keywords.STEPS:
                            lines_map[trim_part] = lines_map[trim_part][-diff:]
                        else:
                            lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                    text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{SimpleResponseGrammar.Keywords.END}"
                    diff = len(text) - char_cnt
                    _idx += 1
        else:
            raise Exception(f"Invalid action {coq_gpt_response.action}")
        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None:
            assert len(text) <= char_cnt, f"Text length {len(text)} is greater than the max token count {char_cnt}. Possibly too few characters per token." +\
            f" characters_per_token = {characters_per_token}, max_token_cnt = {max_token_cnt}"
            # text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text
