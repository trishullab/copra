#!/usr/bin/env python3

import typing
from enum import Enum
from copra.prompt_generator.interpreter import Grammar
from itp_interface.tools.training_data_format import Goal, TrainingDataFormat
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class CoqGptResponseActions(object):
    GOALS = "[GOALS]"
    ERROR = "[ERROR]"
    ERROR_MESSAGE = "[ERROR MESSAGE]"

@dataclass_json
@dataclass
class CoqGptResponse(object):
    action : str = CoqGptResponseActions.GOALS
    success: bool = True
    message: str = ""
    steps: typing.List[str] = field(default_factory=list)
    incorrect_steps: typing.List[str] = field(default_factory=list)
    error_message: typing.Optional[str] = None
    last_step: typing.Optional[str] = None
    training_data_format: typing.Optional[TrainingDataFormat] = None
    informal_proof: typing.Optional[str] = None
    informal_theorem: typing.Optional[str] = None

class CoqGPTResponseDfsGrammar(Grammar):
    grammar = """
Prog:
  GoalsResponse
| ErrorResponse
| String Prog
| Prog String Prog;
ErrorResponse:
   Error String End
|  Error ErrorString End;
GoalsResponse:
  Goals Description String GoalResponses StepsResponses IncorrectStepsResponses LastResponse End
| Goals GoalResponses StepsResponses IncorrectStepsResponses LastResponse End;
GoalResponses:
  GoalResponse
| GoalResponse GoalResponses
| EMPTY;
GoalResponse:
 Goal int String HypsResponses DfnsResponses ThmsResponses;
DfnsResponses:
    Dfns int DfnResponses
|   EMPTY;
DfnResponses:
    DfnResponse
|   DfnResponse DfnResponses;
DfnResponse:
    Dfn String;
ThmsResponses:
    Thms int ThmResponses
|   EMPTY;
ThmResponses:
    ThmResponse
|   ThmResponse ThmResponses;
ThmResponse:
    Thm String;
HypsResponses:
    Hyps int HypResponses
|   EMPTY;
HypResponses:
  HypResponse
| HypResponse HypResponses;
HypResponse:
  Hyp String;
IncorrectStepsResponses:
    IncrctStps StepResponses
|   EMPTY;
LastResponse:
    LastStep String ErrorMessage String
|   LastStep String Success
|   EMPTY;
StepsResponses:
    Stps StepResponses
|   EMPTY;
StepResponses:
    StepResponse
|   StepResponse StepResponses;
StepResponse:
    Stp String;


terminals
int: /\d+/;
Goals: "[GOALS]";
Goal: "[GOAL]";
Hyps: "[HYPOTHESES]";
Hyp: "[HYPOTHESIS]";
Stps: "[STEPS]";
Stp: "[STEP]";
IncrctStps: "[INCORRECT STEPS]";
Dfns: "[DEFINITIONS]";
Dfn: "[DEFINITION]";
Thms: "[THEOREMS]";
Thm: "[THEOREM]";
Error: "[ERROR]";
ErrorMessage: "[ERROR MESSAGE]";
Success: "[SUCCESS]";
LastStep: "[LAST STEP]";
End: "[END]";
Description: "[DESCRIPTION]";
String:;
ErrorString:;
"""
    class Keywords(Enum):
        GOALS = "[GOALS]"
        GOAL = "[GOAL]"
        HYPOTHESES = "[HYPOTHESES]"
        HYPOTHESIS = "[HYPOTHESIS]"
        STEPS = "[STEPS]"
        STEP = "[STEP]"
        INCORRECT_STEPS = "[INCORRECT STEPS]"
        DEFINITIONS = "[DEFINITIONS]"
        DEFINITION = "[DEFINITION]"
        THEOREMS = "[THEOREMS]"
        THEOREM = "[THEOREM]"
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

    keywords = [keyword.value for keyword in Keywords]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in CoqGPTResponseDfsGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1
    
    def error_string(text, pos):
        last = pos
        while last < len(text):
            while last < len(text) and text[last] != '[':
                last += 1
            if last < len(text) and text[last:].startswith("[END]") and text[last:].endswith("[END]"):
                return text[pos:last]
            last += 1

    def __init__(self):
        recognizers = {
            'String': CoqGPTResponseDfsGrammar.before_keyword,
            'ErrorString': CoqGPTResponseDfsGrammar.error_string
        }
        super(CoqGPTResponseDfsGrammar, self).__init__(CoqGPTResponseDfsGrammar.grammar, CoqGPTResponseDfsGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: CoqGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        if coq_gpt_response.action == CoqGptResponseActions.ERROR:
            text = f"{CoqGPTResponseDfsGrammar.Keywords.ERROR}\n{coq_gpt_response.message}\n{CoqGPTResponseDfsGrammar.Keywords.END}"
        elif coq_gpt_response.action == CoqGptResponseActions.GOALS:
            lines_map = {
                CoqGPTResponseDfsGrammar.Keywords.GOALS : [],
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM : [],
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF : [],
                CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS : [],
                CoqGPTResponseDfsGrammar.Keywords.THEOREMS : [],
                CoqGPTResponseDfsGrammar.Keywords.STEPS : [],
                CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS : [],
                CoqGPTResponseDfsGrammar.Keywords.LAST_STEP : [],
                CoqGPTResponseDfsGrammar.Keywords.SUCCESS : [],
                CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE : []
            }
            lines_order = [
                CoqGPTResponseDfsGrammar.Keywords.GOALS,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF,
                CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS,
                CoqGPTResponseDfsGrammar.Keywords.THEOREMS,
                CoqGPTResponseDfsGrammar.Keywords.STEPS,
                CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS,
                CoqGPTResponseDfsGrammar.Keywords.LAST_STEP,
                CoqGPTResponseDfsGrammar.Keywords.SUCCESS,
                CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE
            ]
            priority_order_lo_hi = [
                CoqGPTResponseDfsGrammar.Keywords.THEOREMS,
                CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS,
                CoqGPTResponseDfsGrammar.Keywords.STEPS,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM,
                CoqGPTResponseDfsGrammar.Keywords.GOALS, # trim down the goals
                CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS,
                CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE,
                CoqGPTResponseDfsGrammar.Keywords.LAST_STEP,
                CoqGPTResponseDfsGrammar.Keywords.SUCCESS,
            ]
            assert coq_gpt_response.training_data_format is not None
            new_line = f"Goals to prove:\n{CoqGPTResponseDfsGrammar.Keywords.GOALS}"
            lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS] = [new_line]
            if coq_gpt_response.training_data_format.goal_description is not None:
                new_line = f"{CoqGPTResponseDfsGrammar.Keywords.DESCRIPTION}\n{coq_gpt_response.training_data_format.goal_description}\n"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
            for i, goal in enumerate(coq_gpt_response.training_data_format.start_goals[:1]):
                new_line = f"{CoqGPTResponseDfsGrammar.Keywords.GOAL} {i+1}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                new_line = str(goal.goal)
                lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                if len(goal.hypotheses) > 0:
                    new_line = f"{CoqGPTResponseDfsGrammar.Keywords.HYPOTHESES} {i + 1}"
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                    for hyp in goal.hypotheses:
                        new_line = f"{CoqGPTResponseDfsGrammar.Keywords.HYPOTHESIS} {hyp}"
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                if len(goal.relevant_defns) > 0 and (k is None or k > 0):
                    dfns = goal.relevant_defns
                    if k is not None:
                        dfns = dfns[:k]
                    dfns = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[dfn.lemma_idx]) for dfn in dfns]
                    new_line = f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS} {i + 1}"
                    if len(lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS]) == 0:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS] = [new_line]
                    else:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS].append(new_line)
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITION} {dfn}" for dfn in dfns])
                if len(goal.possible_useful_theorems_external) + len(goal.possible_useful_theorems_local) > 0 and (k is None or k > 0):
                    thms = goal.possible_useful_theorems_local + goal.possible_useful_theorems_external
                    if k is not None:
                        thms = thms[:k]
                    thms = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) for thm in thms]
                    new_line = f"{CoqGPTResponseDfsGrammar.Keywords.THEOREMS} {i + 1}"
                    if len(lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS]) == 0:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS] = [new_line]
                    else:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS].append(new_line)
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.THEOREM} {thm}" for thm in thms])
            if len(coq_gpt_response.steps) > 0:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.STEPS}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.STEPS] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.STEPS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.STEP} {step}" for step in coq_gpt_response.steps])
            if len(coq_gpt_response.incorrect_steps) > 0:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.STEP} {step}" for step in coq_gpt_response.incorrect_steps])
            if coq_gpt_response.last_step is not None:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.LAST_STEP}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.LAST_STEP] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.LAST_STEP].append(coq_gpt_response.last_step)
                if coq_gpt_response.success:
                    new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.SUCCESS}"
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.SUCCESS] = [new_line]
            if coq_gpt_response.error_message is not None:
                # assert coq_gpt_response.last_step is not None
                # assert not coq_gpt_response.success
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE].append(coq_gpt_response.error_message)
            if coq_gpt_response.informal_theorem is not None:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM].append(coq_gpt_response.informal_theorem)
            if coq_gpt_response.informal_proof is not None:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF].append(coq_gpt_response.informal_proof)
            keywords = [keyword for keyword in lines_map.keys()]
            # Convert all the lines under each keyword to a single string
            for keyword in keywords:
                lines_map[keyword] = "\n".join(lines_map[keyword])
            # Frame the first prompt version without any token limit
            text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{CoqGPTResponseDfsGrammar.Keywords.END}"
            
            # Trim the lines based on the max_token_cnt
            if char_cnt is not None and len(text) > char_cnt:
                _idx = 0
                diff = len(text) - char_cnt
                while _idx < len(priority_order_lo_hi) and diff > 0:
                    trim_part = priority_order_lo_hi[_idx]
                    if trim_part in lines_map:
                        if trim_part == CoqGPTResponseDfsGrammar.Keywords.STEPS:
                            lines_map[trim_part] = lines_map[trim_part][-diff:]
                        else:
                            lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                    text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{CoqGPTResponseDfsGrammar.Keywords.END}"
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
    

if __name__ == "__main__":
    import os
    from copra.prompt_generator.dfs_agent_grammar import DfsAgentGrammar
    agent_grammar = DfsAgentGrammar("example_user", "example_assistant")
    conv_file = "data/prompts/conversation/coq-proof-agent-example-long-conv-dfs.md"
    assert os.path.exists(conv_file), f"File {conv_file} does not exist"
    result = agent_grammar.get_openai_conv_messages(conv_file, "system")
    user_messages = [m["content"] for m in result if m["name"] == "example_user"]
    grammar = CoqGPTResponseDfsGrammar()
    for user_message in user_messages:
        compile_result = grammar.compile(user_message)
        print(compile_result)
    # Some other error examples
    training_data_format = TrainingDataFormat(
        goal_description="There are unfocused goals.",
        all_useful_defns_theorems=[],
        start_goals=[
            Goal(
                goal="forall n : nat, 0 + n = n",
            )
        ]
    )
    coq_gpt_response = CoqGptResponse(
        CoqGptResponseActions.GOALS,
        success=False,
        steps=["intros.", "- reflexivity."],
        incorrect_steps=["rewrite <- plus_O_n."],
        last_step="rewrite <- plus_n_O.",
        error_message="Unable to unify the goal with the theorem.",
        training_data_format=training_data_format)
    text = grammar.format_as_per_grammar(coq_gpt_response)
    print("="*50)
    print(text)
    print("="*50)
    compile_result = grammar.compile(text)
    print(compile_result)
    # TODO BUG the error response don't compile
#     result = grammar.compile("""
# [ERROR]
# Unable to parse the expression.
# [END]
# """)
#     print(result)
#     result = grammar.compile("""
# [ERROR]
# Expected [STEPS], but got something else.
# [END]
# """)
#     print(result)


