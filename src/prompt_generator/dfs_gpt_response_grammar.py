#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from enum import Enum
from src.prompt_generator.interpreter import Grammar
from src.tools.training_data_format import TrainingDataFormat
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
    incorrect_step_message: typing.Optional[str] = None
    training_data_format: typing.Optional[TrainingDataFormat] = None

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
  Goals Description String GoalResponses StepsResponses IncorrectStepsResponses End
| Goals GoalResponses StepsResponses IncorrectStepsResponses End;
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
    IncrctStps StepResponses ErrorMessage String
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
    
    def format_as_per_grammar(self, coq_gpt_response: CoqGptResponse) -> str:
        text = ""
        if coq_gpt_response.action == CoqGptResponseActions.ERROR:
            text = f"{CoqGPTResponseDfsGrammar.Keywords.ERROR}\n{coq_gpt_response.message}\n{CoqGPTResponseDfsGrammar.Keywords.END}"
        elif coq_gpt_response.action == CoqGptResponseActions.GOALS:
            assert coq_gpt_response.training_data_format is not None
            text = f"Goals to prove:\n{CoqGptResponseActions.GOALS}"
            if coq_gpt_response.training_data_format.goal_description is not None:
                text += f"{CoqGPTResponseDfsGrammar.Keywords.DESCRIPTION}\n{coq_gpt_response.training_data_format.goal_description}\n"
            lines = []
            for i, goal in enumerate(coq_gpt_response.training_data_format.start_goals):
                lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.GOAL} {i+1}")
                lines.append(str(goal.goal))
                if len(goal.hypotheses) > 0:
                    lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.HYPOTHESES} {i + 1}")
                    for hyp in goal.hypotheses:
                        lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.HYPOTHESIS} {hyp}")
                if len(goal.relevant_defns) > 0:
                    lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS} {i + 1}")
                    dfns = goal.relevant_defns
                    dfns = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[dfn.lemma_idx]) for dfn in dfns]
                    lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS} {i+1}")
                    lines.extend([f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITION} {dfn}" for dfn in dfns])
                if len(goal.possible_useful_theorems_external) + len(goal.possible_useful_theorems_local) > 0:
                    thms = goal.possible_useful_theorems_local + goal.possible_useful_theorems_external
                    thms = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) for thm in thms]
                    lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.THEOREMS} {i + 1}")
                    lines.extend([f"{CoqGPTResponseDfsGrammar.Keywords.THEOREM} {thm}" for thm in thms])
            gls_args = '\n'.join(lines)
            text += f"{gls_args}\n{CoqGPTResponseDfsGrammar.Keywords.END}"
        else:
            raise Exception(f"Invalid action {coq_gpt_response.action}")
        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        return text
    

if __name__ == "__main__":
    import os
    from src.prompt_generator.dfs_agent_grammar import DfsAgentGrammar
    os.chdir(root_dir)
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


