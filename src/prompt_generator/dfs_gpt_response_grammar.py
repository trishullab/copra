#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from enum import Enum
from src.prompt_generator.interpreter import Grammar
from src.tools.training_data_format import Goal, TrainingDataFormat
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
    
    def format_as_per_grammar(self, coq_gpt_response: CoqGptResponse, k: typing.Optional[int] = None) -> str:
        text = ""
        if coq_gpt_response.action == CoqGptResponseActions.ERROR:
            text = f"{CoqGPTResponseDfsGrammar.Keywords.ERROR}\n{coq_gpt_response.message}\n{CoqGPTResponseDfsGrammar.Keywords.END}"
        elif coq_gpt_response.action == CoqGptResponseActions.GOALS:
            assert coq_gpt_response.training_data_format is not None
            text = f"Goals to prove:\n{CoqGPTResponseDfsGrammar.Keywords.GOALS}"
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
                if len(goal.relevant_defns) > 0 and (k is None or k > 0):
                    dfns = goal.relevant_defns
                    if k is not None:
                        dfns = dfns[:k]
                    dfns = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[dfn.lemma_idx]) for dfn in dfns]
                    lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS} {i+1}")
                    lines.extend([f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITION} {dfn}" for dfn in dfns])
                if len(goal.possible_useful_theorems_external) + len(goal.possible_useful_theorems_local) > 0 and (k is None or k > 0):
                    thms = goal.possible_useful_theorems_local + goal.possible_useful_theorems_external
                    if k is not None:
                        thms = thms[:k]
                    thms = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) for thm in thms]
                    lines.append(f"{CoqGPTResponseDfsGrammar.Keywords.THEOREMS} {i + 1}")
                    lines.extend([f"{CoqGPTResponseDfsGrammar.Keywords.THEOREM} {thm}" for thm in thms])
            if len(coq_gpt_response.steps) > 0:
                lines.append(f"\n{CoqGPTResponseDfsGrammar.Keywords.STEPS}")
                lines.extend([f"{CoqGPTResponseDfsGrammar.Keywords.STEP} {step}" for step in coq_gpt_response.steps])
            if len(coq_gpt_response.incorrect_steps) > 0:
                lines.append(f"\n{CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS}")
                lines.extend([f"{CoqGPTResponseDfsGrammar.Keywords.STEP} {step}" for step in coq_gpt_response.incorrect_steps])
            if coq_gpt_response.last_step is not None:
                lines.append(f"\n{CoqGPTResponseDfsGrammar.Keywords.LAST_STEP}")
                lines.append(coq_gpt_response.last_step)
                if coq_gpt_response.success:
                    lines.append(f"\n{CoqGPTResponseDfsGrammar.Keywords.SUCCESS}")
            if coq_gpt_response.error_message is not None:
                assert coq_gpt_response.last_step is not None
                assert not coq_gpt_response.success
                lines.append(f"\n{CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE}")
                lines.append(coq_gpt_response.error_message)
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


