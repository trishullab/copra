import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import enum
import math
from src.tools.training_data_format import TrainingDataFormat, Goal

class MaskConstants:
    BertStyle = '<mask>'
    T5Style = [f'<extra_id_{i}>' for i in range(100)]

class DataFormatLayoutTypes(enum.Enum):
    """The layout of the training data"""
    Start_MProof__GPTStyle = 1
    Start_MProof__T5Style = 2
    Start_Local_MProof__GPTStyle = 3
    Start_Local_MProof__T5Style = 4

class SepToken(object):
    def __init__(self, token: str):
        assert isinstance(token, str)
        self.token = token

class SepTokenConstants:
    Goals = SepToken('[GOALS]')
    End = SepToken('[END_GOAL]')
    Step = SepToken('[STEP]')

class Stringify:
    special_context_starting_chars = ["+", "-", "*"]
    @staticmethod
    def stringify_training_data(training_data_subpart_name: str, val: typing.Any, data_point: TrainingDataFormat, layout_type: DataFormatLayoutTypes) -> str:
        assert isinstance(training_data_subpart_name, str)
        assert training_data_subpart_name in LayoutDetails.possible_members
        if training_data_subpart_name == 'proof_steps':
            assert isinstance(val, list)
            assert all([isinstance(step, str) for step in val])
            is_step_opening_context = [False]*len(val)
            for idx, step in enumerate(val):
                is_opening_context = False
                for char in Stringify.special_context_starting_chars:
                    found = True
                    for ch in step:
                        if ch != char:
                            found = False
                            break
                    if found:
                        is_opening_context = True
                        break
                is_step_opening_context[idx] = is_opening_context
            for idx, step in enumerate(val):
                if is_step_opening_context[idx]:
                    val[idx] = '{' # Replace all opening context with a single '{'
            return "\n".join(val) # Write each proof step on a new line.
        elif training_data_subpart_name == 'start_goals' or training_data_subpart_name == 'end_goals':
            assert isinstance(val, list)
            assert all([isinstance(goal, Goal) for goal in val])
            def _format_goal_hyps(_idx: int, g: Goal, lb: str, is_last: bool) -> str:
                layout_is_local = layout_type in [DataFormatLayoutTypes.Start_Local_MProof__GPTStyle, DataFormatLayoutTypes.Start_Local_MProof__T5Style]
                hyps = "{lb}[HYPOTHESIS] ".join(g.hypotheses)
                hyps_str = ""
                defns_str = ""
                thms_str = ""
                if len(hyps) > 0:
                    hyps_str = f"""{lb}[HYPOTHESES] {_idx + 1}{lb}[HYPOTHESIS] {hyps}"""
                defns = f"{lb}[DEFINITION] ".join([str(data_point.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in g.relevant_defns])
                if len(defns) > 0:
                    defns_str = f"""{lb}[DEFINITIONS] {_idx + 1}{lb}[DEFINITION] {defns}"""
                if (layout_is_local and is_last) or not layout_is_local:
                    include_idx = not (layout_is_local and is_last)
                    thms = "\n[THEOREM] ".join([str(data_point.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in (g.possible_useful_theorems_local + g.possible_useful_theorems_external)])
                    if len(thms) > 0:
                        thms_str = f"""{lb}[THEOREMS] {str(_idx + 1) if include_idx else ''}{lb}[THEOREM] {thms}"""
                return f"""[GOAL] {_idx + 1}{lb}{g.goal if g.goal is not None else ""}{hyps_str}{defns_str}{thms_str}"""
            if training_data_subpart_name == 'start_goals':
                line_break = "\n"
                val_len = len(val)
                formatted_goals = f"{line_break}".join([_format_goal_hyps(_idx, g, line_break, _idx == val_len - 1) for _idx, g in enumerate(val)])
                return formatted_goals
            else:
                # Don't include the hypotheses in the end goals.
                # Because these hypotheses are not part of the proof.
                goals = "\n".join([(goal.goal if goal.goal is not None else "") for goal in val])
                return f"Goals\n{goals}"
        else:
            return str(val)
    
    @staticmethod
    def unstringify_trainining_data(training_data_subpart_name: str, val: str) -> typing.Any:
        assert isinstance(training_data_subpart_name, str)
        assert training_data_subpart_name in LayoutDetails.possible_members
        if training_data_subpart_name == 'proof_steps':
            return val.split('\n')
        elif training_data_subpart_name == 'start_goals' or training_data_subpart_name == 'end_goals':
            try:
                split = val.split('[HYPOTHESES]')
                if len(split) == 1:
                    # There are no hypotheses.
                    goals = split[0].strip().rstrip('-') # Remove the newline and the line break -.
                    hyps = []
                else:
                    goals, hyps = split[0], split[1:]
                    goals = goals.split('[GOAL]')[1:].strip() # Remove the newline and the line break -.
                    hyps = [hyp.strip() for hyp in hyps.split('\n') if hyp.strip() != '']
                goals = [goal.strip() for goal in goals.split('\n') if goal.strip() != '']
                return [Goal(hyps, goal) for goal in goals]
            except:
                return []
        else:
            return val

class LayoutDetails(object):
    # Possible members should be one of the following.
    possible_members = [attr for attr in dir(TrainingDataFormat()) if not callable(getattr(TrainingDataFormat(), attr)) and not attr.startswith("__")]
    def __init__(self, data_parts: typing.List[str], layout_format: DataFormatLayoutTypes, sep_tokens: typing.List[SepToken] = [], masking_token: str = None):
        assert isinstance(data_parts, list)
        assert all([isinstance(part, str) for part in data_parts])
        assert all([part in LayoutDetails.possible_members for part in data_parts]), f"Invalid data part(s) {data_parts}. Valid data parts are {LayoutDetails.possible_members}."
        assert isinstance(sep_tokens, list)
        assert all([isinstance(token, SepToken) for token in sep_tokens])
        assert len(sep_tokens) == len(data_parts) - 1
        self.data_parts : typing.List[str] = data_parts
        self.sep_tokens : typing.List[SepToken] = sep_tokens
        self.is_masked : bool = masking_token is not None
        self.masking_token : str = masking_token
        self.layout_format = layout_format

    def get_formated_data_for_layout_details(self, data_point: TrainingDataFormat, end_with_new_line: bool = True) -> str:
        formatted_data = ""
        new_line = "\n" if end_with_new_line else ""
        for data_part, sep_token in zip(self.data_parts[:-1], self.sep_tokens):
            val = getattr(data_point, data_part)
            assert val is not None
            formatted_data += Stringify.stringify_training_data(data_part, val, data_point, self.layout_format) + new_line
            formatted_data += sep_token.token + new_line
        data_part = self.data_parts[-1]
        val = getattr(data_point, data_part)
        assert val is not None
        formatted_data += Stringify.stringify_training_data(data_part, val, data_point, self.layout_format) + new_line
        return formatted_data

    def parse_formatted_data_from_layout(self, formatted_data: str, data_point: TrainingDataFormat, skip_if_not_found: bool = False) -> None:
        sep_data = []
        data_idx = 0
        for data_part, sep_token in zip(self.data_parts[:-1], self.sep_tokens):
            # scan the data_part till you find the sep_token
            next_data_idx = formatted_data.find(sep_token.token, data_idx)
            if next_data_idx == -1 and skip_if_not_found:
                    return
            assert next_data_idx != -1, f"Could not find {sep_token.token} in {formatted_data}"
            # add the data_part to sep_data
            sep_data.append(formatted_data[data_idx:next_data_idx])
            # update data_idx
            data_idx = next_data_idx + len(sep_token.token)
        data_part = self.data_parts[-1]
        sep_data.append(formatted_data[data_idx:])
        assert len(sep_data) == len(self.data_parts)
        for data_part, line in zip(self.data_parts, sep_data):
            val = Stringify.unstringify_trainining_data(data_part, line)
            setattr(data_point, data_part, val)

class DataLayout(object):
    layout_map = {
        DataFormatLayoutTypes.Start_MProof__T5Style: [
            SepTokenConstants.Goals,
            LayoutDetails(['start_goals'], layout_format=DataFormatLayoutTypes.Start_MProof__T5Style),
            SepTokenConstants.Step, 
            LayoutDetails(['proof_steps'], layout_format=DataFormatLayoutTypes.Start_MProof__T5Style, masking_token=MaskConstants.T5Style[0])
        ],
        DataFormatLayoutTypes.Start_MProof__GPTStyle: [
            SepTokenConstants.Goals,
            LayoutDetails(['start_goals'], layout_format=DataFormatLayoutTypes.Start_MProof__GPTStyle),
            SepTokenConstants.Step, 
            LayoutDetails(['proof_steps'], layout_format=DataFormatLayoutTypes.Start_MProof__GPTStyle, masking_token="")
        ],
        DataFormatLayoutTypes.Start_Local_MProof__T5Style: [
            SepTokenConstants.Goals,
            LayoutDetails(['start_goals'], layout_format=DataFormatLayoutTypes.Start_Local_MProof__T5Style),
            SepTokenConstants.Step,
            LayoutDetails(['proof_steps'], layout_format=DataFormatLayoutTypes.Start_Local_MProof__T5Style, masking_token=MaskConstants.T5Style[0])
        ],
        DataFormatLayoutTypes.Start_Local_MProof__GPTStyle: [
            SepTokenConstants.Goals,
            LayoutDetails(['start_goals'], layout_format=DataFormatLayoutTypes.Start_Local_MProof__GPTStyle),
            SepTokenConstants.Step,
            LayoutDetails(['proof_steps'], layout_format=DataFormatLayoutTypes.Start_Local_MProof__GPTStyle, masking_token="")
        ]
    }

    def __init__(self, data_format_layout: DataFormatLayoutTypes, with_label: bool = False):
        assert isinstance(data_format_layout, DataFormatLayoutTypes)
        assert isinstance(with_label, bool)
        self.data_format_layout = data_format_layout
        self.layout_details_list: list = DataLayout.layout_map[self.data_format_layout]
        self.with_label = with_label

    def get_layout_formatter(self) -> typing.Callable[[TrainingDataFormat, typing.List[typing.Tuple[int, bool]]], typing.Union[typing.Tuple[str, str], str]]:
        """Get the layout formatter for the given data format layout.

        Returns:
            typing.Callable[[TrainingDataFormat], str]: The layout formatter.
        """
        label_is_auto_regressive = str(self.data_format_layout).endswith("__GPTStyle")
        def generic_layout_formatter(data_point: TrainingDataFormat, max_subpart_lens = []) -> typing.Union[typing.Tuple[str, str], str]:
            assert isinstance(data_point, TrainingDataFormat)
            formatted_data = ""
            label_data = ""
            sub_part_idx = 0
            for idx, layout_details in enumerate(self.layout_details_list):    
                should_add_new_line = idx != len(self.layout_details_list) - 1
                new_line = "\n" if should_add_new_line else ""
                if isinstance(layout_details, SepToken):
                    formatted_data += layout_details.token + new_line
                else:
                    max_part_len, should_trim_the_end = max_subpart_lens[sub_part_idx] if len(max_subpart_lens) > sub_part_idx else (math.inf, True)
                    sub_part_idx += 1
                    assert isinstance(layout_details, LayoutDetails), f"Invalid layout details {layout_details}"
                    if layout_details.is_masked:
                        mask = layout_details.masking_token + new_line
                        if self.with_label:
                            new_formatted_data = layout_details.get_formated_data_for_layout_details(data_point, should_add_new_line)
                            if len(new_formatted_data) > max_part_len:
                                new_formatted_data = new_formatted_data[:max_part_len] if should_trim_the_end else new_formatted_data[-max_part_len:]
                                if should_add_new_line and new_formatted_data[-1] != "\n":
                                    new_formatted_data += "\n"
                            if label_is_auto_regressive:
                                label_data += new_formatted_data
                            else:
                                label_data += mask + "\n" + new_formatted_data
                                formatted_data += mask
                        else:
                            formatted_data += mask
                    else:
                        new_formatted_data = layout_details.get_formated_data_for_layout_details(data_point, should_add_new_line)
                        if len(new_formatted_data) > max_part_len:
                            new_formatted_data = new_formatted_data[:max_part_len] if should_trim_the_end else new_formatted_data[-max_part_len:]
                            if should_add_new_line and new_formatted_data[-1] != "\n":
                                new_formatted_data += "\n"
                        formatted_data += new_formatted_data
            if self.with_label:
                return formatted_data, label_data
            return formatted_data
        return generic_layout_formatter

    def _populate_data_point(self, trainining_data: TrainingDataFormat, data: str, label_is_auto_regressive: bool, skip_parts_not_found: bool = False):
        assert isinstance(trainining_data, TrainingDataFormat)
        assert isinstance(data, str)
        assert isinstance(skip_parts_not_found, bool)
        data_idx = 0
        prev_layout_details: typing.Optional[LayoutDetails] = None
        capture_data = False
        for layout_details in self.layout_details_list:
            if isinstance(layout_details, SepToken):
                # find the next occurence of the sep token:
                next_data_idx = data.find(layout_details.token, data_idx)
                if skip_parts_not_found and next_data_idx == -1:
                        continue
                assert next_data_idx != -1, f"Invalid data format. Expected {layout_details.token} after index {data_idx}"
                if capture_data:
                    assert prev_layout_details is not None, "Invalid layout details"
                    assert isinstance(prev_layout_details, LayoutDetails), f"Invalid layout details {prev_layout_details}"
                    prev_layout_details.parse_formatted_data_from_layout(data[data_idx:next_data_idx].strip(), trainining_data, skip_parts_not_found)
                    capture_data = False
                    prev_layout_details = None
                data_idx = next_data_idx + len(layout_details.token)
            else:
                assert isinstance(layout_details, LayoutDetails), f"Invalid layout details {layout_details}"
                if layout_details.is_masked:
                    if not label_is_auto_regressive:
                        # Skip the masked token
                        next_data_idx = data.find(layout_details.masking_token, data_idx)
                        assert next_data_idx != -1, f"Invalid data format. Expected {layout_details.masking_token} after index {data_idx}"
                        if capture_data:
                            assert prev_layout_details is not None, "Invalid layout details"
                            assert isinstance(prev_layout_details, LayoutDetails), f"Invalid layout details {prev_layout_details}"
                            prev_layout_details.parse_formatted_data_from_layout(data[data_idx:next_data_idx].strip(), trainining_data, skip_parts_not_found)
                            capture_data = False
                            prev_layout_details = None
                        data_idx = next_data_idx + len(layout_details.masking_token)
                        capture_data = True
                        prev_layout_details = layout_details
                    else:
                        # Set for capturing the data
                        capture_data = True
                        prev_layout_details = layout_details
        if capture_data and data_idx < len(data):
                assert prev_layout_details is not None, "Invalid layout details"
                assert isinstance(prev_layout_details, LayoutDetails), f"Invalid layout details {prev_layout_details}"
                prev_layout_details.parse_formatted_data_from_layout(data[data_idx:].strip(), trainining_data, skip_parts_not_found)

    def get_format_parser(self) -> typing.Callable[[str, str], TrainingDataFormat]:
        """Get the format parser for the given data format layout.

        Returns:
            typing.Callable[[str, str], TrainingDataFormat]: The format parser.
        """
        label_is_auto_regressive = str(self.data_format_layout).endswith("__GPTStyle")
        def generic_format_parser(formatted_data: str, labelled_data: str = None) -> TrainingDataFormat:
            trainining_data: TrainingDataFormat = TrainingDataFormat()
            trainining_data.proof_id = "<ParsedFromGeneration>"
            self._populate_data_point(trainining_data, formatted_data, label_is_auto_regressive, skip_parts_not_found=True)
            if labelled_data is not None:
                if label_is_auto_regressive:
                    self._populate_data_point(trainining_data, labelled_data, label_is_auto_regressive, skip_parts_not_found=True)
                else:
                    self._populate_data_point(trainining_data, labelled_data, label_is_auto_regressive, skip_parts_not_found=True)
            # data_idx = 0
            # prev_layout_details: typing.Optional[LayoutDetails] = None
            # capture_data = False
            # for layout_details in self.layout_details_list:
            #     if isinstance(layout_details, SepToken):
            #         # find the next occurence of the sep token:
            #         next_data_idx = data.find(layout_details.token, data_idx)
            #         if skip_parts_not_found and next_data_idx == -1:
            #                 continue
            #         assert next_data_idx != -1, f"Invalid data format. Expected {layout_details.token} after index {data_idx}"
            #         if capture_data:
            #             assert prev_layout_details is not None, "Invalid layout details"
            #             assert isinstance(prev_layout_details, LayoutDetails), f"Invalid layout details {prev_layout_details}"
            #             prev_layout_details.parse_formatted_data_from_layout(data[data_idx:next_data_idx].strip(), trainining_data)
            #             capture_data = False
            #             prev_layout_details = None
            #         data_idx = next_data_idx + len(layout_details.token)
            #     else:
            #         assert isinstance(layout_details, LayoutDetails), f"Invalid layout details {layout_details}"
            #         if layout_details.is_masked:
            #             if not label_is_auto_regressive:
            #                 # Skip the masked token
            #                 next_data_idx = data.find(layout_details.masking_token, data_idx)
            #                 assert next_data_idx != -1, f"Invalid data format. Expected {layout_details.masking_token} after index {data_idx}"
            #                 if capture_data:
            #                     assert prev_layout_details is not None, "Invalid layout details"
            #                     assert isinstance(prev_layout_details, LayoutDetails), f"Invalid layout details {prev_layout_details}"
            #                     prev_layout_details.parse_formatted_data_from_layout(data[data_idx:next_data_idx].strip(), trainining_data)
            #                     capture_data = False
            #                     prev_layout_details = None
            #                 data_idx = next_data_idx + len(layout_details.masking_token)
            #                 capture_data = True
            #                 prev_layout_details = layout_details
            #             else:
            #                 # Set for capturing the data
            #                 capture_data = True
            #                 prev_layout_details = layout_details
            # if capture_data:
            #         assert prev_layout_details is not None, "Invalid layout details"
            #         assert isinstance(prev_layout_details, LayoutDetails), f"Invalid layout details {prev_layout_details}"
            #         prev_layout_details.parse_formatted_data_from_layout(data[data_idx:].strip(), trainining_data)
            return trainining_data
        return generic_format_parser


if __name__ == "__main__":
    sample_training_data = TrainingDataFormat(
        proof_id="sample_proof_id1",
        start_goals=[Goal(
        goal="forall (bb : bbmap) (succs : list positive) (l : L.t) \n  (st : state) (n : positive),\nlet st' := propagate_successors bb succs l st in\n(In n succs ->\n bb n = false -> In n (worklist st') /\\ (aval st') !! n = l) /\\\n(~ In n succs \\/ bb n = true -> (aval st') !! n = (aval st) !! n)", 
        hypotheses=[
        "P : L.t -> Prop",
        "code : PTree.t A",
        "Ptop : P L.top",
        "Ptransf : forall (pc : positive) (instr : A) (x : L.t)"])],
        end_goals=[Goal(
        goal = "(False -> bb n = false -> In n (worklist st) /\\ (aval st) !! n = l) /\\\n(~ False \\/ bb n = true -> (aval st) !! n = (aval st) !! n)\n(a = n \\/ In n succs ->\n bb n = false ->\n In n\n   (worklist\n      (if bb a\n       then propagate_successors bb succs l st\n       else\n        propagate_successors bb succs l\n          {|\n          aval := PMap.set a l (aval st);\n          worklist := a :: worklist st |})) /\\\n (aval\n    (if bb a\n     then propagate_successors bb succs l st\n     else\n      propagate_successors bb succs l\n        {|\n        aval := PMap.set a l (aval st);\n        worklist := a :: worklist st |})) !! n = l) /\\\n(~ (a = n \\/ In n succs) \\/ bb n = true ->\n (aval\n    (if bb a\n     then propagate_successors bb succs l st\n     else\n      propagate_successors bb succs l\n        {|\n        aval := PMap.set a l (aval st);\n        worklist := a :: worklist st |})) !! n = (aval st) !! n)",
        hypotheses=[])],
        proof_steps=[
        "induction succs; simpl; intros."
        ]
    )
    mlm_layout = DataLayout(DataFormatLayoutTypes.Start_MProof__T5Style, with_label=True)
    mlm_formatter = mlm_layout.get_layout_formatter()
    formatted_data, label_data = mlm_formatter(sample_training_data, max_subpart_lens=[(300, True), (400, True), (400, True)])
    format_parser = mlm_layout.get_format_parser()
    parsed_data = format_parser(formatted_data, label_data)
    print(formatted_data)
    print("*"*50)
    print(label_data)
    print("="*50)
    print(parsed_data)
    print("[DONE]")
    print("="*50)
    mlm_layout = DataLayout(DataFormatLayoutTypes.Start_MProof__GPTStyle, with_label=True)
    mlm_formatter = mlm_layout.get_layout_formatter()
    formatted_data, label_data = mlm_formatter(sample_training_data, max_subpart_lens=[(300, False), (400, True), (400, True)])
    format_parser = mlm_layout.get_format_parser()
    parsed_data = format_parser(formatted_data, label_data)
    print(formatted_data)
    print("*"*50)
    print(label_data)
    print("="*50)
    print(parsed_data)
    print("[DONE]")
    print("="*50)