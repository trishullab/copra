import copy
import logging
import typing
import re
from itp_interface.rl.simple_proof_env import ProofState, ProofAction, ProofEnvInfo, ProgressState
from itp_interface.lean_server.lean4_utils import Lean4Utils


class HandleHaveTactic:
    """
    A temporary hack to allow the use of Lean without needing to install it.
    This is a temporary solution until we can find a better way to use Lean.
    """
    have_regex1 = re.compile(r'^\s*have\s+([\S]*?)\s*:')
    have_regex2 = re.compile(r'^\s*have\s+([\S]*?)\s*:=\s*')
    simple_have_regex = re.compile(r'^\s*have\s+([\S|\s]*?)\s*:=\s*by')
    def __init__(self):
        self._last_actions = []
        self._last_proof_state = None
        self._last_have_tactic_idx = None
        self._have_tactics = []
        self._step_number = 0
        pass

    @property
    def nested_level(self) -> int:
        return len(self._have_tactics)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def scope_state(self, state: ProofState, action: ProofAction, next_state: ProofState, info: ProofEnvInfo, logger: logging.Logger, ignore: bool = False):
        """
        Returns the current scope state.
        """
        if state.language == ProofAction.Language.LEAN4:
            if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                if info.progress == ProgressState.STATE_CHANGED:
                    logger.info(f"Current goal count: {len(state.training_data_format.start_goals)}")
                    logger.info(f"Next goal count: {len(next_state.training_data_format.start_goals)}")
                    if self._have_tactic_introduced_goal(state, action, next_state) and \
                    not ignore:
                        self._have_tactics.append((action, self._step_number))
                        logger.info(f"1: Got a have tactic that introduced a goal, nested level is now {self.nested_level}")
                    else:
                        if len(self._have_tactics) > 0 and \
                        len(next_state.training_data_format.start_goals) < len(state.training_data_format.start_goals):
                            # Remove the last have tactic from the list
                            self._have_tactics.pop()
                            logger.info(f"2: Finished a have tactic that removed a goal, nested level is now {self.nested_level}")
                if info.progress != ProgressState.FAILED:
                    self._step_number += 1
            elif action.action_type == ProofAction.ActionType.BACKTRACK:
                # Backtrack to the last have tactic
                self._step_number -= 1
                if len(self._have_tactics) > 0:
                    _, seq_num = self._have_tactics[-1]
                    if seq_num == self._step_number:
                        # Remove the last have tactic from the list
                        self._have_tactics.pop()
                        logger.info(f"3: Backtracked to the last have tactic, nested level is now {self.nested_level}")
            else:
                self._step_number += 1
    def is_single_line_have_tactic(self, action: ProofAction) -> bool:
        """
        Returns True if the action is a `have` tactic.
        """
        if action.language == ProofAction.Language.LEAN4 and action.action_type == ProofAction.ActionType.RUN_TACTIC:
            tactic : str = action.kwargs.get('tactics', [None])[0]
            if tactic is not None:
                tactic = Lean4Utils.remove_comments(tactic)
                tactic = tactic.strip()
                # Check if the tactic is a `have` tactic ending with `by`
                if self.simple_have_regex.match(tactic) is not None:
                    return False
                else:
                    return True
        return False

    def fix_action(self, action: ProofAction, logger: logging.Logger) -> typing.Tuple[bool, typing.Optional[ProofAction]]:
        """
        Fixes the given action.
        """
        new_action = copy.deepcopy(action)
        if action.language == ProofAction.Language.LEAN4:
            # Indent the tactics based on the nested level
            if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                self._indent_tactics(new_action)
        return True, new_action
    
    def is_within_have_tactic(self) -> bool:
        """
        Returns True if the current action is within a `have` tactic.
        """
        return self.nested_level > 0
    
    def get_last_have_tactic(self) -> typing.Optional[str]:
        """
        Returns the last `have` tactic.
        """
        if len(self._have_tactics) > 0:
            action, _ = self._have_tactics[-1]
            tactic : str = action.kwargs.get('tactics', [None])[0]
            if tactic is not None:
                return Lean4Utils.remove_comments(tactic)
        return None

    def parse_have_tactic_action(self, action: ProofAction):
        """
        Fixes the given action if it is a `have` tactic.
        """
        original_action = copy.deepcopy(action)
        try:
            if action.language == ProofAction.Language.LEAN4 and action.action_type == ProofAction.ActionType.RUN_TACTIC:
                tactic : str = action.kwargs.get('tactics', [None])[0]
                if tactic is not None:
                    tactic = Lean4Utils.remove_comments(tactic)
                    have_no_def = self.have_regex2.match(tactic)
                    if have_no_def is not None:
                        have_name = str(have_no_def.group(1).strip())
                        have_action_type = 2
                    else:
                        have_colon = self.have_regex1.match(tactic)
                        if have_colon is not None:
                            have_name = str(have_colon.group(1).strip())
                            have_action_type = 1
                        else:
                            have_name = None
                    if have_name is None:
                        return action
                    idx = tactic.rfind(':=')
                    tactic_parts : typing.List[str] = []
                    if idx != -1:
                        # Remove everything after the last `:=`
                        tactic_parts.append(tactic[:idx + 2])
                        # Add the rest of the tactic
                        # After removing the `:=` and `by`, the rest of the tactic is the same
                        # as the original tactic
                        modified_tactic = tactic[idx + 2:]
                        modified_tactic = modified_tactic.strip()
                        if modified_tactic.startswith('by') and have_action_type != 2:
                            # Remove the `by` from the tactic
                            modified_tactic = modified_tactic[2:].strip()
                            tactic_parts[-1] = tactic_parts[-1].strip() + ' by'
                            if modified_tactic != '':
                                # Add the rest of the tactic
                                # Indent the tactic based on the nested level
                                tactic_parts.append(modified_tactic)
                        else:
                            modified_tactic = modified_tactic.strip()
                            if modified_tactic != '':
                                # Add the rest of the tactic
                                # Indent the tactic based on the nested level
                                tactic_parts[-1] = tactic_parts[-1].strip() + f" {modified_tactic}"
                    action.kwargs['tactics'] = tactic_parts
                    action.original_message["content"] = f"[RUN TACTIC]\n{tactic_parts[0]}\n[END]"
        except Exception as e:
            # If there is an error, return the original action
            action = original_action
        return action

    @staticmethod
    def fix_have_tactic(action: ProofAction) -> bool:
        if action.language == ProofAction.Language.LEAN4 and action.action_type == ProofAction.ActionType.RUN_TACTIC:
            tactic : str = action.kwargs.get('tactics', [None])[0]
            if tactic is not None:
                tactic = Lean4Utils.remove_comments(tactic)
                if tactic.strip().startswith('have'):
                    # We don't want to have a `have` tactic without seeing a goal
                    # Find the last occurence of `:=` in the action
                    idx = tactic.rfind(':=')
                    if idx != -1:
                        if tactic[idx + 2:].strip() == 'by':
                            return True
                        # Remove everything after the last `:=`
                        tactic = tactic[:idx + 2]
                        # Add a `by`
                        tactic = tactic[:idx + 2] + ' by'
                        action.kwargs['tactics'][0] = tactic
        return False

    @staticmethod
    def fix_have_tactic_str(tactic: str) -> str:
        """
        Fixes the given tactic string.
        """
        if tactic is not None:
            tactic = Lean4Utils.remove_comments(tactic)
            if tactic.strip().startswith('have'):
                # We don't want to have a `have` tactic without seeing a goal
                # Find the last occurence of `:=` in the action
                idx = tactic.rfind(':=')
                if idx != -1:
                    if tactic[idx + 2:].strip() == 'by':
                        return tactic
                    # Remove everything after the last `:=`
                    tactic = tactic[:idx + 2]
                    # Add a `by`
                    tactic = tactic[:idx + 2] + ' by'
        return tactic

    def _indent_tactics(self, action: ProofAction):
        """
        Indents the tactics based on the nested level.
        """
        if action.language == ProofAction.Language.LEAN4:
            if self.nested_level > 0 and action.action_type == ProofAction.ActionType.RUN_TACTIC:
                indentation = self.nested_level * 2
                action.kwargs['tactics'] = [
                    f"{' ' * indentation}{tactic.strip()}" 
                    for tactic in action.kwargs['tactics']
                ]

    def _have_tactic_introduced_goal(self, state: ProofState, action: ProofAction, next_state: ProofState) -> bool:
        """
        Checks if the tactic has introduced a new goal.
        """
        if state.language == ProofAction.Language.LEAN4:
            tactic : str = action.kwargs.get('tactics', [None])[0]
            if tactic is not None:
                # Remove comments from the tactic
                tactic = Lean4Utils.remove_comments(tactic)
                # Check if the tactic is a `have` tactic
                if not tactic.strip().startswith('have'):
                    return False
                # Check if the # goals have increased
                return True
                # if len(next_state.training_data_format.start_goals) > len(state.training_data_format.start_goals):
                #     return True
        return False