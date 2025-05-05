from itp_interface.rl.simple_proof_env import ProofAction, ProofState, ProofEnvInfo
from itp_interface.rl.abstraction import Policy 
import typing

class HumanPolicy(Policy):
    def __init__(self,
        language: ProofAction.Language = ProofAction.Language.LEAN4):
        self._policy = self
        self._actions_remaining = []
        super().__init__()

    def __call__(self, state: ProofState) -> ProofAction:
        """
        Returns the next action to take.
        """
        # This is a placeholder implementation. Replace with actual logic.
        current_state = self.pretty_print_proof_state(state)
        # Clear the screen
        print("\033[H\033[J")
        print("Current state:")
        print(current_state)
        print("What do you want to do?")
        print("1. Run tactic")
        print("2. Backtrack")
        print("3. Stop")
        choice = input("Enter your choice: ")
        if choice not in ["1", "2", "3"]:
            print("Invalid choice. Defaulting to running a tactic.")
            # Default action if invalid choice
            choice = "1"
        if choice == "1":
            action_type = ProofAction.ActionType.RUN_TACTIC
        elif choice == "2":
            action_type = ProofAction.ActionType.BACKTRACK
        else:
            action_type = ProofAction.ActionType.EXIT
        if action_type == ProofAction.ActionType.RUN_TACTIC:
            # Scan tactic which can potentially have new lines
            # Only stop scanning when two new lines are entered
            print("Enter the tactic to run (end with two new lines):")
            tactic = self.scan_tactic()
            print(f"Will run tactic: {tactic}")
            action = ProofAction(
                action_type=action_type,
                language=state.language,
                tactics= [tactic]
            )
            action.original_message = {
                "role": "assistant",
                "content": f"[RUN TACTIC]\n{tactic}\n[END]",
            }
        else:
            action = ProofAction(
                action_type=action_type,
                language=state.language
            )
        return action


    def add_delayed(self, action: ProofAction):
        """
        Adds a delayed action to the policy.
        """
        self._actions_remaining.append(action)
        pass

    def reset_last_action(self, action: ProofAction):
        """
        Resets the last action taken by the policy.
        """
        pass

    def update(self, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo):
        pass


    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {
            "queries": 0
        }

    def pretty_print_proof_state(self, state: ProofState) -> str:
        """
        Pretty prints the proof state.
        """
        start_goals = state.training_data_format.start_goals
        lines = []
        for goal in start_goals:
            for hypothesis in goal.hypotheses:
                lines.append(hypothesis)
            lines.append(f"⊢ {goal.goal}")
        return "\n".join(lines)

    def scan_tactic(self) -> str:
        """
        Scans the tactics from input which can potentially have new lines.
        """
        if len(self._actions_remaining) > 0:
            # If there are any actions remaining, we should not scan for a new tactic
            return self._actions_remaining.pop(0).kwargs['tactics'][0]
        tactic = ""
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        tactic = "\n".join(lines)
        return tactic
    
    def checkpoint(self):
        pass

    def clone(self):
        pass