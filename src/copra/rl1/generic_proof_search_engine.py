#!/usr/bin/env python3

import typing
import random
import logging
import time
import os
import copy
import math
from itp_interface.rl.proof_tree import ProofSearchResult, ProofTree
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor
from itp_interface.tools.training_data_format import TrainingDataFormat

class TacticGenerationEngineParams(object):
    def get_tactic_generation_engine(self, logger: logging.Logger = None):
        raise NotImplementedError("get_tactic_generation_engine must be implemented")
    pass

class GenericTacticGenerationEngine(object):
    def __init__(self):
        self.logger = None
        pass
    
    def get_next_tactic_step(self, partial_data: TrainingDataFormat, k: int = 1) -> typing.List[TrainingDataFormat]:
        raise NotImplementedError("get_next_tactic_step must be implemented")
    
    def set_logger(self, logger: logging.Logger):
        self.logger = logger

class ProofSearchEngine(object):
    def __init__(self, k: int, proof_depth: int, generic_tactic_engine: GenericTacticGenerationEngine, proof_executor_callback: typing.Callable[[], DynamicProofExecutor], timeout_in_secs: int = 60, disable_bactracking: bool = True, logger : logging.Logger = None, max_inferences_allowed: typing.Optional[int] = None):
        assert k > 0, "top_k must be greater than 0"
        assert proof_depth > 0, "proof_depth must be greater than 0"
        assert generic_tactic_engine is not None, "generic_tactic_engine must not be None"
        assert isinstance(generic_tactic_engine, GenericTacticGenerationEngine), "generic_tactic_engine must be of type GenericTacticEngine"
        assert proof_executor_callback is not None, "start_proof_state must not be None"
        assert proof_executor_callback is not None
        assert timeout_in_secs > 0, "timeout_in_secs must be greater than 0"
        self.timeout_in_secs = timeout_in_secs
        self.tactic_engine = generic_tactic_engine
        self.proof_executor_callback = proof_executor_callback
        self.k = k
        self.max_proof_depth = proof_depth
        self.backtracking_disabled = disable_bactracking
        self.file_name = None
        self.max_inferences_allowed = max_inferences_allowed if max_inferences_allowed is not None else math.inf
        self.logger = logger if logger is not None else logging.getLogger(__name__)
    
    def prove_next_lemma(self, proof_executor: DynamicProofExecutor, lemma_name) -> ProofSearchResult:
        current_depth = 0
        #2. Using stack instead of recursion to avoid stack overflow
        proof_stack : typing.List[typing.Tuple[int, TrainingDataFormat]] = []
        p_tree = ProofTree()
        proof_progressed: bool = True
        search_abandoned: bool = False
        proof_file_name = proof_executor.proof_file
        is_timeout = False
        is_inference_exhausted = False
        current_proof_state = proof_executor.get_current_proof_state_as_training_data()
        current_proof_state.proof_id = lemma_name
        time_budget = self.timeout_in_secs
        inference_budget = self.max_inferences_allowed
        self.logger.info(f"Starting proof search for lemma: {lemma_name}. Timeout set = {time_budget} secs.")
        possible_success_paths = 0
        possible_failure_paths = 0
        num_backtracks = 0
        inferences_used_so_far = 0
        was_qed = False
        start_time = time.time()
        # proof_executor.set_logger(self.logger)
        successful_running_proof_depth = 0
        while proof_executor.is_in_proof_mode() and not search_abandoned:
            # get the top k proof steps and add them to stack
            before_time = time.time()
            needs_qed = proof_executor.needs_qed()
            needs_cut_close = proof_executor.needs_cut_close()
            assert not (needs_qed and needs_cut_close), "needs_qed and needs_cut_close cannot be true at the same time"
            if proof_progressed and not needs_qed and not needs_cut_close and not is_timeout and not is_inference_exhausted:
                # get the top k proof steps and add them to stack
                self._get_top_k_proof_steps(current_proof_state, current_depth, lemma_name, proof_stack, proof_executor)
            elif proof_progressed and needs_qed and not is_timeout and not is_inference_exhausted:
                # get the qed tactic and add it to stack
                qed_tactic = TrainingDataFormat(proof_steps=["Qed."])
                proof_stack.append((current_depth, qed_tactic))
                was_qed = True
            elif proof_progressed and needs_cut_close and not is_timeout and not is_inference_exhausted:
                # get the cut close tactic and add it to stack
                cut_close_tactic = TrainingDataFormat(proof_steps=["}"])
                proof_stack.append((current_depth, cut_close_tactic))

            if (not is_timeout and not is_inference_exhausted) and len(proof_stack) > 0:
                # get the next proof step
                depth, next_tactic = proof_stack[-1]
                if depth == current_depth:
                    proof_stack.pop()
                    # update the proof state
                    tactic_line_num, ran_successfully = proof_executor.run_tactics(next_tactic.proof_steps)
                    if ran_successfully:
                        proof_progressed = (current_depth < self.max_proof_depth and not is_timeout) or was_qed
                        if proof_progressed:
                            previous_proof_state = copy.deepcopy(current_proof_state)
                            current_proof_state = proof_executor.get_current_proof_state_as_training_data()
                            # self.logger.info(f"Goal: {previous_proof_state.start_goals[:1]}")
                            # self.logger.info(f"Tacitc: {next_tactic.proof_steps}")
                            # self.logger.info(f"NextGoal: {current_proof_state.start_goals[:1]}")
                            current_proof_state.proof_id = lemma_name
                            previous_proof_state.proof_steps = copy.deepcopy(next_tactic.proof_steps)
                            # add the proof step to the proof tree
                            # Check if the current proof state is less harder than the previous proof state
                            if next_tactic.proof_steps[0] == "Qed.":
                                next_step_add = p_tree.try_add_tactic(tactic_line_num, previous_proof_state, force_add=True)
                            elif current_proof_state >= previous_proof_state:
                                # This is a cycle. Take a step back
                                next_step_add = False
                            else:
                                next_step_add = p_tree.try_add_tactic(tactic_line_num, previous_proof_state)
                            if not next_step_add:
                                proof_progressed = False
                                # self.logger.info(f"Got a cycle. Taking a step back.")
                            else:
                                current_depth += 1
                    else:
                        proof_progressed = False
                else:
                    proof_progressed = False
            else:
                # prune the proof stack
                proof_stack.clear() # This will cause the search to be abandoned
                proof_progressed = False

            search_abandoned = (len(proof_stack) == 0 and not proof_progressed) or is_timeout or is_inference_exhausted

            if not proof_progressed:
                possible_failure_paths += 1
                if possible_success_paths > 0:
                    possible_success_paths = 0
                if not search_abandoned:
                    assert depth <= current_depth, f"depth ({depth}) must be less than equal to current_depth ({current_depth})"
                    # assert len(proof_tree) == current_depth, "proof_tree must have the same length as current_depth"
                    assert len(p_tree) == current_depth, "proof_tree must have the same length as current_depth"
                    # cancel anything which might got executed
                    if proof_executor.cancel_tactic_till_line(tactic_line_num): #This becomes relevant when we have a cycle
                        num_backtracks += 1
                    next_depth, next_tactic = proof_stack[-1]
                    assert next_depth <= current_depth, "next_depth must be less than or equal to current_depth"
                    
                    # backtrack because the proof step failed
                    if not self.backtracking_disabled:
                        prev_line_num = None
                        while current_depth > next_depth:
                            # assert len(proof_tree) > 0, "proof_tree must have at least one element"
                            assert len(p_tree) > 0, "proof_tree must have at least one element"
                            # prev_line_num, _ = proof_tree.pop() # remove the last proof step
                            prev_line_num, _ = p_tree.try_remove_last_tactic()
                            assert prev_line_num is not None, "prev_line_num must not be None"
                            current_depth -= 1
                            assert current_depth >= 0, "current_depth must be greater than or equal to 0"
                            num_backtracks += 1
                        if prev_line_num is not None:
                            proof_executor.cancel_tactic_till_line(prev_line_num)
                        # # push the proof step back to the stack
                        # proof_stack.append((next_depth, next_tactic))
                        assert next_depth == current_depth, f"next_depth ({next_depth}) must be equal to current_depth ({current_depth})"
                    else:
                        # backtracking is disabled, so we abdon the search if it is beyond more than 1 depth
                        if current_depth > next_depth:
                            # search was aborted, no backtracking needed
                            # cancel anything which might got executed in the proof
                            # if len(proof_tree) > 0:
                            if len(p_tree) > 0:
                                # prev_line_num, _ = proof_tree[0]
                                prev_line_num, _ = p_tree[0]
                                proof_executor.cancel_tactic_till_line(prev_line_num)
                            proof_stack.clear()
                            search_abandoned = True
                        else:
                            # # push the proof step back to the stack
                            # proof_stack.append((next_depth, next_tactic))
                            pass
                        pass
                else:
                    # search was aborted, no backtracking needed
                    # cancel anything which might got executed in the proof
                    # if len(proof_tree) > 0:
                    if len(p_tree) > 0:
                        # prev_line_num, _ = proof_tree[0]
                        prev_line_num, _ = p_tree[0]
                        proof_executor.cancel_tactic_till_line(prev_line_num)
                        current_depth = 0
            else:
                successful_running_proof_depth = current_depth
                possible_success_paths = 1
            after_time = time.time()
            inferences_used_so_far += 1
            inference_budget -= 1
            time_budget -= (after_time - before_time)
            is_timeout = time_budget <= 0
            is_inference_exhausted = inference_budget <= 0
            if inferences_used_so_far % 100 == 0:
                self.logger.info(f"Ran {inferences_used_so_far} inferences so far." +
                f" Possible success paths: {possible_success_paths}," + 
                f" Possible failure paths: {possible_failure_paths}," +
                f" Explored paths: {possible_success_paths + possible_failure_paths}," +
                f" Number of Backtracks so far: {num_backtracks}," +
                f" Running Proof Depth: {successful_running_proof_depth}," +
                f" Inference Budget Remaining: {inference_budget}," + 
                f" Time Budget Remaining: {time_budget} secs.")
        time_taken = after_time - start_time
        assert proof_executor.is_in_proof_mode() or \
            (not proof_executor.is_in_proof_mode() and was_qed), \
                "proof_executor must be in proof mode or the proof must be QED to complete the proof search"
        proof_search_res = ProofSearchResult(proof_file_name, not proof_executor.is_in_proof_mode(), lemma_name, [tactic for _, tactic in p_tree], time_taken, inferences_used_so_far, possible_failure_paths, num_backtracks, is_timeout, is_inference_exhausted, longest_success_path=successful_running_proof_depth)
        return proof_search_res
    
    def _prove_all_attempt(self, all_proofs: list, proofs_attempted_so_far: set, retry_count: int = 0):
        with self.proof_executor_callback() as main_executor:
            if self.file_name is None:
                self.file_name = main_executor.proof_file
            with self.proof_executor_callback() as proof_imitation_executor:
                self.logger.info(f"{f'[Retry {retry_count + 1}]' if retry_count > 0 else ''}Proving all lemmas in {main_executor.proof_file}")
                while not main_executor.execution_complete:
                    self.logger.info(f"\n[MAIN]: {main_executor.current_stmt}\n[IMITATE]: {proof_imitation_executor.current_stmt}")
                    assert not main_executor.is_in_proof_mode(), "main_executor must not be in proof mode"
                    assert not proof_imitation_executor.is_in_proof_mode(), "proof_imitation_executor must not be in proof mode"
                    all_statements_to_be_executed = list(main_executor.run_till_next_lemma_return_exec_stmt())
                    if main_executor.execution_complete:
                        break
                    proof_imitation_executor.run_cmds(all_statements_to_be_executed, raise_exception=True)
                    exec_statements = "\n".join(all_statements_to_be_executed)
                    assert proof_imitation_executor.is_in_proof_mode() == main_executor.is_in_proof_mode(), "proof_imitation_executor must be in the same mode as main_executor\n" + \
                        f"Executed steps: \n{exec_statements}\n[MAIN]: {main_executor.current_stmt}\n[IMITATE]: {proof_imitation_executor.current_stmt}"
                    lemma_name = main_executor.get_lemma_name_if_running()
                    if lemma_name is None or lemma_name in proofs_attempted_so_far:
                        all_statements_to_be_executed = list(main_executor.run_to_finish_lemma_return_exec())
                        if main_executor.execution_complete:
                            break
                        proof_imitation_executor.run_cmds(all_statements_to_be_executed, raise_exception=True)
                        if lemma_name is not None:
                            self.logger.info(f"Lemma \"{lemma_name}\" already attempted. Skipping it.")
                    else:
                        proofs_attempted_so_far.add(lemma_name)
                        lemma_name = main_executor.coq.cur_lemma
                        res = self.prove_next_lemma(proof_imitation_executor, lemma_name)
                        all_proofs.append(res)
                        if res.proof_found:
                            self.logger.info(f"Lemma \"{lemma_name}\" proved. It took {res.proof_time_in_secs} seconds.")
                            self.logger.info(f"\n{res}")
                            main_executor.run_to_finish_lemma()
                        else:
                            self.logger.info(f"Lemma \"{lemma_name}\" not proved." + 
                            f" It {f'timed-out in {res.proof_time_in_secs} seconds.' if res.is_timeout else 'abandoned by depth-limit.'}")
                            self.logger.warning(f"\n{res}")
                            self.logger.warning(f"[INCOMPLETE...]")
                            _, ran_admitted = proof_imitation_executor.run_cmds(["Admitted."]) # admit the lemma
                            if not ran_admitted:
                                self.logger.warning(f"Couldn't run \"Admitted.\" command. Trying to run \"Abort.\" command.")
                                _, ran_abort = proof_imitation_executor.run_cmds(["Abort."]) # abort the lemma
                                self.logger.warning(f"Couldn't run \"Abort.\" command.")
                                if not ran_abort:
                                    all_statements_to_be_executed = list(main_executor.run_to_finish_lemma_return_exec())
                                    if main_executor.execution_complete:
                                        break
                                    proof_imitation_executor.run_cmds(all_statements_to_be_executed, raise_exception=True)
                                else:
                                    main_executor.run_to_finish_lemma()
                            else:
                                main_executor.run_to_finish_lemma()



    def prove_all(self) -> typing.List[ProofSearchResult]:
        all_proofs : typing.List[ProofSearchResult] = []
        lemma_so_far = set()
        all_proofs_attempted = False
        retry_count = 0
        retries_exhausted = False
        while not all_proofs_attempted and not retries_exhausted:
            try:
                self._prove_all_attempt(all_proofs, lemma_so_far, retry_count)
                all_proofs_attempted = True
            except Exception:
                self.logger.exception("Exception occurred while proving all lemmas. Retrying...")
                all_proofs_attempted = False
                retry_count += 1
                retries_exhausted = retry_count >= 100
        if retries_exhausted:
            self.logger.error(f"Retries exhausted. Aborting proof search for file {self.file_name if self.file_name is not None else '<unknown>'}")
        return all_proofs

    def _get_top_k_proof_steps(self, current_proof_state: TrainingDataFormat, current_depth: int, lemma_name: str, proof_stack: typing.List[TrainingDataFormat], proof_executor: DynamicProofExecutor):
        # get the next tactic step
        next_tactic_step = self.tactic_engine.get_next_tactic_step(current_proof_state, self.k)
        tactics = [(current_depth, tactic_step) for tactic_step in next_tactic_step]
        proof_stack.extend(tactics)
        return len(tactics) > 0


class Prover(object):
    def __init__(self, 
            name: str, 
            k: int,
            proof_depth: int, 
            proof_timeouts_in_secs: int, 
            tactic_engine: GenericTacticGenerationEngine,
            context_type: DynamicProofExecutor.ContextType = DynamicProofExecutor.ContextType.NoContext,
            disable_backtracking: bool = True, 
            logger : logging.Logger = None,
            max_inferences_allowed: typing.Optional[int] = None):
        assert isinstance(k, int)
        assert k > 0
        assert isinstance(proof_depth, int)
        assert proof_depth > 0
        assert proof_timeouts_in_secs > 0
        assert isinstance(tactic_engine, GenericTacticGenerationEngine)
        assert isinstance(name, str)
        self.context_type = context_type
        self.k = k
        self.proof_depth = proof_depth
        self.proof_timeouts_in_secs = proof_timeouts_in_secs
        self.max_inferences_allowed = max_inferences_allowed
        self.tactic_engine = tactic_engine
        self.name = name
        self.logger = logger if logger is not None else logging.getLogger(self.name)
        self.disable_backtracking = disable_backtracking

    def try_proving_theorems_in_file(self, file_path: str, project_folder: str = None, k : int = None, proof_depth: int = None, proof_timeouts_in_secs: int = None, max_inferences_allowed: typing.Optional[int] = None) -> typing.List[ProofSearchResult]:
        assert isinstance(file_path, str)
        assert os.path.exists(file_path)
        k = k if k is not None else self.k
        max_inferences_allowed = max_inferences_allowed if max_inferences_allowed is not None else self.max_inferences_allowed
        proof_depth = proof_depth if proof_depth is not None else self.proof_depth
        proof_timeouts_in_secs = proof_timeouts_in_secs if proof_timeouts_in_secs is not None else self.proof_timeouts_in_secs
        proof_exec_callback = ProofExecutorCallback(project_folder, file_path, context_type=self.context_type)
        proof_search_engine = ProofSearchEngine(k=self.k, proof_depth=proof_depth, generic_tactic_engine=self.tactic_engine, proof_executor_callback=proof_exec_callback.get_proof_executor, timeout_in_secs=proof_timeouts_in_secs, disable_bactracking=self.disable_backtracking, logger=self.logger, max_inferences_allowed=max_inferences_allowed)
        all_proofs = proof_search_engine.prove_all()
        return all_proofs

if __name__ == "__main__":
    # Write a mock tactic engine
    import os
    from itp_interface.tools.coq_parse_utils import CoqLineByLineReader
    from itp_interface.tools.coq_executor import CoqExecutor
    class MockTacticEngine(GenericTacticGenerationEngine):
        def __init__(self, file_name: str):
            assert file_name is not None, "file_name must not be None"
            assert isinstance(file_name, str), "file_name must be of type str"
            assert os.path.exists(file_name), "file_name must exist"
            self.file_name = file_name
            self.coq_exec = CoqExecutor(main_file=file_name)
            self.first_run = True
            self.mock_coq_exec: CoqExecutor = None
            super().__init__()
        
        def __enter__(self):
            self.coq_exec.__enter__()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.coq_exec.__exit__(exc_type, exc_val, exc_tb)
        
        def set_mock_coq_exec(self, coq_exec: CoqExecutor):
            assert coq_exec is not None, "coq_exec must not be None"
            assert isinstance(coq_exec, CoqExecutor), "coq_exec must be of type CoqExecutor"
            self.mock_coq_exec = coq_exec

        def get_next_tactic_step(self, partial_data: TrainingDataFormat, top_k: int = 1) -> typing.List[TrainingDataFormat]:
            assert top_k > 0, "top_k must be greater than 0"
            assert partial_data is not None, "partial_data must not be None"
            assert isinstance(partial_data, TrainingDataFormat), "partial_data must be of type TrainingDataFormat"
            assert self.mock_coq_exec is not None, "self.mock_coq_exec must not be None"
            lemma_name = partial_data.proof_id
            dice_roll = random.randint(0, 3)
            if dice_roll == 0:
                extension = [TrainingDataFormat(proof_id=lemma_name,proof_steps=["simpl.", "auto."])]
            elif dice_roll == 1:
                extension =  [TrainingDataFormat(proof_id=lemma_name,proof_steps=["auto.", "reflexivity."])]
            elif dice_roll == 2:
                extension =  [TrainingDataFormat(proof_id=lemma_name,proof_steps=["intros."])]
            elif dice_roll == 3:
                extension =  [TrainingDataFormat(proof_id=lemma_name,proof_steps=["randomos."])]
            else:
                raise Exception("dice_roll must be in [0, 3]")
            self.coq_exec.run_till_next_lemma()
            if self.coq_exec.line_num != self.mock_coq_exec.line_num or self.coq_exec.coq.proof_context.focused_goal != self.mock_coq_exec.coq.proof_context.focused_goal:
                return extension * top_k
            else:
                self.coq_exec.run_next()
                next_tactic = self.coq_exec.current_stmt
                return [TrainingDataFormat(proof_id=lemma_name,proof_steps=[next_tactic])] + extension*(top_k-1)
    def _get_proof_executor():
        return DynamicProofExecutor(file_path="data/custom_group_theory/grp_thm.v")
    with MockTacticEngine("data/custom_group_theory/grp_thm.v") as mock_tactic_engine:
        proof_search_engine = ProofSearchEngine(k=2, proof_depth=6, generic_tactic_engine=mock_tactic_engine, proof_executor_callback=_get_proof_executor)
        mock_tactic_engine.set_mock_coq_exec(proof_search_engine.proof_executor_callback)
        all_proofs = proof_search_engine.prove_all()
        for proof in all_proofs:
            print(f"Proof found: {proof.proof_found}")
            print(f"Proof of: {proof.lemma_name}")
            if len(proof.proof_steps) > 0:
                for proof_step in proof.proof_steps:
                    print(f"Proof step: {proof_step.proof_steps}")
            print("=============================================")