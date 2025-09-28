import Mathlib
import StackMachine.Basic
import StackMachine.Train
open BigOperators

namespace StackMachine

theorem compile_equiv (map : Var → Nat) (e1 e2 : Expr) :
eval map e1 = eval map e2 → run map (compile e1) = run map (compile e2) :=
by
    intro h
    rw [compile_correct]
    rw [compile_correct]
    rw [h]

theorem compile_add_assoc (map : Var → Nat) (e1 e2 e3 : Expr) :
run map (compile (Expr.add (Expr.add e1 e2) e3)) = run map (compile (Expr.add e1 (Expr.add e2 e3))) :=
by
    simp [compile_correct]
    simp [eval]
    ring

theorem compile_add_comm (map : Var → Nat) (e1 e2 : Expr) :
run map (compile (Expr.add e1 e2)) = run map (compile (Expr.add e2 e1)) :=
by
    simp [compile_correct]
    simp [eval]
    linarith

theorem run_multiple_programs (map : Var → Nat) (e1 e2 : Expr) :
run map (compile e1 ++ compile e2 ++ [Instr.add]) =
ProgResult.ok (eval map (Expr.add e1 e2)) :=
by
    simp [run]
    simp [run_append_progs_correct]
    simp [run_program, run_instr]
    simp [compile_append_progs_stack_correct]
    simp [eval]
    linarith

theorem if_result_ok_then_eval_ok (map : Var → Nat) (e : Expr) (n : Nat) :
run map (compile e) = ProgResult.ok n → eval map e = n :=
by
    intro h
    rw [compile_correct] at h
    cases h
    rfl

theorem if_eval_ok_then_result_ok (map : Var → Nat) (e : Expr) (n : Nat) :
eval map e = n → run map (compile e) = ProgResult.ok n :=
by
    intro h
    rw [compile_correct]
    rw [h]

theorem if_result_expr_never_err (map : Var → Nat) (e : Expr) :
run map (compile e) ≠ ProgResult.err :=
by
    rw [compile_correct]
    intro h
    cases h

theorem if_eval_zero_map_zero_then_expr_not_var (map : Var → Nat) (e : Expr) :
eval map e = 0 → ∀ v, map v != 0 → ¬ (∀ v', e = Expr.var v') :=
by
    intro he hmap v hmapv
    simp at v
    induction e generalizing v map
    rename_i num
    simp [eval] at he
    rw [he] at hmapv
    cases hmapv hmap
    rename_i var
    simp [eval] at he
    cases hmapv hmap
    contradiction
    rename_i e1 e2 _ _
    simp [eval] at he
    cases hmapv hmap

theorem eval_composition_flattens (map : Var → Nat) (e e1: Expr) :
eval map (Expr.add (Expr.num (eval map e)) e1) = eval map (Expr.add e1 e) :=
by
   simp [eval]
   linarith

theorem nonempty_instruction_run (map : Var → Nat) (i : Instr) (n : Nat) :
¬ (run_instr map [ProgResult.ok n] i = []) :=
by
    cases i
    intro h
    simp [run_instr] at h
    intro h
    simp [run_instr] at h
    intro h
    simp [run_instr] at h
    done

end StackMachine
