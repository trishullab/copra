import Mathlib
import StackMachine.Basic
open BigOperators

namespace StackMachine

theorem compile_num_correct (map : Var → Nat) (n : Nat) :
run map (compile (Expr.num n)) = ProgResult.ok (eval map (Expr.num n)) :=
by
    simp [compile]
    simp [run, run_program, run_instr]
    simp [eval]

theorem compile_var_correct (map : Var → Nat) (v : Var) :
run map (compile (Expr.var v)) = ProgResult.ok (eval map (Expr.var v)) :=
by
    simp [compile]
    simp [run]
    simp [run_program]
    simp [run_instr]
    simp [eval]

theorem compile_num_add_correct (map : Var → Nat) (n1 n2 : Nat) :
run map (compile (Expr.add (Expr.num n1) (Expr.num n2))) = ProgResult.ok (eval map (Expr.add (Expr.num n1) (Expr.num n2))) :=
by
    simp [compile, run, run_program, run_instr]
    simp [eval]
    simp [Nat.add_comm]

theorem compile_var_add_correct (map : Var → Nat) (v1 v2 : Var) :
run map (compile (Expr.add (Expr.var v1) (Expr.var v2))) = ProgResult.ok (eval map (Expr.add (Expr.var v1) (Expr.var v2))) :=
by
    simp [compile, run, run_program, run_instr]
    simp [eval]
    linarith

theorem compile_var_num_add_correct (map : Var → Nat) (v : Var) (n : Nat) :
run map (compile (Expr.add (Expr.var v) (Expr.num n))) = ProgResult.ok (eval map (Expr.add (Expr.var v) (Expr.num n))) :=
by
    simp [compile]
    simp [run, run_program, run_instr]
    simp [eval]
    ring

theorem compile_num_var_add_correct (map : Var → Nat) (v : Var) (n : Nat) :
run map (compile (Expr.add (Expr.num n) (Expr.var v))) = ProgResult.ok (eval map (Expr.add (Expr.num n) (Expr.var v))) :=
by
    simp [compile, run, run_program, run_instr, eval]
    ring

theorem compile_linear_comb_correct (map : Var → Nat) (v1 v2 : Var) (n1 n2 : Nat) :
run map (compile
    (Expr.add
      (Expr.add (Expr.var v1) (Expr.num n1))
      (Expr.add (Expr.var v2) (Expr.num n2)))) =
ProgResult.ok
(eval map
    (Expr.add (Expr.add (Expr.var v1) (Expr.num n1))
    (Expr.add (Expr.var v2) (Expr.num n2)))) :=
by
    simp [compile]
    simp [run, run_program, run_instr]
    simp [eval]
    generalize h1 : n2 + map v2 = x
    generalize h2 : n1 + map v1 = y
    simp [Nat.add_comm (map v1) n1]
    simp [Nat.add_comm (map v2) n2]
    rw [h2, h1]
    linarith

theorem compile_linear_comb_correct' (map : Var → Nat) (v1 v2 : Var) (n1 n2 : Nat) :
run map (compile
    (Expr.add
      (Expr.add (Expr.num n1) (Expr.var v1))
      (Expr.add (Expr.num n2) (Expr.var v2)))) =
ProgResult.ok
(eval map
    (Expr.add (Expr.add (Expr.num n1) (Expr.var v1))
    (Expr.add (Expr.num n2) (Expr.var v2)))) :=
by
    simp [compile]
    simp [run, run_program, run_instr]
    simp [eval]
    linarith

theorem run_append_progs_correct (map : Var → Nat) (p1 p2 : List Instr) (stack : List ProgResult) :
run_program map stack (p1 ++ p2) = run_program map (run_program map stack p1) p2 :=
by
    induction p1 generalizing stack
    case nil =>
        simp [run_program]
    case cons i is ih =>
        simp [run_program]
        simp [run_program] at ih
        rw [ih]

theorem compile_append_progs_stack_correct (map : Var → Nat) (e : Expr) (stack : List ProgResult) :
run_program map stack (compile e) = [ProgResult.ok (eval map e)] ++ stack :=
by
   induction e generalizing stack
   rename_i n
   simp [compile]
   simp [run, run_program, run_instr]
   simp [eval]
   rename_i v
   simp [compile]
   simp [run, run_program, run_instr]
   simp [eval]
   rename_i e1 e2 ih1 ih2
   simp [compile]
   rw [← List.append_assoc]
   generalize h1 : compile e1 ++ compile e2 = p
   simp [eval]
   rw [run_append_progs_correct]
   rw [← h1]
   rw [run_append_progs_correct]
   rw [ih1]
   rw [ih2]
   simp
   simp [run_program, run_instr]
   ring

theorem compile_correct (map : Var → Nat) (e : Expr) :
run map (compile e) = ProgResult.ok (eval map e) :=
by
    simp [run]
    simp [compile_append_progs_stack_correct]

theorem nonempty_compile (map : Var → Nat) (p: Program) (n: Nat) :
run map p = ProgResult.ok n → run_program map [] p = [ProgResult.ok n] :=
by
    intro h
    simp [run] at h
    generalize o1: run_program map [] p = o
    rw [o1] at h
    match o with
    | [] => contradiction
    | [ProgResult.ok n] => simp at h; rw [h]
    | ProgResult.err :: _ => contradiction
    | ProgResult.ok a :: _ :: _ => contradiction
    done

theorem compile_correct' (map : Var → Nat) (e : Expr) :
run map (compile e) = ProgResult.ok (eval map e) :=
by
    simp [compile_correct]

end StackMachine
