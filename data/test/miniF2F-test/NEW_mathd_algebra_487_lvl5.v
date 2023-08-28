Require Import Reals.
Open Scope R_scope.

(* [TODO] [FIX] Syntax Error: Lexer: Undefined token =/= *)
Theorem mathd_algebra_487 :
  forall (a b c d : R),
    b = a^2 ->
    a + b = 1 ->
    d = c^2 ->
    c + d = 1 ->
    a ≠ c ->
    sqrt ((a - c)^2 + (b - d)^2) = sqrt 10.
Proof.
  (* Your proof goes here *)
Admitted.