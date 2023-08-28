Require Import Reals.

(* [TODO] [FIX] In environment
x : R
The term "x" has type "R" while it is expected to have type
 "nat". *)
Theorem mathd_algebra_113 : forall (x : R),
  x^(2%R)- (14%R) * x + (3%R) >= ((7%R)^(2%R)) - ((14%R) * (7 % R)) + (3%R).
Proof.
  (* Your proof goes here *)
Admitted.