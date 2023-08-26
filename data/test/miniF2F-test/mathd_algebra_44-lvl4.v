(*
 * Author: Max Fan
    Level : 4
 *)

 Require Import Reals Lra Lia.
 Infix "/" := Rdiv.
 Infix "+" := Rplus.
 Infix "*" := Rmult.
 Infix "-" := Rminus.
 
 Theorem mathd_algebra_44 :
   forall (s t : R),
   (s = 9 - 2 * t) ->
   (t = 3 * s + 1) ->
   (s = 1%R /\ t = 4%R).
 Proof.
   intros.
   split; lra.
 Qed.