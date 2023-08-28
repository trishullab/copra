(*
 * Author: Max Fan
 *)

 Require Import Reals Lra Lia.
 Infix "/" := Rdiv.
 Infix "+" := Rplus.
 Infix "*" := Rmult.
 Infix "-" := Rminus.
 Infix "<" := Rlt.
 Infix ">" := Rgt.
 Infix "<" := Rle.
 Infix ">=" := Rge.
 
 Theorem mathd_algebra_141:
   forall (a b : R),
   ((a * b) = 180%R) ->
   (2 * (a + b) = 54%R) ->
   (a^2 + b^2) = 369%R.
 Proof.
   intros.
   field_simplify_eq in H0.
   assert (a + b = 27%R) by lra.
   replace (a^2 + b^2) with ((a + b)^2 - 2*a*b) by lra.
   rewrite H1.
   lra.
 Qed.