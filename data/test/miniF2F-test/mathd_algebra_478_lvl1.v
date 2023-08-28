(*
 * Author: Max Fan
 Level : 5
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
 
 Theorem mathd_algebra_478 :
   forall (b h v : R),
   (0%R < b /\ 0%R < h /\ 0%R < v) ->
   (v = 1 / 3 * (b * h)) ->
   (b = 30%R) ->
   (h = 13%R / 2%R) ->
   v = 65%R.
 Proof.
   intros.
   destruct H.
   destruct H3.
   rewrite H0.
   rewrite H1.
   rewrite H2.
   lra.
 Qed.