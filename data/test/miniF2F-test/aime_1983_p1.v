(*
 * Author: Talia Ringer, refined by Gaëtan Gilbert 
 *
 * We should probably move to SSReflect for the rest of these theorems & proofs,
 * and revisit this proof once we are on SSReflect
 *)
 Require Import Reals Lra Lia.
 Infix "/" := Rdiv.
 Infix "+" := Rplus.
 Infix "*" := Rmult.
 Infix "-" := Rminus.
 
 (*
  * Lift ln to nats
  *)
 Definition ln (n : nat) : R :=
   ln (INR n).
 
 (*
  * Lift ln lemmas to nats
  *)
 Lemma ln_mult :
   forall x y : nat,
     0 < x ->
     0 < y ->
     ln (x * y) = ln x + ln y.
 Proof.
   intros. unfold ln. rewrite mult_INR. apply ln_mult; auto using lt_0_INR.
 Qed.
 
 Ltac multiply_both_sides x H :=
   apply (f_equal (Rmult x)) in H;
   rewrite <- Rmult_assoc in H;
   try rewrite Rinv_r_simpl_m in H;
   auto.
 
 Ltac by_ln_mult_in H :=
   rewrite ln_mult in H; try apply Nat.mul_pos_pos; auto with arith.

(* [TODO] FIX ln_neq_0 was not found in the current environment *)
 Lemma lt_ln_neq :
   forall x,
     1 < x ->
     ln x <> 0%R.
 Proof.
   intros. apply ln_neq_0.
   - unfold not. intros. replace 1%R with (INR 1) in H0 by reflexivity.
     pose proof (INR_eq _ _ H0). subst. inversion H. inversion H2.
   - replace 0%R with (INR 0) by reflexivity. apply lt_INR. auto with arith.
 Qed.
 
 (*
  * The problem, a gross sequence of rewrites
  *)
 Theorem aime_1983_p1:
   forall (x y z w : nat),
     1 < x /\ 1 < y /\ 1 < z ->
     0 <= w ->
     ln w / ln x = 24%R ->
     ln w / ln y = 40%R ->
     ln w / ln (x * y * z) = 12%R ->
     ln w / ln z = 60%R.
 Proof.
   intros. unfold Rdiv in *.
   destruct H. destruct H4.
   pose proof (lt_ln_neq _ H). pose proof (lt_ln_neq _ H4). pose proof (lt_ln_neq _ H5).
   give_up.
   Admitted.
   (* assert (1 < x * y * z) by lia.
   pose proof (lt_ln_neq _ H9).
   repeat by_ln_mult_in H3.
   repeat by_ln_mult_in H10.
   multiply_both_sides (ln x) H1.
   multiply_both_sides (ln y) H2.
   multiply_both_sides (ln x + ln y + ln z) H3.
   assert (ln w = ln z * 60%R) by lra.
   multiply_both_sides (/ (ln z))%R H11.
   rewrite Rinv_l in H11 by assumption.
   lra.
 Qed. *)