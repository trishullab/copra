Require Import Arith.
Require Import NPeano.
Require Import Nsatz.


(* [TODO] FIX lcm reference is not found *)
Theorem mathd_numbertheory_711 :
  forall (m n : nat),
    0 < m /\ 0 < n ->
    gcd m n = 8 ->
    lcm m n = 112 ->
    72 <= m + n.
Proof.
  intros m n h₀ h₁ h₂.
  apply Nat.le_trans with (m * n).
  - apply Nat.le_mul_diag_r. apply Nat.neq_0_lt_0. intros H. destruct h₀. rewrite H in h₁. now rewrite Nat.gcd_0_l in h₁.
  - rewrite <- h₂.
    rewrite <- (Nat.gcd_mul_mono_l _ _ 8) by lia.
    rewrite h₁.
    reflexivity.
Qed.