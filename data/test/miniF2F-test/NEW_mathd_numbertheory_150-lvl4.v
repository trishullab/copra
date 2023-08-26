Require Import Arith.
Require Import NPeano.

Theorem mathd_numbertheory_150 :
  forall (n : nat),
    ~ Nat.prime (7 + 30 * n) ->
    6 <= n.
Proof.
  intros n h₀.
  apply Nat.nle_succ_r.
  intros h.
  apply h₀.
  apply Nat.prime_divisors in h.
  destruct h as [h | [h | h]]; lia.
Qed.