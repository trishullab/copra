Require Import Arith.

Theorem plus_zero: forall (n: nat), n + 0 = n.
Proof.
    intros n.
    induction n.
    - simpl.
      reflexivity.
    - simpl. 
    rewrite IHn. 
    reflexivity.
Qed.

Theorem mul_comm: forall (a b: nat), a * b = b * a.
Proof.
    intros a b.
    induction a.
    - simpl. rewrite <- mult_n_O. reflexivity.
    - symmetry. simpl. rewrite <- mult_n_Sm. rewrite IHa. rewrite plus_comm.
      reflexivity.
Qed.

Theorem mul_plus_dist: forall (a b c: nat), a * (b + c) = a * b + a * c.
Proof.
    intros a b c.
    induction a.
    - simpl. reflexivity.
    - symmetry. simpl. rewrite IHa.
      rewrite plus_assoc. rewrite <- (plus_assoc b (a *b) c).
      rewrite (plus_comm (a*b) c). 
      rewrite plus_assoc. rewrite plus_assoc.
      reflexivity.
Qed.

Theorem a_plus_b_square: forall (a b: nat), (a + b)*(a + b) = a*a + b*b  + 2*a*b.
Proof.
    intros a b.
    rewrite mul_plus_dist.
    rewrite mul_comm.
    rewrite mul_plus_dist.
    rewrite (mul_comm (a + b) b).
    rewrite mul_plus_dist.
    rewrite (plus_comm (b*a) (b*b)).
    rewrite plus_assoc.
    rewrite <- (plus_assoc (a*a) (a*b) (b*b)).
    rewrite (plus_comm (a*b) (b*b)).
    rewrite plus_assoc.
    simpl.
    rewrite (mul_comm (a + (a + 0)) b).
    rewrite mul_plus_dist. rewrite mul_plus_dist.
    rewrite <- (mult_n_O b).
    rewrite plus_zero.
    rewrite plus_assoc.
    rewrite (mul_comm b a).
    reflexivity.
Qed.
