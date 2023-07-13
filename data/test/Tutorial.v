(* set of natural numbers *)
Inductive N : Set :=
| Z : N       (* the zero *)
| NXT : N->N. (* the next number of any natural number *)

(********************************************************************)

Fixpoint add(m n: N): N :=
match m with
| Z => n
| NXT m1 => NXT(add m1 n)
end.

Lemma add_zero: forall n: N, n = add n Z.
Proof.
induction n.
simpl.
reflexivity.
simpl.
f_equal.
assumption.
Qed.

Lemma add_nxt: forall m n: N, NXT (add m n) = add m (NXT n).
Proof.
intros m n.
induction m.
simpl.
reflexivity.
simpl.
f_equal.
assumption.
Qed.

(* addition is commutative *)
Lemma add_comm: forall m n: N, (add m n) = (add n m).
Proof.
intros m n.
induction m.
simpl.
apply add_zero.
simpl.
rewrite IHm.
apply add_nxt.
Qed.

(* addition is associative *)
Lemma add_assoc: forall a b c: N, add(add a b)(c) = add(a)(add b c).
Proof.
intros a b c.
induction a.
simpl.
reflexivity.
simpl.
f_equal.
assumption.
Qed.

(********************************************************************)

(* multiplication *)
Fixpoint mult(m n: N): N :=
match n with
| Z => Z
| NXT n1 => add(mult m n1)(m)
end.

Lemma mult_zero: forall n: N, mult Z n = Z.
Proof.
induction n.
simpl.
reflexivity.
simpl.
rewrite IHn.
simpl.
reflexivity.
Qed.

Lemma mult_one: forall n: N, mult n (NXT Z) = n.
Proof.
intros.
simpl.
reflexivity.
Qed.

Lemma mult_nxt: forall m n: N, mult (NXT m) n = add (mult m n) n.
Proof.
intros m.
induction n.
simpl.
reflexivity.
simpl.
rewrite IHn.
rewrite add_assoc.
rewrite add_assoc.
f_equal.
rewrite add_comm with (m:=n).
rewrite add_comm with (m:=m).
simpl.
f_equal.
apply add_comm.
Qed.

(* multiplication is commutative *)
Lemma mult_comm: forall m n: N, (mult m n) = (mult n m).
Proof.
intros m.
induction n.
simpl.
rewrite mult_zero.
reflexivity.
simpl.
rewrite mult_nxt.
rewrite IHn.
reflexivity.
Qed.

(* multiplication (at left) distributes over addition *)
Lemma mult_distr_add_l: forall a b c: N,
      mult(a)(add b c) = add(mult a b)(mult a c).
Proof.
intros a b c.
induction b.
simpl.
reflexivity.
simpl.
rewrite IHb.
rewrite add_assoc.
rewrite add_assoc.
f_equal.
apply add_comm.
Qed.

(* multiplication (at right) distributes over addition *)
Lemma mult_distr_add_r: forall a b c: N,
      mult(add a b)(c) = add(mult a c)(mult b c).
Proof.
intros a b c.
rewrite mult_comm.
rewrite mult_distr_add_l.
rewrite mult_comm with (m:=c).
rewrite mult_comm with (m:=c).
reflexivity.
Qed.

(* multiplication is associative *)
Lemma mult_assoc: forall a b c: N, mult(mult a b)(c) = mult(a)(mult b c).
Proof.
intros a b c.
induction c.
simpl.
reflexivity.
simpl.
rewrite IHc.
rewrite mult_distr_add_l.
reflexivity.
Qed.

(********************************************************************)

(* subtraction *)
(* Since we are dealing with natural numbers only, we define (sub m n)
   to be zero if m is less than n. *)

Fixpoint sub(m n: N): N :=
match m with
| Z => Z
| NXT m1 => (match n with
             | Z => m
             | NXT n1 => sub m1 n1
             end)
end.

Lemma sub_same: forall n: N, sub n n = Z.
Proof.
induction n.
simpl.
reflexivity.
simpl.
assumption.
Qed.

Lemma sub_zero: forall n: N, n = sub n Z.
Proof.
intros n.
case n.
simpl.
reflexivity.
simpl.
reflexivity.
Qed.

Lemma add_sub: forall m n: N, sub(add m n) n = m.
Proof.
intros m.
induction n.
rewrite <- add_zero.
rewrite <- sub_zero.
reflexivity.
rewrite <- add_nxt.
simpl.
assumption.
Qed.

(********************************************************************)

(* less than or equal to (<=) *)
Inductive lq: N->N->Prop :=
| lq_same: forall n:N, lq n n
| lq_nxt: forall m n: N, lq m n -> lq m (NXT n).

Lemma lq_add1: forall a b: N, lq a b -> lq (NXT a) (NXT b).
Proof.
intros a b H.
induction b.
inversion H.
constructor.
inversion H.
constructor.
apply IHb in H2.
constructor.
assumption.
Qed.

Lemma add_lq: forall a b c: N, lq a b -> lq (add c a) (add c b).
Proof.
intros a b c H.
induction c.
simpl.
assumption.
simpl.
apply lq_add1.
assumption.
Qed.

(* <= is transitive *)
Lemma lq_trans: forall a b c: N, lq a b -> lq b c -> lq a c.
Proof.
intros a b c H1 H2.
induction c.
inversion H2.
subst b.
subst n.
assumption.
inversion H2.
rewrite <- H0.
assumption.
apply IHc in H3.
constructor.
assumption.
Qed.

Lemma mult_lq: forall a b m: N, lq a b -> lq (mult a m) (mult b m).
Proof.
intros a b m H.
induction m.
simpl.
constructor.
simpl.
assert(H1: lq (add (mult a m) a) (add (mult b m) a)).
rewrite add_comm with (n:=a).
rewrite add_comm with (n:=a).
apply add_lq.
assumption.
assert(H2: lq (add (mult b m) a) (add (mult b m) b)).
apply add_lq.
assumption.
apply lq_trans with (b:=(add (mult b m) a)).
assumption.
assumption.
Qed.