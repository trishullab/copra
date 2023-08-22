`conv start`

`example_user`
[THEOREM] algb_nat_zero : forall a, 0 + a = a.

[DEFINITIONS]
[DEFINITION] nat: Set

[LEMMAS]
[LEMMA] plus_O_n: forall n : nat, 0 + n = n
[LEMMA] plus_n_O: forall n : nat, n = n + 0
[LEMMA] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[LEMMA] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[END]

`example_assistant`
Proof.
    intros.
    induction a.
    simpl.
    reflexivity.
    simpl.
    reflexivity.
Qed.

`conv end`