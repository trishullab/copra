`conv start`

`example_user`
[THEOREM] algb_nat_zero : forall a, 0 + a = a.

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