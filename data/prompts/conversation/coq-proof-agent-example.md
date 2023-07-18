`conv start`

`example_user`
Goals to prove:
[GLS]
[GL] 1
forall a : nat, a + 0 = a
[HYPS] 1


[STPS]


`example_assistant`
[RUN TACTIC][STP]intros.
[END]


`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]
New goals to prove:
[GLS]
[GL] 1
n + 0 = n
[HYPS] 1
[HYP] n: nat


[STPS]


`example_assistant`
[RUN TACTIC][STP]reflexivity.
[END]


`example_user`
[RUN TACTIC RESULT][ERROR]
In environment
n : nat
Unable to unify "n" with "n + 0".
[END]


`example_assistant`
[GET THMS]
[END]


`example_user`
[GET THMS RESULT]
[THMS] 1
[THM]plus_n_O : forall n  nat, n = n + 0
[THM]plus_O_n : forall n  nat, 0 + n = n
[THM]mult_n_O : forall n  nat, 0 = n * 0
[THM]plus_n_Sm : forall n m  nat, S (n
[END]


`example_assistant`
[RUN TACTIC]
[STP]symmetry.
[STP]apply plus_n_O.
[STP]Qed.
[END]


`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]

`conv end`