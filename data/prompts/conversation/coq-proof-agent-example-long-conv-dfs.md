`conv start`

`example_user1`
Goals to prove:
[GOALS]
[GOAL] 1
forall a : nat, a + 0 = a
[HYPOTHESES] 1

[END]


`example_assistant1`
[RUN TACTIC][STEP]reflexivity.
[END]


`example_user2`
Goals to prove:
[GOALS]
[GOAL] 1
forall a : nat, a + 0 = a
[HYPOTHESES] 1

[STEPS]

[INCORRECT STEPS]
[STEP]reflexivity.

[ERROR MESSAGE]Error: In environment
a : nat
Unable to unify "a" with "a + 0".
[END]



`example_assistant2`
[RUN TACTIC]
[STEP]intros a.
[END]


`example_user3`
Goals to prove:
[GOALS]
[GOAL] 1
a + 0 = a
[HYPOTHESES] 1
[HYPOTHESIS] a : nat

[STEPS]
[STEP] intros a.
[END]


`example_assistant3`
[RUN TACTIC][STEP]rewrite Nat.add_0_r. reflexivity.
[END]


`example_user4`
Goals to prove:
[GOALS]
[GOAL] 1
a + 0 = a
[HYPOTHESES] 1
[HYPOTHESIS] a : nat

[STEPS]
[STEP] intros a.

[INCORRECT STEPS]
[STEP]rewrite Nat.add_0_r. reflexivity.

[ERROR MESSAGE]The reference Nat.add_0_r was not found in the current environment.
[END]


`example_assistant4`
[RUN TACTIC][STEP]induction a as [| n IHn].
[END]

`example_user5`
Goals to prove:
[GOALS]
[GOAL] 1
0 + 0 = 0
[HYPOTHESES] 1

[GOAL] 2
S n + 0 = S n
[HYPOTHESES] 2

[STEPS]
[STEP] intros a.
[STEP] induction a as [| n IHn].

[END]

`example_assistant5`
[RUN TACTIC][STEP] - reflexivity.
[END]

`example_user6`
Goals to prove:
[GOALS]
[DESCRIPTION] There are unfocussed goals.
[GOAL] 1
S n + 0 = S n
[HYPOTHESES] 1
[HYPOTHESIS] n : nat
[HYPOTHESIS] IHn : n + 0 = n

[STEPS]
[STEP] intros a.
[STEP] induction a as [| n IHn].
[STEP] - reflexivity.
[END]


`example_assistant6`
[GET THEOREMS][END]

`example_user7`
Goals to prove:
[GOALS]
[DESCRIPTION] There are unfocussed goals.
[GOAL] 1
S n + 0 = S n
[HYPOTHESES] 1
[HYPOTHESIS] n : nat
[HYPOTHESIS] IHn : n + 0 = n
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[STEPS]
[STEP] intros a.
[STEP] induction a as [| n IHn].
[STEP] - reflexivity.
[END]

`example_assistant7`
[RUN TACTIC][STEP]- rewrite plus_n_O. reflexivity.
[END]

`example_user8`
Goals to prove:
[GOALS]
[DESCRIPTION] Proof finished.

[STEPS]
[STEP] intros a.
[STEP] induction a as [| n IHn].
[STEP] - reflexivity.
[STEP] - rewrite plus_n_O. reflexivity.
[END]

`example_assistant8`
[RUN TACTIC][STEP]Qed.
[END]

`conv end`