`conv start`

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
forall a : nat, a + 0 = a
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[END]


`example_assistant`
[RUN TACTIC]
reflexivity.
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
forall a : nat, a + 0 = a
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[LAST STEP]
reflexivity.

[ERROR MESSAGE]
Error: In environment
a : nat
Unable to unify "a" with "a + 0".
[END]



`example_assistant`
[RUN TACTIC]
intros a.
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
a + 0 = a
[HYPOTHESES] 1
[HYPOTHESIS] a : nat
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[LAST STEP]
intros a.
[SUCCESS]

[END]


`example_assistant`
[RUN TACTIC]
rewrite Nat.add_0_r. reflexivity.
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
a + 0 = a
[HYPOTHESES] 1
[HYPOTHESIS] a : nat
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[STEPS]
[STEP] intros a.

[LAST STEP]
rewrite Nat.add_0_r. reflexivity.

[ERROR MESSAGE]
The reference Nat.add_0_r was not found in the current environment.
[END]

`example_assistant`
[RUN TACTIC]
rewrite Nat.add_r_0. reflexivity.
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
a + 0 = a
[HYPOTHESES] 1
[HYPOTHESIS] a : nat
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[STEPS]
[STEP] intros a.

[INCORRECT STEPS]
[STEP] rewrite Nat.add_0_r. reflexivity.

[LAST STEP]
rewrite Nat.add_r_0. reflexivity.

[ERROR MESSAGE]
The reference Nat.add_r_0 was not found in the current environment.
[END]

`example_assistant`
[RUN TACTIC]
induction a as [| n IHn].
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
0 + 0 = 0
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[GOAL] 2
S n + 0 = S n
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[STEPS]
[STEP] intros a.

[LAST STEP]
induction a as [| n IHn].

[SUCCESS]
[END]

`example_assistant`
[RUN TACTIC]
- reflexivity.
[END]

`example_user`
Goals to prove:
[GOALS]
[DESCRIPTION] There are unfocussed goals.
[GOAL] 1
S n + 0 = S n
[HYPOTHESES] 1
[HYPOTHESIS] n : nat
[HYPOTHESIS] IHn : n + 0 = n
[DEFINITIONS] 1
[DEFINITION] nat: Set
[THEOREMS] 1
[THEOREM] plus_O_n: forall n : nat, 0 + n = n
[THEOREM] plus_n_O: forall n : nat, n = n + 0
[THEOREM] plus_n_Sm: forall n m : nat, S (n + m) = n + S m
[THEOREM] plus_Sn_m: forall n m : nat, S n + m = S (n + m)

[STEPS]
[STEP] intros a.
[STEP] induction a as [| n IHn].

[LAST STEP]
- reflexivity.

[SUCCESS]
[END]

`example_assistant`
[RUN TACTIC]
- rewrite plus_n_O. reflexivity.
[END]

`conv end`