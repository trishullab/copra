`conv start`

`example_user`
Goals to prove:
[GLS]
[GL] 1
forall a : nat, a + 0 = a
[HYPS] 1

[STPS]


`example_assistant`
[RUN TACTIC][STP]reflexivity.
[END]


`example_user`
[RUN TACTIC RESULT][ERROR]Error: In environment
a : nat
Unable to unify "a" with "a + 0".
[END]


`example_assistant`
[RUN TACTIC][STP]intros a. simpl. reflexivity.
[END]


`example_user`
[RUN TACTIC RESULT][ERROR]In environment
a : nat
Unable to unify "a" with
 "a + 0".[END]


`example_assistant`
[RUN TACTIC][STP]rewrite Nat.add_0_r. reflexivity.
[END]


`example_user`
[RUN TACTIC RESULT][ERROR]The reference Nat.add_0_r was not found in the current environment.
[END]


`example_assistant`
[RUN TACTIC][STP]intros a. induction a as [| n IHn].
[END]

`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]

Goals to prove:
[GLS]
[GL] 1
0 + 0 = 0
[HYPS] 1

[GL] 2
S n + 0 = S n
[HYPS] 2

[STPS]

`example_assistant`
[RUN TACTIC][STP]- reflexivity.
[END]


`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]

Goals to prove:
[GLS]
[DESCRIPTION]
There are unfocussed goals.
[END]

`example_assistant`
[RUN TACTIC][STP]- simpl. rewrite <- IHn. reflexivity.
[END]

`example_user`
[RUN TACTIC RESULT][ERROR]Error: In environment
n : nat
IHn : n + 0 = n
Unable to unify "S (n + 0)"
with "S (n + 0 + 0)".
[END]

`example_assistant`
[RUN TACTIC][STP]- simpl. rewrite IHn. reflexivity.
[END]

`conv end`