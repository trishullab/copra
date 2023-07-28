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
[RUN TACTIC][STP]simpl.
[END]


`example_user`
[RUN TACTIC RESULT][ERROR]Goal didn't simplify. It is same as one of the previous goals attempted for simplification[END]


`example_assistant`
[RUN TACTIC][STP]rewrite Nat.add_0_r.
[END]


`example_user`
[RUN TACTIC RESULT][ERROR]Error: Found no subterm matching "?M1050 + 0" in the current goal.
[END]


`example_assistant`
[RUN TACTIC][STP]intros a. reflexivity.
[END]

`example_user`
[RUN TACTIC RESULT][ERROR]Error: In environment
a : nat
Unable to unify "a" with "a + 0".
[END]

`example_assistant`
[RUN TACTIC][STP]induction a as [| a' IH].
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
S a' + 0 = S a'
[HYPS] 2

[STPS]

`example_assistant`
[RUN TACTIC][STP]reflexivity.
[END]

`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]

Goals to prove:
[GLS]
[GL] 1
S a' + 0 = S a'
[HYPS] 1
a': nat
IH: a' + 0 = a'

[STPS]

`example_assistant`
[RUN TACTIC][STP]rewrite IH. reflexivity.
[END]

`example_user`
[RUN TACTIC RESULT][ERROR]Error: Found no subterm matching "a' + 0" in the current goal.
[END]

`example_assistant`
[RUN TACTIC][STP]simpl.[END]

`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]

Goals to prove:
[GLS]
[GL] 1
S (a' + 0) = S a'
[HYPS] 1
a': nat
IH: a' + 0 = a'

[STPS]

`example_assistant`
[RUN TACTIC][STP]f_equal. apply IH.
[END]

`example_user`
[RUN TACTIC RESULT][SUCCESS]
[END]

Goals to prove:
[GLS]

[STPS]

`example_assistant`
[RUN TACTIC][STP]Qed.
[END]

`conv end`