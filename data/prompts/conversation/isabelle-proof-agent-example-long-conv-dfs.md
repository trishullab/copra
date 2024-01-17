`conv start`

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
sqrt 2 + sqrt 3 < sqrt 10
[HYPOTHESES] 1

[END]


`example_assistant`
[RUN TACTIC]
proof -
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
sqrt 2 + sqrt 3 < sqrt 10
[HYPOTHESES] 1

[END]


`example_assistant`
[RUN TACTIC]
have "2 < 3" by simp 
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
sqrt 2 + sqrt 3 < sqrt 10
[HYPOTHESES] 1

[LAST STEP]
have "2 < 3" by simp 

[ERROR MESSAGE]
Step error: Failed to apply initial proof method (line 1):
goal (1 subgoal):
 1. (2::'a) < (3::'a)
 At command \"by\" (line 1)
[END]



`example_assistant`
[RUN TACTIC]
have "(sqrt 2 + sqrt 3)^2 < (sqrt 10)^2" 
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
(sqrt 2 + sqrt 3)\<^sup>2 < (sqrt 10)\<^sup>2 
[HYPOTHESES] 1

[LAST STEP]
have "(sqrt 2 + sqrt 3)^2 < (sqrt 10)^2" 
[SUCCESS]

[END]


`example_assistant`
[RUN TACTIC]
proof (simp add: algebra_simps eval_nat_numeral)
  have "(2 * (sqrt 2 * sqrt 3))^2 < 5 ^ 2"
    by (simp add: algebra_simps eval_nat_numeral)
  then show "2 * (sqrt 2 * sqrt 3) < 5"
    by (smt (verit, best) power_mono)
qed
[END]

`conv end`