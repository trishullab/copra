`conv start`

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
(x + y) * (x + 2 * y) * (x + 3 * y) = x ^ 3 + 6 * x\<^sup>2 * y + 11 * x * y\<^sup>2 + 6 * y ^ 3
[THEOREMS] 1
[THEOREM] Power.comm_semiring_1_class.power2_sum : test power2_sum: fixes x :: "'a"   and y :: "'a" shows "(x + y)\<^sup>2 = x\<^sup>2 + y\<^sup>2 + (2::'a) * x * y"
[THEOREM] Complex.Complex_divide : test Complex_divide: fixes x :: "complex"   and y :: "complex" shows "x / y = Complex ((Re x * Re y + Im x * Im y) / ((Re y)\<^sup>2 + (Im y)\<^sup>2)) ((Im x * Re y - Re x * Im y) / ((Re y)\<^sup>2 + (Im y)\<^sup>2))"
[THEOREM] Power.comm_ring_1_class.power2_diff : test power2_diff: fixes x :: "'a"   and y :: "'a" shows "(x - y)\<^sup>2 = x\<^sup>2 + y\<^sup>2 - (2::'a) * x * y"
[END]


`example_assistant`
[RUN TACTIC]
have "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)" by (simp add: algebra_simps)
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
(x + y) * (x + 2 * y) * (x + 3 * y) = x ^ 3 + 6 * x\<^sup>2 * y + 11 * x * y\<^sup>2 + 6 * y ^ 3
[HYPOTHESES] 1
[HYPOTHESIS] shows "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)"
[THEOREMS] 1
[THEOREM] Power.comm_semiring_1_class.power2_sum : test power2_sum: fixes x :: "'a"   and y :: "'a" shows "(x + y)\<^sup>2 = x\<^sup>2 + y\<^sup>2 + (2::'a) * x * y"
[THEOREM] Complex.Complex_divide : test Complex_divide: fixes x :: "complex"   and y :: "complex" shows "x / y = Complex ((Re x * Re y + Im x * Im y) / ((Re y)\<^sup>2 + (Im y)\<^sup>2)) ((Im x * Re y - Re x * Im y) / ((Re y)\<^sup>2 + (Im y)\<^sup>2))"
[THEOREM] Power.comm_ring_1_class.power2_diff : test power2_diff: fixes x :: "'a"   and y :: "'a" shows "(x - y)\<^sup>2 = x\<^sup>2 + y\<^sup>2 - (2::'a) * x * y"

[LAST STEP]
have "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)" by (simp add: algebra_simps)

[SUCCESS]
[END]


`example_assistant`
[RUN TACTIC]
have "(x + y) * (x + y + y) * (x + y + 2 * y) = x ^ 3 + 6 * x\<^sup>2 * y + 11 * x * y\<^sup>2 + 6 * y ^ 3" by (simp add: algebra_simps)
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
(x + y) * (x + 2 * y) * (x + 3 * y) = x ^ 3 + 6 * x\<^sup>2 * y + 11 * x * y\<^sup>2 + 6 * y ^ 3
[HYPOTHESES] 1
[HYPOTHESIS] shows "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)"
[THEOREMS] 1
[THEOREM] Power.comm_semiring_1_class.power2_sum : test power2_sum: fixes x :: "'a"   and y :: "'a" shows "(x + y)\<^sup>2 = x\<^sup>2 + y\<^sup>2 + (2::'a) * x * y"
[THEOREM] Complex.Complex_divide : test Complex_divide: fixes x :: "complex"   and y :: "complex" shows "x / y = Complex ((Re x * Re y + Im x * Im y) / ((Re y)\<^sup>2 + (Im y)\<^sup>2)) ((Im x * Re y - Re x * Im y) / ((Re y)\<^sup>2 + (Im y)\<^sup>2))"
[THEOREM] Power.comm_ring_1_class.power2_diff : test power2_diff: fixes x :: "'a"   and y :: "'a" shows "(x - y)\<^sup>2 = x\<^sup>2 + y\<^sup>2 - (2::'a) * x * y"

[STEPS]
[STEP] have "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)"
  by (simp add: algebra_simps)

[LAST STEP]
have "(x + y) * (x + y + y) * (x + y + 2 * y) = x ^ 3 + 6 * x\<^sup>2 * y + 11 * x * y\<^sup>2 + 6 * y ^ 3"
  by (simp add: algebra_simps)

[ERROR MESSAGE]
Step error: Failed to finish proof (line 2):
goal (1 subgoal):
 1. x * (x * x) + (x * (x * (y * 6)) + (y * (y * (y * 6)) + x * (y * (y * 11)))) = x ^ 3 + (6 * y ^ 3 + (y * (6 * x\<^sup>2) + x * (11 * y\<^sup>2)))
At command "by" (line 2)
[END]

`conv end`