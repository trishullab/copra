`conv start`

`example_user`
[THEOREM] lemma sqrt_comp: "sqrt 2 + sqrt 3 < sqrt 10"
[END]

`example_assistant`
[PROOF]
proof -
  have "(2 * (sqrt 2 * sqrt 3))^2 < 5 ^ 2" by (simp add: algebra_simps eval_nat_numeral)
  then have "2 * (sqrt 2 * sqrt 3) < 5" sledgehammer
  then have "(sqrt 2 + sqrt 3)^2 < (sqrt 10)^2" by (simp add: algebra_simps eval_nat_numeral)
  then show ?thesis sledgehammer
qed
[END]

`conv end`