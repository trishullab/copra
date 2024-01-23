`conv start`

`example_user`
[THEOREM] lemma sqrt_comp: "sqrt 2 + sqrt 3 < sqrt 10"

[INFORMAL-THEOREM]
The square root of 2 plus the square root of 3 is less than the square root of 10.
[INFORMAL-PROOF]
We know that $(2 * sqrt 2 * sqrt 3)^2 < 5^2$ by just evaluating both sides, which simplifies to $24 < 25$.
Then we take the square root of both sides, obtaining $2 * sqrt 2 * sqrt 3 < 5$.
We add 5 to each side as follows: $2 + 2 * (sqrt 2 sqrt 3) + 3 < 10$.
Then we factor the expression, obtaining $(sqrt 2 + sqrt 3)^2 < 10$.
Taking the square root of both sides completes the proof.

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