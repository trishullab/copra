`conv start`

`example_user`
[THEOREM] theorem basic_arithmetic: 
  fixes x::real
  shows "(x+y)*(x+2*y)*(x+3*y) = x^3 + 6*x^2*y + 11*x*y^2 + 6*y^3"
[END]

`example_assistant`
[PROOF]
proof -
  show ?thesis by (simp add: algebra_simps eval_nat_numeral)
qed
[END]

`conv end`