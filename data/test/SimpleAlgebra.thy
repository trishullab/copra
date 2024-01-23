theory Scratch
    imports Complex_Main
begin

theorem n_less_2_pow_n: 
    shows "n <= 2^n"
proof -
  show ?thesis sledgehammer
qed

lemma basic_arithmetic:
  fixes x::real
  shows "(x+y)*(x+2*y)*(x+3*y) = x^3 + 6*x^2*y + 11*x*y^2 + 6*y^3"
  by (simp add: algebra_simps eval_nat_numeral)

lemma sqrt_comp: "sqrt 2 + sqrt 3 < sqrt 10"
proof -
  have "(sqrt 2 + sqrt 3)^2 
    < (sqrt 10)^2"
  proof (simp add: algebra_simps eval_nat_numeral)
    have "(2 * (sqrt 2 * sqrt 3))^2 < 5 ^ 2"
      by (simp add: algebra_simps eval_nat_numeral)
    then show "2 * (sqrt 2 * sqrt 3) < 5"
      by (smt (verit, best) power_mono)
  qed
  then show ?thesis
    by (simp add: real_less_rsqrt)
qed

lemma sqrt_two_irrational: "sqrt 2 \<notin> \<rat>"
sorry

end