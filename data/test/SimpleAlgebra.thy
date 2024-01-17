theory Scratch
    imports Complex_Main
begin

theorem n_less_2_pow_n: 
    shows "n <= 2^n"
    sorry

lemma basic_arithmetic:
  fixes x::real
  shows "(x+y)*(x+2*y)*(x+3*y) = x^3 + 6*x^2*y + 11*x*y^2 + 6*y^3"
  by (simp add: algebra_simps eval_nat_numeral)

lemma sqrt_comp: "sqrt 2 + sqrt 3 < sqrt 10"
proof -
  have "(sqrt 2 + sqrt 3)^2 < (sqrt 10)^2"
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
proof
  assume "sqrt 2 \<in> \<rat>"
  then obtain q::rat where "sqrt 2 = of_rat q"
    using Rats_cases by blast
  then have "q^2 = 2"
    by (metis abs_numeral of_rat_eq_iff of_rat_numeral_eq of_rat_power power2_eq_square
              real_sqrt_mult_self)
  then obtain m n
    where "coprime m n" "q = of_int m / of_int n"
    by (metis Fract_of_int_quotient Rat_cases)
  then have "of_int m ^ 2 / of_int n ^ 2 = (2::rat)"
    by (metis ‹q⇧2 = 2› power_divide)
  then have 2: "of_int m ^ 2 = (2::rat) * of_int n ^ 2"
    by (metis division_ring_divide_zero double_eq_0_iff mult_2_right mult_zero_right
              nonzero_divide_eq_eq)
  then have "2 dvd m"
    by (metis (mono_tags, lifting) even_mult_iff even_numeral of_int_eq_iff of_int_mult
              of_int_numeral power2_eq_square)
  then obtain r where "m = 2*r"
    by blast
  then have "2 dvd n"
    by (smt (verit) "2" ‹even m› dvdE even_mult_iff mult.left_commute mult_cancel_left of_int_1 
            of_int_add of_int_eq_iff of_int_mult one_add_one power2_eq_square)
  then show False
    using ‹coprime m n› ‹m = 2 * r› by simp
qed

end