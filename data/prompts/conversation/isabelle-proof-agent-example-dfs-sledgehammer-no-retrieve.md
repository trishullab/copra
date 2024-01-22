`conv start`

`example_user`
Goals to prove: 
[GOALS] 
[GOAL] 1 (x + y) * (x + 2 * y) * (x + 3 * y) = x ^ 3 + 6 * x<^sup>2 * y + 11 * x * y<^sup>2 + 6 * y ^ 3 
[HYPOTHESES] 1 
[HYPOTHESIS] shows "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)" 

[LAST STEP] have "(x + y) * (x + 2 * y) * (x + 3 * y) = (x + y) * (x + y + y) * (x + y + 2 * y)" by (simp add: algebra_simps)

[SUCCESS] 
[END]

`example_assistant` 
[RUN TACTIC]
have "(x + y) * (x + y + y) * (x + y + 2 * y) = x ^ 3 + 6 * x<^sup>2 * y + 11 * x * y<^sup>2 + 6 * y ^ 3" by (simp add: algebra_simps)
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
"(x+1)^2 * (y+5)^3 mod 19 = (13:: nat)"
[HYPOTHESES] 1
[HYPOTHESIS] x y :: nat
[HYPOTHESIS] h0 : "x mod 19 = (4:: nat)"
[HYPOTHESIS] h1 : "y mod 19 = (7:: nat)"

[STEPS]
[STEP] have "(x+1) mod 19 = 5" using h0 by (metis One_nat_def Suc_1 Suc_eq_plus1 eval_nat_numeral(3) mod_Suc num.size(4) num.size(5) num.size (6) numeral_eq_iff old.nat.inject zero_neq_numeral)
[STEP] then have "(x+1)^2 mod 19 = 5^2 mod 19" by (metis power_mod)

[LAST STEP]
have ?thesis sledgehammer

[ERROR MESSAGE]
Step error: sledgehammer failed to find a proof before timing out.
[END]

`example_assistant`
[RUN TACTIC]
also have "... = 6" sledgehammer   
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
"(x+1)^2 * (y+5)^3 mod 19 = (13:: nat)"
[HYPOTHESES] 1
[HYPOTHESIS] x y :: nat
[HYPOTHESIS] h0 : "x mod 19 = (4:: nat)"
[HYPOTHESIS] h1 : "y mod 19 = (7:: nat)"

[STEPS]
[STEP] have "(x+1) mod 19 = 5" using h0 sledgehammer

[LAST STEP]
then have "(x+1)^2 mod 19 = 5^2 mod 19" by simp

[ERROR MESSAGE]
Failed to finish proof⌂:
goal (1 subgoal):
 1. Suc x mod 19 = 5 ⟹ (Suc x)⇧2 mod 19 = 6
[END]

`example_assistant`
[RUN TACTIC]
then have "(x+1)^2 mod 19 = 5^2 mod 19" sledgehammer
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
s 5 = 16
[HYPOTHESES] 1
[HYPOTHESIS] (s : ℕ → ℝ)
[HYPOTHESIS] h0 : "\<And>n. s (n+2) = s (n+1) + s n"
[HYPOTHESIS] h1 : "s 9 = 110"
[HYPOTHESIS] h2 : "s 7 = 42"

[STEPS]
[STEP] have "s 8 = 68" using h1 h2 h0[of 7] by simp
[STEP] hence h3: "s 6 = 26" using h2 h0[of 6] by simp

[LAST STEP]
thus "s 5 = 16" by linarith

[ERROR MESSAGE]
Failed to apply initial proof method⌂:
using this:
  s 6 = 26
goal (1 subgoal):
 1. s 5 = 16
[END]

`example_assistant`
[RUN TACTIC]
show ?thesis sledgehammer
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
3 dvd 2 ^ (2 * 0 + 1) + 1
[HYPOTHESES] 1
[HYPOTHESIS] n :: nat
[GOAL] 2
⋀n. 3 dvd 2 ^ (2 * n + 1) + 1 ⟹ 3 dvd 2 ^ (2 * Suc n + 1) + 1
[HYPOTHESES] 2
[HYPOTHESIS] n :: nat

[STEPS]
[STEP] (induct n)

[LAST STEP]
case 0
[SUCCESS]
[END]

`example_assistant`
[RUN TACTIC]
  then show ?case by auto
[END]

`conv end`