`conv start`

`example_user`
[THEOREM] some_theorem
  (n : ℕ)
  (h0 : 1 ≤ n) :
  n < 2^n := by
[END]

`example_assistant`
[PROOF]
by
induction' h0 with k h0 ih
norm_num
rw [succ_eq_add_one]
rw [pow_add, pow_one, mul_two]
refine add_lt_add_of_lt_of_lt ih ?step.h₂
exact one_lt_two_pow k h0
[END]

`example_user`
[THEOREM] some_theorem
  (b : ℝ)
  (f : ℝ → ℝ)
  (h0 : ∀ x, f x = 3 * x^4 - 7 * x^3 + 2 * x^2 - b * x + 1)
  (h1 : f 1 = 1) :
  b = -2 := 
[END]

`example_assistant`
[PROOF]
rw [h0] at h1
norm_num at h1
have h2 : (-2 - b) + b = 0 + b := by rw [←h1]
simp at h2
exact h2.symm 
[END]

`example_user`
[THEOREM] some_theorem
  Int.floor ((9:ℝ) / 160 * 100) = 5 :=
[END]

`example_assistant`
[PROOF]
by
rw [Int.floor_eq_iff]
norm_num
[END]

`conv end`