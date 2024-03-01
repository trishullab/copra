`conv start`

`example_user`
[THEOREM] some_theorem
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=
[END]

`example_assistant`
[PROOF]
begin
  induction h₀ with k h₀ IH,
  { norm_num },
  {
    calc k + 1 < 2 * 2^k : by linarith
  }
end
[END]

`example_user`
[THEOREM] some_theorem
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * x^4 - 7 * x^3 + 2 * x^2 - b * x + 1)
  (h₁ : f 1 = 1) :
  b = -2 :=
[END]

`example_assistant`
[PROOF]
begin
  rw h₀ at h₁,
  linarith,
end
[END]

`example_user`
[THEOREM] some_theorem
  int.floor ((9:ℝ) / 160 * 100) = 5 :=
[END]

`example_assistant`
[PROOF]
begin
  rw int.floor_eq_iff,
  split,
  all_goals { norm_num },
end
[END]

`conv end`