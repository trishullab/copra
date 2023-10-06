`conv start`

`example_user`
[THEOREM] n_less_2_pow_n
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=
[END]

`example_assistant`
begin
  induction h₀ with k h₀ IH,
  { norm_num },
  {
    calc k + 1 < 2 * 2^k : by linarith
  }
end

`conv end`