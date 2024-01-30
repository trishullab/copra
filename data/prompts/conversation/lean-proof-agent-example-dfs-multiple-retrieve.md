`conv start`

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
y = 1
[HYPOTHESES] 1
[HYPOTHESIS] x y : ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1
[THEOREMS] 1
[THEOREM] nat.triangle_succ: (n : ℕ) : (n + 1) * ((n + 1) - 1) / 2 = n * (n - 1) / 2 + n
[THEOREM] inner_product_geometry.mul_norm_eq_abs_sub_sq_norm: {x y z : V}   (h₁ : ∃ k : ℝ, k ≠ 1 ∧ x + y = k • (x - y)) (h₂ : ∥z - y∥ = ∥z + y∥) :   ∥x - y∥ * ∥x + y∥ = |∥z + y∥ ^ 2 - ∥z - x∥ ^ 2|
[THEOREM] complex.sub_conj: (z : ℂ) : z - conj z = (2 * z.im : ℝ) * I
[THEOREM] euclidean_geometry.mul_dist_eq_abs_sub_sq_dist: {a b p q : P}   (hp : ∃ k : ℝ, k ≠ 1 ∧ b -ᵥ p = k • (a -ᵥ p)) (hq : dist a q = dist b q) :   dist a p * dist b p = |dist b q ^ 2 - dist p q ^ 2|

[STEPS]
[STEP] split,
[STEP] {
  rw h₁ at h₂,
  linarith
},

[INCORRECT STEPS]
[STEP] rw h₂ at h₁,

[LAST STEP]
rw ←h₁,

[ERROR MESSAGE]
rewrite tactic failed, did not find instance of the pattern in the target expression
  3 - 2 * y
[END]

`example_assistant`
[RUN TACTIC]
rw h₁ at h₂,
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
n = 57
[HYPOTHESES] 1
[HYPOTHESIS] n : ℕ
[HYPOTHESIS] h₀ : n < 398 
[HYPOTHESIS] h₁ : (n * 7) % 398 = 1
[THEOREMS] 1
[THEOREM] tactic.is_prime_helper: (n : ℕ)   (h₁ : 1 < n) (h₂ : nat.min_fac n = n) : nat.prime n
[THEOREM] nat.one_mul: (n : ℕ) : 1 * n = n
[THEOREM] nat.mul_one:  ∀ (n : ℕ), n * 1 = n
[THEOREM] nat.lt_trans: {n m k : ℕ} (h₁ : n < m) : m < k → n < k
[THEOREM] nat.lt_of_le_of_lt: {n m k : ℕ} (h₁ : n ≤ m) : m < k → n < k

[LAST STEP]
norm_num,
[ERROR MESSAGE]
norm_num failed to simplify
state:
n : ℕ,
h₀ : n < 398,
h₁ : n * 7 % 398 = 1
⊢ n = 57
[END]

`example_assistant`
[RUN TACTIC]
have h₂ : 1 % 398 = (57 * 7) % 398 := by norm_num
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
s 5 = 16
[HYPOTHESES] 1
[HYPOTHESIS] (s : ℕ → ℝ)
[HYPOTHESIS] h₀: ∀ n, s (n + 2) = s (n + 1) + s n
[HYPOTHESIS] h₁ : s 9 = 110
[HYPOTHESIS] h₂ : s 7 = 42
[HYPOTHESIS] h : ∀ (n : ℕ), s n = s (n + 2) - s (n + 1),
[HYPOTHESIS] h₃ : s 9 = s 8 + s 7,
[HYPOTHESIS] h₄ : s 8 = 68
[THEOREMS] 1
[THEOREM] nat.triangle_succ: (n : ℕ) : (n + 1) * ((n + 1) - 1) / 2 = n * (n - 1) / 2 + n
[THEOREM] nat.add_left_cancel:  ∀ {n m k : ℕ}, n + m = n + k → m = k | 0        m k
[THEOREM] nat.fib_add_two_sub_fib_add_one: {n : ℕ} : fib (n + 2) - fib (n + 1) = fib n
[THEOREM] nat.add_left_comm:  ∀ (n m k : ℕ), n + (m + k) = m + (n + k)


[STEP] have h : ∀ n, s n = s (n + 2) - s (n + 1) := by { intro n, simp [h₀ n] }, 
[STEP] have h₃ : s 9 = s 8 + s 7, from h₀ 7,
[STEP] have h₄ : s 8 = 68 := by linarith,


[LAST STEP]
linarith,
[ERROR MESSAGE]
linarith failed to find a contradiction
state:
s : ℕ → ℕ,
h₀ : ∀ (n : ℕ), s (n + 2) = s (n + 1) + s n,
h₁ : s 9 = 110,
h₂ : s 7 = 42,
h : ∀ (n : ℕ), s n = s (n + 2) - s (n + 1),
h₃ : s 9 = s 8 + s 7,
h₄ : s 8 = 68,
ᾰ : s 5 < 16
⊢ false
[END]

`example_assistant`
[RUN TACTIC]
rw (h 5),
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
3 ∣ d ^ 3 + 2 * d + 3 * (d ^ 2 + d + 1)
[HYPOTHESES] 1
[HYPOTHESIS] d : ℕ
[HYPOTHESIS] id : 3 ∣ d ^ 3 + 2 * d
[HYPOTHESIS] id_expanded : (d + 1) ^ 3 + 2 * (d + 1) = d ^ 3 + 3 * d ^ 2 + 5 * d + 3
[HYPOTHESIS] id_factored : d ^ 3 + 3 * d ^ 2 + 5 * d + 3 = d ^ 3 + 2 * d + 3 * (d ^ 2 + d + 1)
[THEOREMS] 1
[THEOREM] nat.triangle_succ: (n : ℕ) : (n + 1) * ((n + 1) - 1) / 2 = n * (n - 1) / 2 + n
[THEOREM] complex.norm_sq_add_mul_I: (x y : ℝ) : norm_sq (x + y * I) = x ^ 2 + y ^ 2
[THEOREM] nat.prime_three:  prime 3
[THEOREM] complex.norm_sq_add: (z w : ℂ) : norm_sq (z + w) =   norm_sq z + norm_sq w + 2 * (z * conj w).re

[STEPS]
[STEP] induction n with d id,
[STEP] simp,
[STEP] have id_expanded : (d + 1)^3 + 2 * (d + 1) = d^3 + 3 * d^2 + 5 * d + 3 := by ring,
[STEP] rw id_expanded,
[STEP] have id_factored : d^3 + 3 * d^2 + 5 * d + 3 = d^3 + 2 * d + 3 * (d^2 + d + 1) := by ring,

[LAST STEP]
rw id_factored,
[SUCCESS]
[END]

`example_assistant`
[RUN TACTIC]
  apply dvd_add id,
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
b = -2
[HYPOTHESES] 1
[HYPOTHESIS] b : ℝ
[HYPOTHESIS] f : ℝ → ℝ,
[HYPOTHESIS] h₀ : ∀ (x : ℝ), f x = 3 * x ^ 4 - 7 * x ^ 3 + 2 * x ^ 2 - b * x + 1,
[HYPOTHESIS] h₁ : 3 * 1 ^ 4 - 7 * 1 ^ 3 + 2 * 1 ^ 2 - b * 1 + 1 = 1
[THEOREMS] 1
[THEOREM] nat.triangle_succ: (n : ℕ) : (n + 1) * ((n + 1) - 1) / 2 = n * (n - 1) / 2 + n
[THEOREM] inner_product_geometry.mul_norm_eq_abs_sub_sq_norm: {x y z : V}   (h₁ : ∃ k : ℝ, k ≠ 1 ∧ x + y = k • (x - y)) (h₂ : ∥z - y∥ = ∥z + y∥) :   ∥x - y∥ * ∥x + y∥ = |∥z + y∥ ^ 2 - ∥z - x∥ ^ 2|
[THEOREM] complex.norm_sq_add_mul_I: (x y : ℝ) : norm_sq (x + y * I) = x ^ 2 + y ^ 2
[THEOREM] complex.sub_conj: (z : ℂ) : z - conj z = (2 * z.im : ℝ) * I

[LAST STEP]
rw h₀ at h₁,
[SUCCESS]
[END]

`example_assistant`
[RUN TACTIC]
linarith,
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
a + b = b + a
[HYPOTHESES] 1
[HYPOTHESIS] a b : ℕ
[THEOREMS] 1
[THEOREM] nat.add_right_comm:  ∀ (n m k : ℕ), n + m + k = n + k + m
[THEOREM] nat.add_assoc:  ∀ n m k : ℕ, (n + m) + k = n + (m + k) | n m 0
[THEOREM] pnat.add_coe: (m n : ℕ+) : ((m + n : ℕ+) : ℕ) = m + n
[THEOREM] nat.add_comm:  ∀ n m : ℕ, n + m = m + n | n 0
[END]

`example_assistant`
[RUN TACTIC]
rw nat.add_comm,
[END]

`conv end`