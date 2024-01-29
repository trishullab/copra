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

[LAST STEP]
rw h₀ at h₁,
[SUCCESS]
[END]

`example_assistant`
[RUN TACTIC]
linarith,
[END]

`conv end`