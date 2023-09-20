import data.real.basic
import data.nat.factorial.basic

theorem x: ∀ (a b : ℕ), a + b = b + a :=
begin
sorry
end

theorem n_less_2_pow_n
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=
begin
  induction h₀ with k h₀ IH,
  { norm_num },
  {
    calc k + 1 < 2 * 2^k : by linarith
  }
end

theorem wrong_proof1: ∀ (a b : ℕ), a + 0 = a :=
begin
sorry
end

theorem wrong_proof2: ∀ (a b : ℕ), a + 0 = a :=
begin
sorry
end

theorem mathd_algebra_478
  (b h v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2) :
  v = 65 :=
begin
  rw [h₂, h₃] at h₁,
  rw h₁,
  norm_num
end

theorem ab_square: ∀ a: ℝ, a + 0 = a :=
begin
    intro a,
    induction a,
    simp,
end

-- variables (w x y z : ℕ) (p : ℕ → Prop)

-- local attribute [simp] mul_comm mul_assoc mul_left_comm
-- local attribute [simp] add_assoc add_comm add_left_comm

-- example (h : p (x * y + z * w  * x)) : p (x * w * z + y * x) :=
-- by { simp at *, assumption }

-- example (h₁ : p (1 * x + y)) (h₂ : p  (x * z * 1)) :
--   p (y + 0 + x) ∧ p (z * x) :=
-- by { simp at *, split; assumption }