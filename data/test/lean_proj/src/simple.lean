import data.real.basic
import data.nat.factorial.basic

theorem a_plus_b_b_plus_a 
(a b : ℕ) : a + b = b + a :=
begin
  sorry
end

theorem mod_arith_1
(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
  sorry
end

theorem n_less_2_pow_n
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=
begin
  sorry
end

theorem a_plus_zero: ∀ (a : ℕ), a + 0 = a :=
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
  sorry
end

theorem ab_square: 
∀ (a b: ℝ), (a + b)^2 = a^2 + b^2 + 2*a*b :=
begin
  sorry
end
