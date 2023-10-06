import data.real.basic
import data.nat.factorial.basic

theorem mathd_algebra_478
(b h v : ℝ)
(h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
(h₁ : v = 1 / 3 * (b * h))
(h₂ : b = 30)
(h₃ : h = 13 / 2) :
v = 65 :=
begin
rw [h₃, h₂] at h₁,
end