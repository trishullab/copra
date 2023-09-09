import data.real.basic

theorem ab_square: ∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0 :=
begin
intros a b h,
cases lt_trichotomy a 0 with ha ha,
cases lt_trichotomy b 0 with hb hb,
have h1 : a * b < 0,

end