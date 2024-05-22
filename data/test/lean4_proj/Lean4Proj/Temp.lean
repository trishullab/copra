import Mathlib
open BigOperators

def hello := "world"

theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    exact ⟨hp, hq, hp⟩

theorem sq_ab
: ∀ a, ∀ b, a^2 + b^2 + 2*a*b = (a + b)^2 := by
    sorry
