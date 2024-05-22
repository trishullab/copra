namespace Lean4Proj1

def hello := "world"

theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp


theorem test2 : p -> q -> p ∧ q ∧ p := fun hp hq => ⟨hp, ⟨hq, hp⟩⟩

end Lean4Proj1

namespace Lean4Proj2

theorem test (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp

theorem test3 (p q : Prop) (hp : p) (hq : q)
: p ∧ q ∧ p := by
    apply And.intro
    exact hp
    apply And.intro
    exact hq
    exact hp

end Lean4Proj2
