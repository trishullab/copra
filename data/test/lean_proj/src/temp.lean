import data.real.basic
import data.nat.factorial.basic

theorem wrong_proof1: ∀ (a b : ℕ), a + 0 = a :=
begin
intros,
{simp}
,