import data.real.basic
import data.nat.factorial.basic
import data.nat.prime

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


 theorem mod_arith_2
(x : ℕ) : x % 2 = 0 → (x * x) % 2 = 0 :=
begin
    intro h,
    rw nat.mul_mod,
    rw h,
    rw nat.zero_mul,
    refl,
end