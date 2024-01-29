`conv start`

`example_user`
[THEOREM] some_theorem
  (n : ℕ)
  (h₀ : 1 ≤ n) :
  n < 2^n :=

[INFORMAL-THEOREM]
For all natural numbers $n$, if $1 \le n$, then $n < 2^n$.
[INFORMAL-PROOF]
We will prove this by induction on $n$.
First, we prove the base case $n = 1$.
We have $1 \le 1$, so $1 < 2^1$.
Now, suppose that $1 < n$.
By the inductive hypothesis, $n < 2^n$ \dots (1)
Combining these two inequalities, we have $1 < n < 2^n$.
So $1 < 2^n$ \dots (2)
By (1) and (2), we have $n + 1 < 2^n + 2^n$.
which can be rewritten as $n + 1 < 2^{n + 1}$.
This completes the proof.

[END]

`example_assistant`
[PROOF]
begin
  induction h₀ with k h₀ IH,
  { norm_num },
  {
    calc k + 1 < 2 * 2^k : by linarith
  }
end
[END]

`example_user`
[THEOREM] some_theorem
  (b : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = 3 * x^4 - 7 * x^3 + 2 * x^2 - b * x + 1)
  (h₁ : f 1 = 1) :
  b = -2 :=

[INFORMAL-THEOREM]
Let $f(x)=3x^4-7x^3+2x^2-bx+1$. For what value of $b$ is $f(1)=1$? Show that it is -2.
[INFORMAL-PROOF]
Evaluating, we get $f(1) = 3-7+2-b+1 = -b-1 = 1.$ Solving for $b,$ we find that $b = -2.$
[END]

`example_assistant`
[PROOF]
begin
  rw h₀ at h₁,
  linarith,
end
[END]

`example_user`
[THEOREM] some_theorem
  int.floor ((9:ℝ) / 160 * 100) = 5 :=

[INFORMAL-THEOREM]
What is the digit in the hundredths place of the decimal equivalent of $\\frac{9}{160}$? Show that it is 5.
[INFORMAL-PROOF]
Since the denominator of $\\dfrac{9}{160}$ is $2^5\\cdot5$, we multiply numerator and denominator by $5^4$ to obtain  \\[\n\\frac{9}{160} = \\frac{9\\cdot 5^4}{2^5\\cdot 5\\cdot 5^4} = \\frac{9\\cdot 625}{10^5} = \\frac{5625}{10^5} = 0.05625.\n\\]So, the digit in the hundredths place is $5$.
[END]

`example_assistant`
[PROOF]
begin
  rw int.floor_eq_iff,
  split,
  all_goals { norm_num },
end
[END]
`conv end`