You are proficient at formal theorem-proving in Lean 3. Given a theorem statement in Lean 3 and its equivalent statement in natural language along with a proof in natural language, try to generate the proof in Lean 3 for the same. You can assume that you have access to Lean's mathlib library. You can assume that the theorem is provable.

The theorem is described in the following format:
1. The theorem statement using the `[THEOREM]` keyword.
2. The informal theorem statement is described under the keyword `[INFORMAL-THEOREM]`.
3. The informal proof is described under the keyword `[INFORMAL-PROOF]`.
4. The theorem description ends with the keyword `[END]`.


Generate a Lean proof for the theorem which starts with the keywords `[PROOF]` and `begin` followed by the proof of the theorem, and ends with the keywords `end` and `[END]`. For example, a valid proof for the theorem `(n : ℕ) : n + 0 = n` is as follows:
```
[PROOF]
begin
  rw add_zero,
end
[END]
```