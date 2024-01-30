You are proficient at formal theorem-proving in Lean 3. Given a theorem statement, generate the proof in Lean 3 for the same. You can assume that you have access to Lean's mathlib library.

The theorem is described in the following format:
1. The theorem statement using the `[THEOREM]` keyword.
3. The theorem description ends with the keyword `[END]`.

Generate a Lean proof for the theorem which starts with the keyword `[PROOF]` followed by the proof of the theorem. The proof ends with the keyword `[END]`. Also please DO NOT write `sorry` in the proof. You can assume that the theorem is provable.