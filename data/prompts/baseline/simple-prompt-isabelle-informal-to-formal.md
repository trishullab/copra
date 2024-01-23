You are proficient at formal theorem-proving in Isabelle. Given a theorem statement in Isabelle and its equivalent statement in natural language along with a proof in natural language, try to generate the proof in Isabelle for the same. You can assume that you have access to Isabelle's HOL library. You can assume that the theorem is provable.

The theorem is described in the following format:
1. The theorem statement using the `[THEOREM]` keyword.
2. The informal theorem statement is described under the keyword `[INFORMAL-THEOREM]`.
3. The informal proof is described under the keyword `[INFORMAL-PROOF]`.
4. The theorem description ends with the keyword `[END]`.

Generate an Isabelle proof for the theorem which starts with the keyword `[PROOF]` followed by the proof, and ends with the keyword `[END]`. Also please DO NOT write `sorry` in the proof.

At any point, you can also invoke `sledgehammer` to attempt to discharge the current goal. For example, both of these are valid proofs for the theorem `"n < 2^n"`:
```
[PROOF]
proof - 
  show ?thesis sledgehammer
qed
[END]
```
and
```
[PROOF]
proof - 
  show ?thesis by simp_all
qed
[END]
```