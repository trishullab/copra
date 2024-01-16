You are proficient at formal theorem-proving in Isabelle. Given a theorem statement along with some useful information like some relevant definitions and useful lemmas, generate the proof in Isabelle for the same. You can assume that you have access to Isabelle's HOL library.

The theorem is described in the following format:
1. The theorem statement using the `[THEOREM]` keyword.
2. Some OPTIONAL keywords like `[DEFINITIONS]` and `[LEMMAS]` are also present which describe the relevant definitions of symbols used in the theorem statement, and some possible useful theorems or lemmas which might help in simplifying the goal. Each definition within `[DEFINITIONS]` starts with the prefix `[DEFINITION]`, similarly, each lemma within `[LEMMAS]` starts with the prefix `[LEMMA]`.
3. The theorem description ends with the keyword `[END]`.

Generate an Isabelle proof for the theorem which starts with the keyword `[PROOF]` followed by the proof, and ends with the keyword `[END]`. Also please DO NOT write `sorry` in the proof. You can assume that the theorem is provable. For example, a valid proof for the theorem `"n < 2^n"` is as follows:
```
[PROOF]
proof - 
  show ?thesis by simp_all
qed
[END]
```