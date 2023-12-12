`conv start`

`example_user`
[THEOREM]
For all natural numbers $n$, if $1 \le n$, then $n < 2^n$.
[END]

`example_assistant`
[PROOF]
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

`conv end`