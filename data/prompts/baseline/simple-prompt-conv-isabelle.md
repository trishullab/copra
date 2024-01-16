`conv start`

`example_user`
[THEOREM] theorem n_less_2_pow_n: 
    shows "n < 2^n"
[END]

`example_assistant`
[PROOF]
proof - 
  show ?thesis by simp_all
qed
[END]

`conv end`