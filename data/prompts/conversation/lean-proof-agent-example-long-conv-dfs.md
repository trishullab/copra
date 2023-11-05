`conv start`

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
x = 1 ∧ y = 1
[HYPOTHESES] 1
[HYPOTHESIS] x y : ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1

[END]


`example_assistant`
[RUN TACTIC]
linarith,
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
x = 1 ∧ y = 1
[HYPOTHESES] 1
[HYPOTHESIS] x y : ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1

[LAST STEP]
linarith,

[ERROR MESSAGE]
linarith failed to find a contradiction
state:
x y : ℝ,
h₁ : x = 3 - 2 * y,
h₂ : 2 * x - y = 1
⊢ false
[END]



`example_assistant`
[RUN TACTIC]
split,
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
x = 1
[HYPOTHESES] 1
[HYPOTHESIS] xy: ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1
[GOAL] 2
y = 1
[HYPOTHESES] 2
[HYPOTHESIS] xy: ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESES] h₂: 2 * x - y = 1

[LAST STEP]
split,
[SUCCESS]

[END]


`example_assistant`
[RUN TACTIC]
{
  rw h₁ at h₂,
  linarith
},
[END]


`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
y = 1
[HYPOTHESES] 1
[HYPOTHESIS] xy: ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1

[STEPS]
[STEP] split,

[LAST STEP]
{
  rw h₁ at h₂,
  linarith
},

[SUCCESS]

[END]

`example_assistant`
[RUN TACTIC]
rw h₂ at h₁,
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
y = 1
[HYPOTHESES] 1
[HYPOTHESIS] xy: ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1

[STEPS]
[STEP] split,
[STEP] {
  rw h₁ at h₂,
  linarith
},

[LAST STEP]
rw h₂ at h₁,

[ERROR MESSAGE]
rewrite tactic failed, did not find instance of the pattern in the target expression
  2 * x - y
state:
x y : ℝ,
h₁ : x = 3 - 2 * y,
h₂ : 2 * x - y = 1
⊢ y = 1
[END]

`example_assistant`
[RUN TACTIC]
rw ←h₁,
[END]

`example_user`
Goals to prove:
[GOALS]
[GOAL] 1
y = 1
[HYPOTHESES] 1
[HYPOTHESIS] xy: ℝ
[HYPOTHESIS] h₁: x = 3 - 2 * y
[HYPOTHESIS] h₂: 2 * x - y = 1

[STEPS]
[STEP] split,
[STEP] {
  rw h₁ at h₂,
  linarith
},

[INCORRECT STEPS]
[STEP] rw h₂ at h₁,

[LAST STEP]
rw ←h₁,

[ERROR MESSAGE]
rewrite tactic failed, did not find instance of the pattern in the target expression
  3 - 2 * y
[END]

`example_assistant`
[RUN TACTIC]
{
  rw h₁ at h₂,
  linarith
},
[END]

`conv end`