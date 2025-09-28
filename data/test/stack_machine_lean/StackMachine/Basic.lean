import Mathlib
open BigOperators
namespace StackMachine

-- define a simple variable which allows
inductive Var where
| a | b | c | d | e

inductive Expr where
| num : Nat → Expr
| var : Var → Expr
| add : Expr → Expr → Expr

instance : Repr Var where
    reprPrec
    | Var.a, _ => "a"
    | Var.b, _ => "b"
    | Var.c, _ => "c"
    | Var.d, _ => "d"
    | Var.e, _ => "e"

-- check some valid expressions
#check Expr.add (Expr.num 1) (Expr.var Var.a)
#check Expr.add (Expr.add (Expr.num 1) (Expr.num 2)) (Expr.num 3)

-- Now we define a simple evaluator for our expressions
def eval (map : Var → Nat) (e : Expr) : Nat :=
  match e with
  | Expr.num n => n
  | Expr.var v => map v
  | Expr.add e1 e2 => eval map e1 + eval map e2

-- check that the evaluator works
#eval eval (λ v => match v with
  | Var.a => 1
  | Var.b => 2
  | Var.c => 3
  | Var.d => 4
  | Var.e => 5) (Expr.add (Expr.add (Expr.num 1) (Expr.num 2)) (Expr.num 3))

-- Now lets prove some simple theorems about our expressions

-- Additive associativity
theorem eval_add (map : Var → Nat) (e1 e2 : Expr) :
eval map (Expr.add e1 e2) = eval map e1 + eval map e2 :=
by simp [eval]

-- Additive commutativity
theorem eval_add_comm (map : Var → Nat) (e1 e2 : Expr) :
eval map (Expr.add e1 e2) = eval map (Expr.add e2 e1) :=
by
    simp [eval]
    rw [Nat.add_comm]

-- Two equal expressions evaluate to the same value
-- This is a simple congruence theorem
theorem eval_congr (map : Var → Nat) (e1 e2 : Expr) :
e1 = e2 → eval map e1 = eval map e2 :=
by
    intro h
    rw [h]

-- Now let us define an assembly language which converts the
-- expressions to a simple stack machine
inductive Instr where
| push : Nat → Instr
| add : Instr
| load : Var → Instr

-- Define a simple Program as a list of instructions
abbrev Program := List Instr

instance : Repr Instr where
  reprPrec
  | Instr.push n, _ => "push " ++ repr n
  | Instr.add, _ => "add"
  | Instr.load v, _ => "load " ++ repr v

instance : Repr Program where
  reprPrec
  | [], _ => "[]"
  | i::is, _ => repr i ++ ", " ++ repr is

inductive ProgResult where
| ok : Nat → ProgResult
| err : ProgResult

-- define representation for ProgResult
instance : Repr ProgResult where
  reprPrec
  | ProgResult.ok n, _ => "ok " ++ repr n
  | ProgResult.err, _ => "err"

-- Define a simple run function which runs the program
def run_instr (map : Var → Nat) (stack : List ProgResult) (i : Instr) : List ProgResult :=
  match i with
  | Instr.push n => ProgResult.ok n :: stack
  | Instr.load v => ProgResult.ok (map v) :: stack
  | Instr.add =>
    match stack with
    | (ProgResult.ok n1)::(ProgResult.ok n2)::ns => ProgResult.ok (n1 + n2) :: ns
    | _ => ProgResult.err :: stack

def run_program (map : Var → Nat) (stack : List ProgResult) (p : Program) : List ProgResult :=
    match p with
    | [] => stack
    | i::is => run_program map (run_instr map stack i) is



def run (map : Var → Nat) (p : Program) : ProgResult :=
    match run_program map [] p with
    | [ProgResult.ok n] => ProgResult.ok n
    | ProgResult.err :: _ => ProgResult.err
    | ProgResult.ok _ :: _ => ProgResult.err
    | [] => ProgResult.err


-- Now we can define a simple compiler which compiles the expressions to a program
def compile (e : Expr) : List Instr :=
    match e with
    | Expr.num n => [Instr.push n]
    | Expr.var v => [Instr.load v]
    | Expr.add e1 e2 => compile e1 ++ compile e2 ++ [Instr.add]

def compile_prog (e: Expr) : Program := compile e

-- Now print the compiled programs for some expressions
def expr1 := Expr.add (Expr.add (Expr.num 1) (Expr.num 2)) (Expr.num 3)
def map_expr1 (v : Var) : Nat :=
  match v with
  | Var.a => 1
  | Var.b => 2
  | Var.c => 3
  | Var.d => 4
  | Var.e => 5
#eval compile expr1
#eval run map_expr1 (compile expr1)
#eval run_program map_expr1 [ProgResult.ok 2] (compile (Expr.add expr1 (Expr.num 2)))
#eval run_instr map_expr1 [] (compile expr1)[0]
#eval eval map_expr1 expr1
#check run map_expr1 (compile expr1) = ProgResult.ok (eval map_expr1 expr1)


end StackMachine
