From Coq Require Extraction.
Require Import Bool Arith List.
(* Set Implicit Arguments.
Set Asymmetric Patterns. *)

(*Define the source language first*)
Inductive binop : Set := Plus | Times.

Inductive exp : Set := 
 Const : nat -> exp
| Binop : binop -> exp -> exp -> exp. (* Binop is a function which takes exp and exp and gives exp. This is just currying*)

Definition binopDenote (b : binop) : nat -> nat -> nat :=
match b with 
    | Plus => plus
    | Times => mult
end.

Fixpoint expDenote (e: exp) : nat :=
    match e with
    | Const n => n
    | Binop b e1 e2 => (binopDenote b) (expDenote e1) (expDenote e2)
    end.

Eval simpl in expDenote (Const 42).
Eval simpl in expDenote (Binop Plus (Const 2) (Const 2)).
Eval simpl in expDenote (Binop Times (Binop Plus (Const 2) (Const 2)) (Const 7)).

(*Define the target language*)

Inductive instr: Set :=
| iConst : nat -> instr
| iBinop : binop -> instr. (*Instructions can be either constants or binary operation*)

Definition prog := list instr. (*Program is a list of instructions*)
Definition stack := list nat. (*Instruction either pushes a constant to the stack or applies binop on two elements on the stack*)

Definition instrDenote (i : instr) (s: stack): option stack :=
match i with
 | iConst n => Some (n :: s)
 | iBinop b =>
    match s with
     | arg1 :: arg2 :: s' => Some ((binopDenote b) arg1 arg2 :: s')
     | _ => None
    end    
end.

Fixpoint progDenote (p : prog) (s: stack) : option stack :=
match p with
 | nil => Some s
 | i::p' =>
   match instrDenote i s with
    | None => None
    | Some s' => progDenote p' s'
   end
end. (*Run instructions one by one*)

(*Check the execution of program once*)
Eval simpl in progDenote (iConst 32:: iConst 42 :: iBinop Plus :: nil) nil.

(*Translation for the language i.e. Compiler*)
Fixpoint compile (e : exp): prog := 
    match e with
       | Const n => iConst n::nil
       | Binop b e1 e2 => compile e2 ++ compile e1 ++ iBinop b :: nil
    end.

Eval simpl in compile (Binop Plus (Const 2) (Const 3)).

(*Check the correctness of the compiler itself*)
Lemma compile_one_instr: forall e p s, progDenote (compile e ++ p) s = progDenote p (expDenote e :: s).
    induction e.
    intros.
    unfold compile.
    unfold expDenote.
    unfold progDenote at 1.
    simpl.
    fold progDenote.
    reflexivity.
    intros.
    simpl.
    rewrite app_assoc_reverse.
    rewrite IHe2.
    rewrite app_assoc_reverse.
    rewrite IHe1.
    simpl.
    reflexivity.
Qed.

Theorem compile_correct: forall e, progDenote (compile e) nil = Some (expDenote e::nil).
    intros.
    rewrite (app_nil_end (compile e)).
    rewrite compile_one_instr.
    simpl.
    reflexivity.
Qed.


Lemma bin_op_comm: forall b e1 e2, expDenote (Binop b e1 e2) = expDenote (Binop b e2 e1).
Proof.
    intros.
    induction b.
    - simpl.
      apply plus_comm.
    - simpl.
      apply mult_comm.
Qed.

Lemma reverse_merge: forall e1 e2 b, compile e2 ++ compile e1 ++ iBinop b::nil = compile (Binop b e1 e2).
Proof.
    intros.
    simpl.
    reflexivity.
Qed.

Theorem compile_op_comm: forall b e1 e2, progDenote (compile e2 ++ compile e1 ++ iBinop b::nil) nil = progDenote (compile e1 ++ compile e2 ++ iBinop b::nil) nil.
Proof.
    intros.
    rewrite reverse_merge.
    rewrite reverse_merge.
    rewrite compile_correct.
    rewrite compile_correct.
    rewrite bin_op_comm.
    reflexivity.
Qed.

Theorem const_eq: forall n n', Const n = Const n' -> n = n'.
Proof.
    intros.
    inversion H.
    reflexivity.
Qed.

Theorem const_ins_eq: forall n n', iConst n = iConst n' -> n = n'.
Proof.
    intros.
    inversion H.
    reflexivity.
Qed.

Theorem const_only_const: forall e n, Const n = e -> e = Const n.
Proof.
    intros.
    inversion H.
    reflexivity.
Qed.

Theorem list_eq: forall  a1 a2 : Type, a1::nil = a2::nil -> a1 = a2.
Proof.
    intros.
    inversion H.
    reflexivity.
Qed.


Lemma const_cmpl: forall n b e1 e2, compile e2 ++ compile e1 ++ iBinop b :: nil <> iConst n :: nil.
Proof.
    intros.
    unfold not.
    intros.
    rewrite app_assoc in H.
    remember (compile e2 ++ compile e1) as lst1.
    remember (iBinop b :: nil) as lst2.
    remember (iConst n) as A.
    destruct lst1.
    - subst. discriminate.
    - subst. inversion H. destruct lst1. discriminate H2. discriminate.
Qed.

(*Code generates a Haskell file which can be compiled and run. The output is as follows*)

Extraction Language Haskell.
Extraction "data/test/coq/stack_machine/StackMachine.hs" compile.