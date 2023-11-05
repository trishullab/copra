module StackMachine where

import qualified Prelude

data Nat =
   O
 | S Nat

data List a =
   Nil
 | Cons a (List a)

app :: (List a1) -> (List a1) -> List a1
app l m =
  case l of {
   Nil -> m;
   Cons a l1 -> Cons a (app l1 m)}

data Binop =
   Plus
 | Times

data Exp =
   Const Nat
 | Binop0 Binop Exp Exp

data Instr =
   IConst Nat
 | IBinop Binop

type Prog = List Instr

compile :: Exp -> Prog
compile e =
  case e of {
   Const n -> Cons (IConst n) Nil;
   Binop0 b e1 e2 ->
    app (compile e2) (app (compile e1) (Cons (IBinop b) Nil))}

