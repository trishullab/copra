(*
 * Author: Max Fan
 *)

 Require Import ZArith Lia.

 (* Infix "/" := Zdiv. *)
 Infix "+" := Zplus.
 Infix "*" := Zmult.
 Infix "-" := Zminus.
 
 Theorem numbertheory_4x3m7y3neq2003 :
   forall (x y : Z),
   (4%Z * x^(3%Z) - 7%Z * y^(3%Z)) <> 2003%Z.
 Proof.
   intros.
   give_up.
 Admitted.