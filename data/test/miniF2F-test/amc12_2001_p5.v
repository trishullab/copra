(*
 * Author: Max Fan
 *)
 Require Import Nat Arith.

 Fixpoint odd_range_prod (n: nat) : nat :=
 match n with
 | O => 1
 | S p => match odd p with 
     | true => odd_range_prod p * p
     | false => odd_range_prod p
     end
 end.
 
 (* Check if the function is defined properly *)
 Lemma odd_range_prod_test_5: odd_range_prod 5 = 3.
 Proof.
 auto.
 Qed.
 
Theorem amc12_2001_p5: odd_range_prod 10000 = (fact 10000) / ((2^5000) * (fact 5000)).
 Proof.
   give_up.
 Admitted.