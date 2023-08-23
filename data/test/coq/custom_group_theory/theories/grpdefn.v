Inductive algb : Set := 
e : algb
| t : algb.

Definition algb_eq (x y : algb) : bool :=
match x, y with
| e, e => true
| t, t => true
| _, _ => false
end.

Definition algb_add (a b : algb) : algb :=
match a, b with
| e, _ => b
| _, e => a
| t, t => e
end.

Definition algb_mul (a b : algb) : algb :=
match a, b with
| e, _ => e
| _, e => e
| t, t => t
end.

Fixpoint algb_mul_scalar (b : nat) (a : algb) : algb :=
match b, a with
| 0, _ => e
| S b', _ => algb_add a (algb_mul_scalar b' a)
end.