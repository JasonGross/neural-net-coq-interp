From Coq Require Import Arith.
From NeuralNetInterp.Util Require Import Notations.

Fixpoint radd (n m : nat) {struct m} : nat
  := match m with
     | 0 => n
     | S p => S (radd n p)
     end.
Fixpoint rsub (n m : nat) {struct m} : nat
  := match m, n with
     | 0, n => n
     | S p, S k => rsub k p
     | S p, 0 => 0
     end.

Module Export Notations.
  Infix "+'" := radd : nat_scope.
  Infix "-'" := rsub : nat_scope.
End Notations.
