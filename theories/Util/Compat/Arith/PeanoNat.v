From Coq Require Import Arith.

Module Nat.
  Export PeanoNat.Nat.

  (** Missing from Coq <= 8.17 *)
  Lemma iter_succ :
    forall n (A:Type) (f:A -> A) (x:A),
      iter (S n) f x = f (iter n f x).
  Proof.
    reflexivity.
  Qed.
End Nat.
