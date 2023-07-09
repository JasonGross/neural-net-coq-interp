From Coq Require Import Bool.

Definition where_ {A} (condition : bool) (input other : A) : A := if condition then input else other.
