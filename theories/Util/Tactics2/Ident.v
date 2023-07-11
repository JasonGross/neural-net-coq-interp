From Ltac2 Require Import Ltac2 Ident String.

Ltac2 compare (x : ident) (y : ident) : int
  := String.compare (Ident.to_string x) (Ident.to_string y).
