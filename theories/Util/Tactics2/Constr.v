From Ltac2 Require Import Ltac2.
From Ltac2 Require Export Constr.

Module Unsafe.
  Export Ltac2.Constr.Unsafe.
  Ltac2 rec kind_nocast (c : constr)
    := let k := kind c in
       match k with
       | Cast c _ _ => kind_nocast c
       | _ => k
       end.
End Unsafe.
