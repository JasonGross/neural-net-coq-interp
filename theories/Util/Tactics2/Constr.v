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

Ltac2 is_float(c: constr) :=
  match Unsafe.kind c with
  | Unsafe.Float _ => true
  | _ => false
  end.

Ltac2 is_uint63(c: constr) :=
  match Unsafe.kind c with
  | Unsafe.Uint63 _ => true
  | _ => false
  end.

Ltac2 is_array(c: constr) :=
  match Unsafe.kind c with
  | Unsafe.Array _ _ _ _ => true
  | _ => false
  end.
