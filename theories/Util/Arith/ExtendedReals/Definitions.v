From Coq Require Import Reals.

Inductive XR : Set :=
| Xr (_ : R)
| Xneg_zero
| Xnan
| Xinfinity (is_neg : bool)
.

#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion Xr : R >-> XR.
#[local] Set Warnings Append "unsupported-attributes".
(*
Module XR.
  Search sumbool R.
  Search (R -> R -> bool).
  Definition opp (x : XR) : XR
    := match x with
       | Xnan => Xnan
       | Xr r => if Req_dec_T r 0
                 then Xneg_zero
                 else
         => Xr (Ropp r)
       | Xinfinity is_neg => Xinfinity (negb is_neg)
       |
    := lift1 negb Ropp.
  Definition R
  Print RinvImpl.Rinv.

  Definition add (x y : XR) : XR
    := match x, y with
       | Xnan, _ | _, Xnan => Xnan
       | Xr x, Xr y => Xr (Rplus x y)
       | Xinfinity xneg, Xinfinity yneg
         => if Bool.eqb xneg yneg
            then Xinfinity xneg
            else Xnan
       | Xinfinity is_neg, Xr _
       | Xr _, Xinfinity is_neg
         => Xinfinity is_neg
       end.

  Definition sub (x y : XR) : XR := add x (opp y).

  De

Module Export ExtendedRealsNotations.
  Declare Scope extended_real_scope.
  Delimit Scope extended_real_scope with XR.
  Bind Scope extended_real_scope with XR.
  Notation "∞" := (Xinfinity false) : extended_real_scope.
  Notation "-∞" := (Xinfinity true) : extended_real_scope.
End ExtendedRealsNotations.
*)
