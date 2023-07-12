From Coq Require Import ZArith.
Local Open Scope Z_scope.

Module Z.
  Definition pow_N (b : Z) (e : N) : Z
    := match e with
       | N0 => 1
       | Npos p => Z.pow_pos b p
       end.

  Definition log2_round (x : Z) : Z
    := let '(a, b) := (Z.log2 x, Z.log2_up x) in
       let '(a2, b2) := (2^a, 2^b) in
       let '(aerr, berr) := (Z.abs (x - a2), Z.abs (x - b2)) in
       if aerr <=? berr
       then a
       else b.
End Z.
