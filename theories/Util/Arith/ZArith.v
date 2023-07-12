From Coq Require Import ZArith.
Local Open Scope Z_scope.

Module Z.
  Definition pow_N (b : Z) (e : N) : Z
    := match e with
       | N0 => 1
       | Npos p => Z.pow_pos b p
       end.
End Z.
