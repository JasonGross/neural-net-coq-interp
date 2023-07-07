From Coq Require Import QArith ZArith.

(* XXX FIXME *)
Definition Qsqrt (v : Q) : Q := Qred (Qmake (Z.sqrt (Qnum v * Zpos (Qden v))) (Qden v)).
