From Coq Require Import Reals.
#[local] Open Scope R_scope.

Definition Rleb (x y : R) : bool := if Rle_dec x y then true else false.
Definition Rltb (x y : R) : bool := if Rlt_dec x y then true else false.
Definition Rgeb (x y : R) : bool := if Rge_dec x y then true else false.
Definition Rgtb (x y : R) : bool := if Rgt_dec x y then true else false.
Definition Reqb (x y : R) : bool := if Req_dec_T x y then true else false.

Infix "<=?" := Rleb : R_scope.
Infix "<?" := Rltb : R_scope.
Infix ">=?" := Rgeb : R_scope.
Infix ">?" := Rgtb : R_scope.
Infix "=?" := Reqb : R_scope.
