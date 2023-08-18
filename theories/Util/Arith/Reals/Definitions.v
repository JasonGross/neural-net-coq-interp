From Coq Require Import Reals.
From Flocq.Core Require Import Raux.
#[local] Open Scope R_scope.

Infix "<=?" := Rle_bool : R_scope.
Infix "<?" := Rlt_bool : R_scope.
Infix "=?" := Req_bool : R_scope.
