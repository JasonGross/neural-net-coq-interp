From Coq Require Import Reals List Floats PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From Flocq.Core Require Import Raux.
From NeuralNetInterp.Util.Arith Require Import Classes Instances Reals.Definitions.
Import ListNotations.
Set Implicit Arguments.
#[global] Set Warnings Append "-ambiguous-paths".
#[export] Set Warnings Append "-ambiguous-paths".

Local Open Scope core_scope.

#[local] Open Scope R_scope.
#[export] Instance R_has_leb : has_leb R := Rle_bool.
#[export] Instance R_has_ltb : has_ltb R := Rlt_bool.
#[export] Instance R_has_eqb : has_ltb R := Req_bool.
