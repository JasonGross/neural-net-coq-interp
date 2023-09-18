From Coq Require Import Lra Reals.
From Flocq.Core Require Import Raux.
From NeuralNetInterp.Util.Arith Require Import Reals.Definitions.
#[local] Open Scope R_scope.

Lemma Rlt_bool_iff x y : Rlt_bool x y = true <-> Rlt x y.
Proof. destruct (Rlt_bool_spec x y); intuition; try congruence; lra. Qed.

Lemma Rle_bool_iff x y : Rle_bool x y = true <-> Rle x y.
Proof. destruct (Rle_bool_spec x y); intuition; try congruence; lra. Qed.

Lemma Req_bool_iff x y : Req_bool x y = true <-> x = y.
Proof. destruct (Req_bool_spec x y); intuition; try congruence; lra. Qed.
