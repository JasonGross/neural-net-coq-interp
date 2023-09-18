From Coq Require Import RMicromega Zify PreOmega Lia Lra Reals List Floats PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From Flocq.Core Require Import Raux.
From NeuralNetInterp.Util.Arith Require Import Classes Instances Instances.Reals Reals.Definitions Reals.Proofs.

#[local] Open Scope R_scope.

#[local] Coercion is_true : bool >-> Sortclass.
#[local] Open Scope core_scope.

#[export] Instance R_eqb_Reflexive : Reflexive (eqb (A:=R)) | 10.
Proof. cbv [eqb R_has_eqb is_true]; intros ?; rewrite !Req_bool_iff; lra. Qed.
#[export] Instance R_eqb_Symmetric : Symmetric (eqb (A:=R)) | 10.
Proof. cbv [eqb R_has_eqb is_true]; intros ??; rewrite !Req_bool_iff; lra. Qed.
#[export] Instance R_eqb_Transitive : Transitive (eqb (A:=R)) | 10.
Proof. cbv [eqb R_has_eqb is_true]; intros ???; rewrite !Req_bool_iff; lra. Qed.
#[export] Instance R_leb_Reflexive : Reflexive (leb (A:=R)) | 10.
Proof. cbv [leb R_has_leb is_true]; intros ?; rewrite Rle_bool_iff; lra. Qed.
#[export] Instance R_leb_Transitive : Transitive (leb (A:=R)) | 10.
Proof. cbv [leb R_has_leb is_true]; intros ???; rewrite !Rle_bool_iff; lra. Qed.
#[export] Instance R_leb_Antisymmetric : Antisymmetric R eq leb | 10.
Proof. cbv [leb R_has_leb is_true]; intros ??; rewrite !Rle_bool_iff; lra. Qed.
#[export] Instance R_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=R) x y)) | 10.
Proof. cbv [leb R_has_leb is_true]; intros ??; rewrite !negb_true_iff, <- !not_true_iff_false, !Rle_bool_iff; lra. Qed.
#[export] Instance R_ltb_Transitive : Transitive (ltb (A:=R)) | 10.
Proof. cbv [ltb R_has_ltb is_true]; intros ???; rewrite !Rlt_bool_iff; lra. Qed.
#[export] Instance R_ltb_Irreflexive : Irreflexive (ltb (A:=R)) | 10.
Proof. cbv [ltb R_has_ltb is_true Irreflexive complement]; intros ?; rewrite !Rlt_bool_iff; lra. Qed.
#[export] Instance R_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=R) x y)) | 10.
Proof. cbv [ltb R_has_ltb is_true]; intros ?; rewrite !negb_true_iff, <- !not_true_iff_false, !Rlt_bool_iff; lra. Qed.
#[export] Instance R_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=R) x y)) | 10.
Proof. cbv [ltb R_has_ltb is_true]; intros ???; rewrite !negb_true_iff, <- !not_true_iff_false, !Rlt_bool_iff; lra. Qed.
#[export] Instance R_negb_ltb_Antisymmetric : Antisymmetric R eq (fun x y => negb (ltb (A:=R) x y)) | 10.
Proof. cbv [ltb R_has_ltb is_true]; intros ??; rewrite !negb_true_iff, <- !not_true_iff_false, !Rlt_bool_iff; lra. Qed.
#[export] Instance R_ltb_Asymmetric : Asymmetric (ltb (A:=R)) | 10.
Proof. cbv [ltb R_has_ltb is_true]; intros ??; rewrite !Rlt_bool_iff; lra. Qed.
