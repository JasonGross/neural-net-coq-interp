From Coq Require Import QMicromega ZifyUint63 Zify PreOmega Lia Lra Lqa Reals List Floats PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From Flocq.Core Require Import Raux.
From NeuralNetInterp.Util.Arith Require Import Instances Classes Classes.Laws FloatArith.Definitions QArith ZArith Reals.Definitions.
Import ListNotations.
Set Implicit Arguments.

#[local] Coercion is_true : bool >-> Sortclass.
Local Open Scope core_scope.

#[export] Instance bool_add_0_l : LeftId (@eq bool) add 0 := Bool.orb_false_l.
#[export] Instance bool_add_0_r : RightId (@eq bool) add 0 := Bool.orb_false_r.
#[export] Instance bool_mul_1_l : LeftId (@eq bool) mul 1 := Bool.andb_true_l.
#[export] Instance bool_mul_1_r : RightId (@eq bool) mul 1 := Bool.andb_true_r.
#[export] Instance bool_mul_0_l : LeftZero (@eq bool) mul 0 := Bool.andb_false_l.
#[export] Instance bool_mul_0_r : RightZero (@eq bool) mul 0 := Bool.andb_false_r.
#[export] Instance bool_add_assoc : Associative (@eq bool) add := Bool.orb_assoc.
#[export] Instance bool_mul_assoc : Associative (@eq bool) mul := Bool.andb_assoc.
#[export] Instance bool_add_comm : Commutative (@eq bool) add := Bool.orb_comm.
#[export] Instance bool_mul_comm : Commutative (@eq bool) mul := Bool.andb_comm.
#[export] Instance bool_mul_add_distr_l : LeftDistributive (@eq bool) mul add := Bool.andb_orb_distrib_r.
#[export] Instance bool_mul_add_distr_r : RightDistributive (@eq bool) mul add := Bool.andb_orb_distrib_l.
#[export] Instance bool_eqb_Reflexive : Reflexive (eqb (A:=bool)) | 10 := Bool.eqb_reflx.
#[export] Instance bool_eqb_Symmetric : Symmetric (eqb (A:=bool)) | 10.
Proof. repeat intros []; reflexivity. Qed.
#[export] Instance bool_eqb_Transitive : Transitive (eqb (A:=bool)) | 10.
Proof. cbv; do 3 intros []; congruence. Qed.

#[export] Instance nat_add_0_l : LeftId (@eq nat) add 0 := Nat.add_0_l.
#[export] Instance nat_add_0_r : RightId (@eq nat) add 0 := Nat.add_0_r.
#[export] Instance nat_mul_1_l : LeftId (@eq nat) mul 1 := Nat.mul_1_l.
#[export] Instance nat_mul_1_r : RightId (@eq nat) mul 1 := Nat.mul_1_r.
#[export] Instance nat_mul_0_l : LeftZero (@eq nat) mul 0 := Nat.mul_0_l.
#[export] Instance nat_mul_0_r : RightZero (@eq nat) mul 0 := Nat.mul_0_r.
#[export] Instance nat_add_assoc : Associative (@eq nat) add := Nat.add_assoc.
#[export] Instance nat_mul_assoc : Associative (@eq nat) mul := Nat.mul_assoc.
#[export] Instance nat_add_comm : Commutative (@eq nat) add := Nat.add_comm.
#[export] Instance nat_mul_comm : Commutative (@eq nat) mul := Nat.mul_comm.
#[export] Instance nat_mul_add_distr_l : LeftDistributive (@eq nat) mul add := Nat.mul_add_distr_l.
#[export] Instance nat_mul_add_distr_r : RightDistributive (@eq nat) mul add := Nat.mul_add_distr_r.
#[export] Instance nat_eqb_Reflexive : Reflexive (eqb (A:=nat)) | 10 := Nat.eqb_refl.
#[export] Instance nat_eqb_Symmetric : Symmetric (eqb (A:=nat)) | 10.
Proof. cbv [eqb nat_has_eqb]; intros ??; now rewrite Nat.eqb_sym. Qed.
#[export] Instance nat_eqb_Transitive : Transitive (eqb (A:=nat)) | 10.
Proof. cbv [eqb nat_has_eqb is_true]; intros ???; lia. Qed.
#[export] Instance nat_leb_Reflexive : Reflexive (leb (A:=nat)) | 10 := Nat.leb_refl.
#[export] Instance nat_leb_Transitive : Transitive (leb (A:=nat)) | 10.
Proof. cbv [leb nat_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance nat_leb_Antisymmetric : Antisymmetric nat eq leb | 10.
Proof. cbv [leb nat_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance nat_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=nat) x y)) | 10.
Proof. cbv [leb nat_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance nat_ltb_Transitive : Transitive (ltb (A:=nat)) | 10.
Proof. cbv [ltb nat_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance nat_ltb_Irreflexive : Irreflexive (ltb (A:=nat)) | 10.
Proof. cbv [ltb nat_has_ltb is_true]; intros ??; lia. Qed.
#[export] Instance nat_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=nat) x y)) | 10.
Proof. cbv [ltb nat_has_ltb is_true]; intros ?; lia. Qed.
#[export] Instance nat_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=nat) x y)) | 10.
Proof. cbv [ltb nat_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance nat_negb_ltb_Antisymmetric : Antisymmetric nat eq (fun x y => negb (ltb (A:=nat) x y)) | 10.
Proof. cbv [ltb nat_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance nat_ltb_Asymmetric : Asymmetric (ltb (A:=nat)) | 10.
Proof. cbv [ltb nat_has_ltb is_true]; intros ???; lia. Qed.

#[export] Instance N_add_0_l : LeftId (@eq N) add 0 := N.add_0_l.
#[export] Instance N_add_0_r : RightId (@eq N) add 0 := N.add_0_r.
#[export] Instance N_mul_1_l : LeftId (@eq N) mul 1 := N.mul_1_l.
#[export] Instance N_mul_1_r : RightId (@eq N) mul 1 := N.mul_1_r.
#[export] Instance N_mul_0_l : LeftZero (@eq N) mul 0 := N.mul_0_l.
#[export] Instance N_mul_0_r : RightZero (@eq N) mul 0 := N.mul_0_r.
#[export] Instance N_add_assoc : Associative (@eq N) add := N.add_assoc.
#[export] Instance N_mul_assoc : Associative (@eq N) mul := N.mul_assoc.
#[export] Instance N_add_comm : Commutative (@eq N) add := N.add_comm.
#[export] Instance N_mul_comm : Commutative (@eq N) mul := N.mul_comm.
#[export] Instance N_mul_add_distr_l : LeftDistributive (@eq N) mul add := N.mul_add_distr_l.
#[export] Instance N_mul_add_distr_r : RightDistributive (@eq N) mul add := N.mul_add_distr_r.
#[export] Instance N_eqb_Reflexive : Reflexive (eqb (A:=N)) | 10 := N.eqb_refl.
#[export] Instance N_eqb_Symmetric : Symmetric (eqb (A:=N)) | 10.
Proof. cbv [eqb N_has_eqb]; intros ??; now rewrite N.eqb_sym. Qed.
#[export] Instance N_eqb_Transitive : Transitive (eqb (A:=N)) | 10.
Proof. cbv [eqb N_has_eqb is_true]; intros ???; lia. Qed.
#[export] Instance N_leb_Reflexive : Reflexive (leb (A:=N)) | 10 := N.leb_refl.
#[export] Instance N_leb_Transitive : Transitive (leb (A:=N)) | 10.
Proof. cbv [leb N_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance N_leb_Antisymmetric : Antisymmetric N eq leb | 10.
Proof. cbv [leb N_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance N_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=N) x y)) | 10.
Proof. cbv [leb N_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance N_ltb_Irreflexive : Irreflexive (ltb (A:=N)) | 10.
Proof. cbv [ltb N_has_ltb is_true]; intros ??; lia. Qed.
#[export] Instance N_ltb_Transitive : Transitive (ltb (A:=N)) | 10.
Proof. cbv [ltb N_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance N_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=N) x y)) | 10.
Proof. cbv [ltb N_has_ltb is_true]; intros ?; lia. Qed.
#[export] Instance N_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=N) x y)) | 10.
Proof. cbv [ltb N_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance N_negb_ltb_Antisymmetric : Antisymmetric N eq (fun x y => negb (ltb (A:=N) x y)) | 10.
Proof. cbv [ltb N_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance N_ltb_Asymmetric : Asymmetric (ltb (A:=N)) | 10.
Proof. cbv [ltb N_has_ltb is_true]; intros ???; lia. Qed.

#[export] Instance positive_mul_1_l : LeftId (@eq positive) mul 1 := Pos.mul_1_l.
#[export] Instance positive_mul_1_r : RightId (@eq positive) mul 1 := Pos.mul_1_r.
#[export] Instance positive_add_assoc : Associative (@eq positive) add := Pos.add_assoc.
#[export] Instance positive_mul_assoc : Associative (@eq positive) mul := Pos.mul_assoc.
#[export] Instance positive_add_comm : Commutative (@eq positive) add := Pos.add_comm.
#[export] Instance positive_mul_comm : Commutative (@eq positive) mul := Pos.mul_comm.
#[export] Instance positive_mul_add_distr_l : LeftDistributive (@eq positive) mul add := Pos.mul_add_distr_l.
#[export] Instance positive_mul_add_distr_r : RightDistributive (@eq positive) mul add := Pos.mul_add_distr_r.
#[export] Instance positive_eqb_Reflexive : Reflexive (eqb (A:=positive)) | 10 := Pos.eqb_refl.
#[export] Instance positive_eqb_Symmetric : Symmetric (eqb (A:=positive)) | 10.
Proof. cbv [eqb positive_has_eqb]; intros ??; now rewrite Pos.eqb_sym. Qed.
#[export] Instance positive_eqb_Transitive : Transitive (eqb (A:=positive)) | 10.
Proof. cbv [eqb positive_has_eqb is_true]; intros ???; lia. Qed.
#[export] Instance positive_leb_Reflexive : Reflexive (leb (A:=positive)) | 10 := Pos.leb_refl.
#[export] Instance positive_leb_Transitive : Transitive (leb (A:=positive)) | 10.
Proof. cbv [leb positive_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance positive_leb_Antisymmetric : Antisymmetric positive eq leb | 10.
Proof. cbv [leb positive_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance positive_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=positive) x y)) | 10.
Proof. cbv [leb positive_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance positive_ltb_Transitive : Transitive (ltb (A:=positive)) | 10.
Proof. cbv [ltb positive_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance positive_ltb_Irreflexive : Irreflexive (ltb (A:=positive)) | 10.
Proof. cbv [ltb positive_has_ltb is_true]; intros ??; lia. Qed.
#[export] Instance positive_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=positive) x y)) | 10.
Proof. cbv [ltb positive_has_ltb is_true]; intros ?; lia. Qed.
#[export] Instance positive_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=positive) x y)) | 10.
Proof. cbv [ltb positive_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance positive_negb_ltb_Antisymmetric : Antisymmetric positive eq (fun x y => negb (ltb (A:=positive) x y)) | 10.
Proof. cbv [ltb positive_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance positive_ltb_Asymmetric : Asymmetric (ltb (A:=positive)) | 10.
Proof. cbv [ltb positive_has_ltb is_true]; intros ???; lia. Qed.

#[export] Instance Z_add_0_l : LeftId (@eq Z) add 0 := Z.add_0_l.
#[export] Instance Z_add_0_r : RightId (@eq Z) add 0 := Z.add_0_r.
#[export] Instance Z_mul_1_l : LeftId (@eq Z) mul 1 := Z.mul_1_l.
#[export] Instance Z_mul_1_r : RightId (@eq Z) mul 1 := Z.mul_1_r.
#[export] Instance Z_mul_0_l : LeftZero (@eq Z) mul 0 := Z.mul_0_l.
#[export] Instance Z_mul_0_r : RightZero (@eq Z) mul 0 := Z.mul_0_r.
#[export] Instance Z_add_assoc : Associative (@eq Z) add := Z.add_assoc.
#[export] Instance Z_mul_assoc : Associative (@eq Z) mul := Z.mul_assoc.
#[export] Instance Z_add_comm : Commutative (@eq Z) add := Z.add_comm.
#[export] Instance Z_mul_comm : Commutative (@eq Z) mul := Z.mul_comm.
#[export] Instance Z_mul_add_distr_l : LeftDistributive (@eq Z) mul add := Z.mul_add_distr_l.
#[export] Instance Z_mul_add_distr_r : RightDistributive (@eq Z) mul add := Z.mul_add_distr_r.
#[export] Instance Z_eqb_Reflexive : Reflexive (eqb (A:=Z)) | 10 := Z.eqb_refl.
#[export] Instance Z_eqb_Symmetric : Symmetric (eqb (A:=Z)) | 10.
Proof. cbv [eqb Z_has_eqb]; intros ??; now rewrite Z.eqb_sym. Qed.
#[export] Instance Z_eqb_Transitive : Transitive (eqb (A:=Z)) | 10.
Proof. cbv [eqb Z_has_eqb is_true]; intros ???; lia. Qed.
#[export] Instance Z_leb_Reflexive : Reflexive (leb (A:=Z)) | 10 := Z.leb_refl.
#[export] Instance Z_leb_Transitive : Transitive (leb (A:=Z)) | 10.
Proof. cbv [leb Z_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance Z_leb_Antisymmetric : Antisymmetric Z eq leb | 10.
Proof. cbv [leb Z_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance Z_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=Z) x y)) | 10.
Proof. cbv [leb Z_has_leb is_true]; intros ???; lia. Qed.
#[export] Instance Z_ltb_Transitive : Transitive (ltb (A:=Z)) | 10.
Proof. cbv [ltb Z_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance Z_ltb_Irreflexive : Irreflexive (ltb (A:=Z)) | 10.
Proof. cbv [ltb Z_has_ltb is_true]; intros ??; lia. Qed.
#[export] Instance Z_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=Z) x y)) | 10.
Proof. cbv [ltb Z_has_ltb is_true]; intros ?; lia. Qed.
#[export] Instance Z_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=Z) x y)) | 10.
Proof. cbv [ltb Z_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance Z_negb_ltb_Antisymmetric : Antisymmetric Z eq (fun x y => negb (ltb (A:=Z) x y)) | 10.
Proof. cbv [ltb Z_has_ltb is_true]; intros ???; lia. Qed.
#[export] Instance Z_ltb_Asymmetric : Asymmetric (ltb (A:=Z)) | 10.
Proof. cbv [ltb Z_has_ltb is_true]; intros ???; lia. Qed.

#[export] Instance Q_add_0_l : LeftId Qeq add 0 := Qplus_0_l.
#[export] Instance Q_add_0_r : RightId Qeq add 0 := Qplus_0_r.
#[export] Instance Q_mul_1_l : LeftId Qeq mul 1 := Qmult_1_l.
#[export] Instance Q_mul_1_r : RightId Qeq mul 1 := Qmult_1_r.
#[export] Instance Q_mul_0_l : LeftZero Qeq mul 0 := Qmult_0_l.
#[export] Instance Q_mul_0_r : RightZero Qeq mul 0 := Qmult_0_r.
#[export] Instance Q_add_assoc : Associative Qeq add := Qplus_assoc.
#[export] Instance Q_mul_assoc : Associative Qeq mul := Qmult_assoc.
#[export] Instance Q_add_comm : Commutative Qeq add := Qplus_comm.
#[export] Instance Q_mul_comm : Commutative Qeq mul := Qmult_comm.
#[export] Instance Q_mul_add_distr_l : LeftDistributive Qeq mul add := Qmult_plus_distr_r.
#[export] Instance Q_mul_add_distr_r : RightDistributive Qeq mul add := Qmult_plus_distr_l.
#[export] Instance Q_eqb_Reflexive : Reflexive (eqb (A:=Q)) | 10 := Qeq_bool_refl.
#[export] Instance Q_eqb_Symmetric : Symmetric (eqb (A:=Q)) | 10.
Proof. cbv [eqb Q_has_eqb]; intros ???; now rewrite Qeq_bool_sym. Qed.
#[export] Instance Q_eqb_Transitive : Transitive (eqb (A:=Q)) | 10.
Proof. cbv [eqb Q_has_eqb is_true]; intros ???; rewrite !Qeq_bool_iff; intros; etransitivity; eassumption. Qed.
#[export] Instance Q_leb_Reflexive : Reflexive (leb (A:=Q)) | 10.
Proof. cbv [leb Q_has_leb is_true]; intros ?; rewrite !Qle_bool_iff; apply Qle_refl. Qed.
#[export] Instance Q_leb_Transitive : Transitive (leb (A:=Q)) | 10.
Proof. cbv [leb Q_has_leb is_true]; intros ???; rewrite !Qle_bool_iff; intros; eapply Qle_trans; eassumption. Qed.
#[export] Instance Q_leb_Antisymmetric : Antisymmetric Q Qeq leb | 10.
Proof. cbv [leb Q_has_leb is_true]; intros ??; rewrite !Qle_bool_iff; apply Qle_antisym. Qed.
#[export] Instance Q_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=Q) x y)) | 10.
Proof. cbv [leb Q_has_leb is_true]; intros ??; rewrite !negb_true_iff, <- !not_true_iff_false, !Qle_bool_iff; Lqa.lra. Qed.
#[export] Instance Q_ltb_Transitive : Transitive (ltb (A:=Q)) | 10.
Proof. cbv [ltb Q_has_ltb is_true]; intros ???; rewrite !Qlt_bool_iff; Lqa.lra. Qed.
#[export] Instance Q_ltb_Irreflexive : Irreflexive (ltb (A:=Q)) | 10.
Proof. cbv [ltb Q_has_ltb is_true Irreflexive complement]; intros ?; rewrite Qlt_bool_iff; Lqa.lra. Qed.
#[export] Instance Q_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=Q) x y)) | 10.
Proof. cbv [ltb Q_has_ltb is_true]; intros ?; rewrite !negb_true_iff, <- !not_true_iff_false, !Qlt_bool_iff; Lqa.lra. Qed.
#[export] Instance Q_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=Q) x y)) | 10.
Proof. cbv [ltb Q_has_ltb is_true]; intros ???; rewrite !negb_true_iff, <- !not_true_iff_false, !Qlt_bool_iff; Lqa.lra. Qed.
#[export] Instance Q_negb_ltb_Antisymmetric : Antisymmetric Q Qeq (fun x y => negb (ltb (A:=Q) x y)) | 10.
Proof. cbv [ltb Q_has_ltb is_true]; intros ??; rewrite !negb_true_iff, <- !not_true_iff_false, !Qlt_bool_iff; Lqa.lra. Qed.
#[export] Instance Q_ltb_Asymmetric : Asymmetric (ltb (A:=Q)) | 10.
Proof. cbv [ltb Q_has_ltb is_true]; intros ??; rewrite !Qlt_bool_iff; Lqa.lra. Qed.

Local Ltac zify_convert_to_euclidean_division_equations_flag ::= constr:(true).
#[export] Instance int_add_0_l : LeftId (@eq int) add 0 := ltac:(cbv; lia).
#[export] Instance int_add_0_r : RightId (@eq int) add 0 := ltac:(cbv; lia).
#[export] Instance int_mul_1_l : LeftId (@eq int) mul 1 := ltac:(cbv; lia).
#[export] Instance int_mul_1_r : RightId (@eq int) mul 1 := ltac:(cbv; lia).
#[export] Instance int_mul_0_l : LeftZero (@eq int) mul 0 := ltac:(cbv; lia).
#[export] Instance int_mul_0_r : RightZero (@eq int) mul 0 := ltac:(cbv; lia).
#[export] Instance int_add_assoc : Associative (@eq int) add := ltac:(cbv; lia).
#[export] Instance int_mul_assoc : Associative (@eq int) mul := ltac:(cbv; lia).
#[export] Instance int_add_comm : Commutative (@eq int) add := ltac:(cbv; lia).
#[export] Instance int_mul_comm : Commutative (@eq int) mul := ltac:(cbv; lia).
#[export] Instance int_mul_add_distr_l : LeftDistributive (@eq int) mul add := ltac:(cbv; lia).
#[export] Instance int_mul_add_distr_r : RightDistributive (@eq int) mul add := ltac:(cbv; lia).
#[export] Instance int_eqb_Reflexive : Reflexive (eqb (A:=int)) | 10.
Proof. cbv [eqb int_has_eqb is_true]; intros ?; lia. Qed.
#[export] Instance int_eqb_Symmetric : Symmetric (eqb (A:=int)) | 10.
Proof. cbv [eqb int_has_eqb is_true]; intros ??; lia. Qed.
#[export] Instance int_eqb_Transitive : Transitive (eqb (A:=int)) | 10.
Proof. cbv [eqb int_has_eqb is_true]; intros ???; lia. Qed.
Module Sint63.
  Import Instances.Sint63 Int63.Sint63 Arith.Classes.
  #[export] Instance int_leb_Reflexive : Reflexive (leb (A:=int)) | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ?; rewrite leb_spec; lia. Qed.
  #[export] Instance int_leb_Transitive : Transitive (leb (A:=int)) | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ???; rewrite !leb_spec; lia. Qed.
  (* Work around COQBUG(https://github.com/coq/coq/issues/17983) *)
  #[export] Instance int_leb_Antisymmetric : Antisymmetric int eq leb | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ??; rewrite !leb_spec; intros; apply Sint63.to_Z_inj; lia. Qed.
  #[export] Instance int_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=int) x y)) | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ??; rewrite !negb_true_iff, <- !not_true_iff_false, !leb_spec; lia. Qed.
  #[export] Instance int_ltb_Transitive : Transitive (ltb (A:=int)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ???; rewrite !ltb_spec; lia. Qed.
  #[export] Instance int_ltb_Irreflexive : Irreflexive (ltb (A:=int)) | 10.
  Proof. cbv [ltb int_has_ltb is_true Irreflexive complement]; intros ?; rewrite !ltb_spec; lia. Qed.
  #[export] Instance int_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=int) x y)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ?; rewrite !negb_true_iff, <- !not_true_iff_false, !ltb_spec; lia. Qed.
  #[export] Instance int_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=int) x y)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ???; rewrite !negb_true_iff, <- !not_true_iff_false, !ltb_spec; lia. Qed.
  #[export] Instance int_negb_ltb_Antisymmetric : Antisymmetric int eq (fun x y => negb (ltb (A:=int) x y)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ??; rewrite !negb_true_iff, <- !not_true_iff_false, !ltb_spec; intros; apply Sint63.to_Z_inj; lia. Qed.
  #[export] Instance int_ltb_Asymmetric : Asymmetric (ltb (A:=int)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ??; rewrite !ltb_spec; lia. Qed.
End Sint63.
Export (hints) Sint63.

Module Uint63.
  Import Instances.Uint63 ZifyUint63.
  #[export] Instance int_leb_Reflexive : Reflexive (leb (A:=int)) | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ?; lia. Qed.
  #[export] Instance int_leb_Transitive : Transitive (leb (A:=int)) | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ???; lia. Qed.
  #[export] Instance int_leb_Antisymmetric : Antisymmetric int eq leb | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ???; lia. Qed.
  #[export] Instance int_negb_leb_Asymmetric : Asymmetric (fun x y => negb (leb (A:=int) x y)) | 10.
  Proof. cbv [leb int_has_leb is_true]; intros ???; lia. Qed.
  #[export] Instance int_ltb_Transitive : Transitive (ltb (A:=int)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ???; lia. Qed.
  #[export] Instance int_ltb_Irreflexive : Irreflexive (ltb (A:=int)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ??; lia. Qed.
  #[export] Instance int_negb_ltb_Reflexive : Reflexive (fun x y => negb (ltb (A:=int) x y)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ?; lia. Qed.
  #[export] Instance int_negb_ltb_Transitive : Transitive (fun x y => negb (ltb (A:=int) x y)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ???; lia. Qed.
  #[export] Instance int_negb_ltb_Antisymmetric : Antisymmetric int eq (fun x y => negb (ltb (A:=int) x y)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ???; lia. Qed.
  #[export] Instance int_ltb_Asymmetric : Asymmetric (ltb (A:=int)) | 10.
  Proof. cbv [ltb int_has_ltb is_true]; intros ???; lia. Qed.
End Uint63.
Export (hints) Uint63.

#[export] Instance R_add_0_l : LeftId (@eq R) add 0 := Rplus_0_l.
#[export] Instance R_add_0_r : RightId (@eq R) add 0 := Rplus_0_r.
#[export] Instance R_mul_1_l : LeftId (@eq R) mul 1 := Rmult_1_l.
#[export] Instance R_mul_1_r : RightId (@eq R) mul 1 := Rmult_1_r.
#[export] Instance R_mul_0_l : LeftZero (@eq R) mul 0 := Rmult_0_l.
#[export] Instance R_mul_0_r : RightZero (@eq R) mul 0 := Rmult_0_r.
#[export] Instance R_add_assoc : Associative (@eq R) add := fun x y z => eq_sym (Rplus_assoc x y z).
#[export] Instance R_mul_assoc : Associative (@eq R) mul := fun x y z => eq_sym (Rmult_assoc x y z).
#[export] Instance R_add_comm : Commutative (@eq R) add := Rplus_comm.
#[export] Instance R_mul_comm : Commutative (@eq R) mul := Rmult_comm.
#[export] Instance R_mul_add_distr_l : LeftDistributive (@eq R) mul add := Rmult_plus_distr_l.
#[export] Instance R_mul_add_distr_r : RightDistributive (@eq R) mul add := Rmult_plus_distr_r.
