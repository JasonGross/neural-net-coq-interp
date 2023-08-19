From Coq Require Import ZifyUint63 Zify PreOmega Lia Lra Lqa Reals List Floats PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From Flocq.Core Require Import Raux.
From NeuralNetInterp.Util.Arith Require Import Instances Classes Classes.Laws FloatArith.Definitions QArith ZArith Reals.Definitions.
Import ListNotations.
Set Implicit Arguments.

Local Open Scope core_scope.

Local Open Scope bool_scope.
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

Local Open Scope nat_scope.
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

Local Open Scope N_scope.
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

Local Open Scope positive_scope.
#[export] Instance positive_mul_1_l : LeftId (@eq positive) mul 1 := Pos.mul_1_l.
#[export] Instance positive_mul_1_r : RightId (@eq positive) mul 1 := Pos.mul_1_r.
#[export] Instance positive_add_assoc : Associative (@eq positive) add := Pos.add_assoc.
#[export] Instance positive_mul_assoc : Associative (@eq positive) mul := Pos.mul_assoc.
#[export] Instance positive_add_comm : Commutative (@eq positive) add := Pos.add_comm.
#[export] Instance positive_mul_comm : Commutative (@eq positive) mul := Pos.mul_comm.
#[export] Instance positive_mul_add_distr_l : LeftDistributive (@eq positive) mul add := Pos.mul_add_distr_l.
#[export] Instance positive_mul_add_distr_r : RightDistributive (@eq positive) mul add := Pos.mul_add_distr_r.

Local Open Scope Z_scope.
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

Local Open Scope Q_scope.
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

Local Open Scope int63_scope.
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

#[local] Open Scope R_scope.
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
