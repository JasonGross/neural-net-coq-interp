From Coq Require Import List Floats PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From NeuralNetInterp.Util.Arith Require Import Classes FloatArith.Definitions QArith ZArith.
Import ListNotations.
Set Implicit Arguments.
#[global] Set Warnings Append "-ambiguous-paths".
#[export] Set Warnings Append "-ambiguous-paths".

#[export] Instance default_pow_N {A} {oneA : has_one A} {mulA : has_mul A} : has_pow_by A N A
  := @pow_N A 1%core mul.

Local Open Scope core_scope.
#[export] Instance lift_coer_has_zero {A B} {coerAB : has_coer_to (tycons B tynil) A B} {zeroA : has_zero A} : has_zero B := @coer A B coerAB 0.
#[export] Instance lift_coer_has_one {A B} {coerAB : has_coer_to (tycons B tynil) A B} {oneA : has_one A} : has_one B := @coer A B coerAB 1.

Local Open Scope bool_scope.
#[export] Instance bool_has_eqb : has_eqb bool := Bool.eqb.
#[export] Instance bool_has_add : has_add bool := orb.
#[export] Instance bool_has_mul : has_mul bool := andb.
#[export] Instance bool_has_max : has_max bool := orb.
#[export] Instance bool_has_min : has_min bool := andb.
#[export] Instance bool_has_zero : has_zero bool := false.
#[export] Instance bool_has_one : has_one bool := true.

Local Open Scope nat_scope.
#[export] Instance nat_has_ltb : has_ltb nat := Nat.ltb.
#[export] Instance nat_has_leb : has_leb nat := Nat.leb.
#[export] Instance nat_has_eqb : has_eqb nat := Nat.eqb.
#[export] Instance nat_has_add : has_add nat := Nat.add.
#[export] Instance nat_has_sub : has_sub nat := Nat.sub.
#[export] Instance nat_has_mul : has_mul nat := Nat.mul.
#[export] Instance nat_has_int_div : has_int_div nat := Nat.div.
#[export] Instance nat_has_mod : has_mod nat := Nat.modulo.
#[export] Instance nat_has_max : has_max nat := Nat.max.
#[export] Instance nat_has_min : has_min nat := Nat.min.
#[export] Instance nat_has_pow : has_pow nat := Nat.pow.
#[export] Instance nat_has_abs : has_abs nat := fun x => x.
#[export] Instance nat_has_zero : has_zero nat := 0.
#[export] Instance nat_has_one : has_one nat := 1.
#[export] Instance nat_has_sqrt : has_sqrt nat := Nat.sqrt.

Local Open Scope N_scope.
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion N.of_nat : nat >-> N.
#[export] Coercion N.to_nat : N >-> nat.
#[local] Set Warnings Append "unsupported-attributes".
#[export] Instance N_of_nat_coer : has_coer nat N := fun x => x.
#[export] Instance N_to_nat_coer : has_coer N nat := fun x => x.
#[export] Hint Cut [ ( _ * ) N_of_nat_coer ( _ * ) N_to_nat_coer ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) N_to_nat_coer ( _ * ) N_of_nat_coer ( _ * ) ] : typeclass_instances.
#[export] Hint Extern 10 (has_coer_from nat ?B) => check_unify_has_coer_from N : typeclass_instances.
#[export] Hint Extern 10 (has_coer_from N ?B) => check_unify_has_coer_from nat : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A nat) => check_unify_has_coer_to N : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A N) => check_unify_has_coer_to nat : typeclass_instances.
#[export] Instance N_has_ltb : has_ltb N := N.ltb.
#[export] Instance N_has_leb : has_leb N := N.leb.
#[export] Instance N_has_eqb : has_eqb N := N.eqb.
#[export] Instance N_has_add : has_add N := N.add.
#[export] Instance N_has_sub : has_sub N := N.sub.
#[export] Instance N_has_mul : has_mul N := N.mul.
#[export] Instance N_has_int_div : has_int_div N := N.div.
#[export] Instance N_has_mod : has_mod N := N.modulo.
#[export] Instance N_has_max : has_max N := N.max.
#[export] Instance N_has_min : has_min N := N.min.
#[export] Instance N_has_pow : has_pow N := N.pow.
#[export] Instance N_has_abs : has_abs N := fun x => x.
#[export] Instance N_has_zero : has_zero N := 0.
#[export] Instance N_has_one : has_one N := 1.
#[export] Instance N_has_sqrt : has_sqrt N := N.sqrt.

#[export] Instance default_pow_nat {A} {oneA : has_one A} {mulA : has_mul A} : has_pow_by A nat A | 10
  := default_pow_N.

Local Open Scope positive_scope.
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion N.pos : positive >-> N.
#[local] Set Warnings Append "unsupported-attributes".
#[export] Instance N_pos_coer : has_coer positive N := fun x => x.
#[export] Hint Extern 10 (has_coer_from positive ?B) => check_unify_has_coer_from N : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A N) => check_unify_has_coer_to positive : typeclass_instances.
#[export] Instance positive_has_ltb : has_ltb positive := Pos.ltb.
#[export] Instance positive_has_leb : has_leb positive := Pos.leb.
#[export] Instance positive_has_eqb : has_eqb positive := Pos.eqb.
#[export] Instance positive_has_add : has_add positive := Pos.add.
#[export] Instance positive_has_sub : has_sub positive := Pos.sub.
#[export] Instance positive_has_mul : has_mul positive := Pos.mul.
#[export] Instance positive_has_max : has_max positive := Pos.max.
#[export] Instance positive_has_min : has_min positive := Pos.min.
#[export] Instance positive_has_pow : has_pow positive := Pos.pow.
#[export] Instance positive_has_abs : has_abs positive := fun x => x.
#[export] Instance positive_has_one : has_one positive := 1.
#[export] Instance positive_has_sqrt : has_sqrt positive := Pos.sqrt.

#[export] Instance default_pow_positive {A} {oneA : has_one A} {mulA : has_mul A} : has_pow_by A positive A | 10
  := default_pow_N.

Local Open Scope Z_scope.
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion Z.of_N : N >-> Z.
#[local] Set Warnings Append "unsupported-attributes".
#[export] Instance Z_of_N_coer : has_coer N Z := fun x => x.
#[export] Hint Extern 10 (has_coer_from N ?B) => check_unify_has_coer_from Z : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A Z) => check_unify_has_coer_to N : typeclass_instances.
#[export] Instance Z_has_ltb : has_ltb Z := Z.ltb.
#[export] Instance Z_has_leb : has_leb Z := Z.leb.
#[export] Instance Z_has_eqb : has_eqb Z := Z.eqb.
#[export] Instance Z_has_opp : has_opp Z := Z.opp.
#[export] Instance Z_has_add : has_add Z := Z.add.
#[export] Instance Z_has_sub : has_sub Z := Z.sub.
#[export] Instance Z_has_mul : has_mul Z := Z.mul.
#[export] Instance Z_has_int_div : has_int_div Z := Z.div.
#[export] Instance Z_has_mod : has_mod Z := Z.modulo.
#[export] Instance Z_has_max : has_max Z := Z.max.
#[export] Instance Z_has_min : has_min Z := Z.min.
#[export] Instance Z_has_pow : has_pow_by Z N Z := Z.pow_N.
#[export] Instance Z_has_zero : has_zero Z := 0.
#[export] Instance Z_has_one : has_one Z := 1.
#[export] Instance Z_has_sqrt : has_sqrt Z := Z.sqrt.

#[export] Instance default_pow_Z {A} {oneA : has_one A} {mulA : has_mul A} {divA : has_div A} : has_pow_by A Z A | 10
  := @pow_Z A oneA mulA divA.

Local Open Scope Q_scope.
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion inject_Z : Z >-> Q.
#[local] Set Warnings Append "unsupported-attributes".
#[export] Instance inject_Z_coer : has_coer Z Q := fun x => x.
#[export] Hint Extern 10 (has_coer_from Z ?B) => check_unify_has_coer_from Q : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A Q) => check_unify_has_coer_to Z : typeclass_instances.
#[export] Instance Q_has_leb : has_leb Q := Qle_bool.
#[export] Instance Q_has_eqb : has_eqb Q := Qeq_bool.
#[export] Instance Q_has_opp : has_opp Q := Qopp.
#[export] Instance Q_has_add : has_add Q := Qplus.
#[export] Instance Q_has_sub : has_sub Q := Qminus.
#[export] Instance Q_has_mul : has_mul Q := Qmult.
#[export] Instance Q_has_div : has_div Q := Qdiv.
#[export] Instance Q_has_zero : has_zero Q := 0.
#[export] Instance Q_has_one : has_one Q := 1.
#[export] Instance Q_has_sqrt : has_sqrt Q := Qsqrt.
#[export] Instance Q_has_pow_Z : has_pow_by Q Z Q := Qpower.
#[export] Instance Q_has_exp : has_exp Q := Qexp.

Local Open Scope int63_scope.
(* eta expand to get around COQBUG(https://github.com/coq/coq/issues/17663) *)
#[local] Notation eta1 f := (fun x => f x) (only parsing).
#[local] Notation eta2 f := (fun x y => f x y) (only parsing).
#[export] Instance int_has_eqb : has_eqb int := eta2 Uint63.eqb.
#[export] Instance int_has_opp : has_opp int := eta1 Uint63.opp.
#[export] Instance int_has_add : has_add int := eta2 Uint63.add.
#[export] Instance int_has_sub : has_sub int := eta2 Uint63.sub.
#[export] Instance int_has_mul : has_mul int := eta2 Uint63.mul.
#[export] Instance int_has_zero : has_zero int := 0.
#[export] Instance int_has_one : has_one int := 1.
#[export] Instance int_has_sqrt : has_sqrt int := eta1 Uint63.sqrt.

#[export] Hint Extern 10 (has_coer_from int ?B) => check_unify_has_coer_from Z : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A Z) => check_unify_has_coer_to int : typeclass_instances.

#[export] Hint Extern 10 (has_coer_from int ?B) => check_unify_has_coer_from float : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to ?A float) => check_unify_has_coer_to int : typeclass_instances.

Module Sint63.
  #[export] Instance coer_int_Z : has_coer int Z := eta1 Sint63.to_Z.
  #[export] Instance int_div : has_int_div int := eta2 Sint63.div.
  #[export] Instance modulo : has_mod int := eta2 Sint63.rem.
  #[export] Instance ltb : has_ltb int := eta2 Sint63.ltb.
  #[export] Instance leb : has_leb int := eta2 Sint63.leb.
  #[export] Instance min : has_min int := _.
  #[export] Instance max : has_max int := _.

  #[export] Instance coer_int_float : has_coer int float := PrimFloat.of_sint63.
  #[export] Set Warnings Append "-ambiguous-paths".
  #[local] Set Warnings Append "-unsupported-attributes".
  #[export] Coercion Sint63.to_Z : int >-> Z.
  #[export] Coercion PrimFloat.of_sint63 : int >-> float.
  #[local] Set Warnings Append "unsupported-attributes".
  #[export] Set Warnings Append "ambiguous-paths".
End Sint63.

Module Uint63.
  #[export] Instance coer_int_Z : has_coer int Z := eta1 Uint63.to_Z.
  #[export] Instance coer_int_N : has_coer int N := fun x => Z.to_N (Uint63.to_Z x).
  #[export] Instance int_div : has_int_div int := eta2 Uint63.div.
  #[export] Instance modulo : has_mod int := eta2 Uint63.mod.
  #[export] Instance ltb : has_ltb int := eta2 Uint63.ltb.
  #[export] Instance leb : has_leb int := eta2 Uint63.leb.
  #[export] Instance min : has_min int := _.
  #[export] Instance max : has_max int := _.

  #[export] Instance coer_int_float : has_coer int float := PrimFloat.of_uint63.
  #[local] Set Warnings Append "-unsupported-attributes".
  #[export] Set Warnings Append "-ambiguous-paths".
  #[export] Coercion Uint63.to_Z : int >-> Z.
  #[export] Coercion coer_int_N' (x : int) : N := Z.to_N (Uint63.to_Z x).
  #[export] Coercion PrimFloat.of_uint63 : int >-> float.
  #[local] Set Warnings Append "unsupported-attributes".
  #[export] Set Warnings Append "ambiguous-paths".
End Uint63.

Local Open Scope float_scope.
#[export] Instance float_has_leb : has_leb float := eta2 PrimFloat.leb.
#[export] Instance float_has_ltb : has_ltb float := eta2 PrimFloat.ltb.
#[export] Instance float_has_opp : has_opp float := eta1 PrimFloat.opp.
#[export] Instance float_has_abs : has_abs float := eta1 PrimFloat.abs.
#[export] Instance float_has_sqrt : has_sqrt float := eta1 PrimFloat.sqrt.
#[export] Instance float_has_add : has_add float := eta2 PrimFloat.add.
#[export] Instance float_has_sub : has_sub float := eta2 PrimFloat.sub.
#[export] Instance float_has_mul : has_mul float := eta2 PrimFloat.mul.
#[export] Instance float_has_div : has_div float := eta2 PrimFloat.div.
#[export] Instance float_has_zero : has_zero float := 0.
#[export] Instance float_has_one : has_one float := 1.
#[export] Instance float_has_exp : has_exp float := eta1 PrimFloat.exp.
#[export] Instance float_has_ln : has_ln float := eta1 PrimFloat.ln.

Module Float.
  Module IEEE754Eq.
    #[export] Instance float_has_eqb : has_eqb float := eta2 PrimFloat.eqb.
  End IEEE754Eq.

  Module Leibniz.
    #[export] Instance float_has_eqb : has_eqb float := eta2 PrimFloat.Leibniz.eqb.
  End Leibniz.
End Float.

Module Truncating.
  #[local] Set Warnings Append "-unsupported-attributes".
  #[export] Set Warnings Append "-ambiguous-paths".
  #[export] Coercion Uint63.of_Z : Z >-> Uint63.int.
  #[export] Set Warnings Append "ambiguous-paths".
  #[local] Set Warnings Append "unsupported-attributes".
  #[export] Instance coer_Z_int : has_coer Z int := Uint63.of_Z.
  #[export] Hint Extern 10 (has_coer_from Z ?B) => check_unify_has_coer_from int : typeclass_instances.
  #[export] Hint Extern 10 (has_coer_to ?A int) => check_unify_has_coer_to Z : typeclass_instances.

  #[local] Set Warnings Append "-unsupported-attributes".
  #[export] Set Warnings Append "-ambiguous-paths".
  #[export] Coercion PrimFloat.of_Z : Z >-> float.
  #[export] Set Warnings Append "ambiguous-paths".
  #[local] Set Warnings Append "unsupported-attributes".
  #[export] Instance coer_Z_float : has_coer Z float := PrimFloat.of_Z.
  #[export] Hint Extern 10 (has_coer_from Z ?B) => check_unify_has_coer_from float : typeclass_instances.
  #[export] Hint Extern 10 (has_coer_to ?A float) => check_unify_has_coer_to Z : typeclass_instances.
End Truncating.


Local Open Scope list_scope.
Scheme Equality for list.
#[export] Instance list_has_eqb {A} {Aeqb : has_eqb A} : has_eqb (list A)
  := list_beq Aeqb.
#[export] Instance list_has_zero {A} : has_zero (list A) := nil.

Module ListMonoid.
  #[export] Instance list_has_add {A} : has_add (list A) := @List.app _.
End ListMonoid.

#[export] Set Warnings Append "ambiguous-paths".
#[global] Set Warnings Append "ambiguous-paths".
