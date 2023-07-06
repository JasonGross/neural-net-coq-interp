From Coq Require Import List PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From NeuralNetInterp.Util Require Import Arith.Classes.
Import ListNotations.
Set Implicit Arguments.
#[global] Set Warnings Append "-ambiguous-paths".
#[export] Set Warnings Append "-ambiguous-paths".

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
#[export] Instance nat_has_abs : has_abs nat := fun x => x.
#[export] Instance nat_has_zero : has_zero nat := 0.
#[export] Instance nat_has_one : has_one nat := 1.

Local Open Scope N_scope.
#[global] Coercion N.of_nat : nat >-> N.
#[global] Coercion N.to_nat : N >-> nat.
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
#[export] Instance N_has_abs : has_abs N := fun x => x.
#[export] Instance N_has_zero : has_zero N := 0.
#[export] Instance N_has_one : has_one N := 1.

Local Open Scope positive_scope.
#[global] Coercion N.pos : positive >-> N.
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
#[export] Instance positive_has_abs : has_abs positive := fun x => x.
#[export] Instance positive_has_one : has_one positive := 1.

Local Open Scope Z_scope.
#[global] Coercion Z.of_N : N >-> Z.
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
#[export] Instance Z_has_zero : has_zero Z := 0.
#[export] Instance Z_has_one : has_one Z := 1.

Local Open Scope Q_scope.
#[global] Coercion inject_Z : Z >-> Q.
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

Local Open Scope int63_scope.
#[export] Instance int_has_eqb : has_eqb int := Uint63.eqb.
#[export] Instance int_has_opp : has_opp int := Uint63.opp.
#[export] Instance int_has_add : has_add int := Uint63.add.
#[export] Instance int_has_sub : has_sub int := Uint63.sub.
#[export] Instance int_has_mul : has_mul int := Uint63.mul.
#[export] Instance int_has_zero : has_zero int := 0.
#[export] Instance int_has_one : has_one int := 1.

Module Export Uint63.
  #[export] Instance int_has_int_div : has_int_div int := Uint63.div.
  #[export] Instance int_has_mod : has_mod int := Uint63.mod.
  #[export] Instance int_has_ltb : has_ltb int := Uint63.ltb.
  #[export] Instance int_has_leb : has_leb int := Uint63.leb.
End Uint63.

Module Sint63.
  #[export] Instance int_has_int_div : has_int_div int := Sint63.div.
  #[export] Instance int_has_mod : has_mod int := Sint63.rem.
  #[export] Instance int_has_ltb : has_ltb int := Sint63.ltb.
  #[export] Instance int_has_leb : has_leb int := Sint63.leb.
End Sint63.

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
