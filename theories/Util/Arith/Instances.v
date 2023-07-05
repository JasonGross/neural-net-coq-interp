From Coq Require Import List PArray Sint63 Uint63 Arith PArith NArith ZArith QArith.
From NeuralNetInterp.Util Require Import Arith.Classes.
Import ListNotations.
Set Implicit Arguments.

Local Open Scope nat_scope.
#[export] Instance nat_has_ltb : has_ltb nat := Nat.ltb.
#[export] Instance nat_has_leb : has_leb nat := Nat.leb.
#[export] Instance nat_has_eqb : has_eqb nat := Nat.eqb.
#[export] Instance nat_has_add : has_add nat := Nat.add.
#[export] Instance nat_has_sub : has_sub nat := Nat.sub.
#[export] Instance nat_has_mul : has_mul nat := Nat.mul.
#[export] Instance nat_has_div : has_div nat := Nat.div.
#[export] Instance nat_has_mod : has_mod nat := Nat.modulo.
#[export] Instance nat_has_zero : has_zero nat := 0.
#[export] Instance nat_has_one : has_one nat := 1.

Local Open Scope N_scope.
#[export] Instance N_has_ltb : has_ltb N := N.ltb.
#[export] Instance N_has_leb : has_leb N := N.leb.
#[export] Instance N_has_eqb : has_eqb N := N.eqb.
#[export] Instance N_has_add : has_add N := N.add.
#[export] Instance N_has_sub : has_sub N := N.sub.
#[export] Instance N_has_mul : has_mul N := N.mul.
#[export] Instance N_has_div : has_div N := N.div.
#[export] Instance N_has_mod : has_mod N := N.modulo.
#[export] Instance N_has_zero : has_zero N := 0.
#[export] Instance N_has_one : has_one N := 1.

Local Open Scope positive_scope.
#[export] Instance positive_has_ltb : has_ltb positive := Pos.ltb.
#[export] Instance positive_has_leb : has_leb positive := Pos.leb.
#[export] Instance positive_has_eqb : has_eqb positive := Pos.eqb.
#[export] Instance positive_has_add : has_add positive := Pos.add.
#[export] Instance positive_has_sub : has_sub positive := Pos.sub.
#[export] Instance positive_has_mul : has_mul positive := Pos.mul.
#[export] Instance positive_has_one : has_one positive := 1.

Local Open Scope Z_scope.
#[export] Instance Z_has_ltb : has_ltb Z := Z.ltb.
#[export] Instance Z_has_leb : has_leb Z := Z.leb.
#[export] Instance Z_has_eqb : has_eqb Z := Z.eqb.
#[export] Instance Z_has_opp : has_opp Z := Z.opp.
#[export] Instance Z_has_add : has_add Z := Z.add.
#[export] Instance Z_has_sub : has_sub Z := Z.sub.
#[export] Instance Z_has_mul : has_mul Z := Z.mul.
#[export] Instance Z_has_div : has_div Z := Z.div.
#[export] Instance Z_has_mod : has_mod Z := Z.modulo.
#[export] Instance Z_has_zero : has_zero Z := 0.
#[export] Instance Z_has_one : has_one Z := 1.

Local Open Scope Q_scope.
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
  #[export] Instance int_has_div : has_div int := Uint63.div.
  #[export] Instance int_has_mod : has_mod int := Uint63.mod.
  #[export] Instance int_has_ltb : has_ltb int := Uint63.ltb.
  #[export] Instance int_has_leb : has_leb int := Uint63.leb.
End Uint63.

Module Sint63.
  #[export] Instance int_has_div : has_div int := Sint63.div.
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
