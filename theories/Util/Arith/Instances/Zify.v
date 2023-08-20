From Coq Require Import ZArith Zify ZifyClasses ZifyInst ZifyBool ZifyN ZifyNat ZifyPow.
From NeuralNetInterp.Util.Arith Require Import ZArith Classes Instances.

#[local] Open Scope core_scope.

#[export] Instance Op_bool_add : BinOp (add (A:=bool)) := _.
Add Zify BinOp Op_bool_add.
#[export] Instance Op_bool_mul : BinOp (mul (A:=bool)) := _.
Add Zify BinOp Op_bool_mul.
#[export] Instance Op_bool_eqb : BinOp (eqb (A:=bool)) := _.
Add Zify BinOp Op_bool_eqb.
#[export] Instance Op_bool_min : BinOp (min (A:=bool)) := _.
Add Zify BinOp Op_bool_min.
#[export] Instance Op_bool_max : BinOp (max (A:=bool)) := _.
Add Zify BinOp Op_bool_max.
#[export] Instance Op_bool_zero : CstOp (zero (A:=bool)) := _.
Add Zify CstOp Op_bool_zero.
#[export] Instance Op_bool_one : CstOp (one (A:=bool)) := _.
Add Zify CstOp Op_bool_one.

#[export] Instance Op_nat_add : BinOp (add (A:=nat)) := _.
Add Zify BinOp Op_nat_add.
#[export] Instance Op_nat_mul : BinOp (mul (A:=nat)) := _.
Add Zify BinOp Op_nat_mul.
#[export] Instance Op_nat_sub : BinOp (sub (A:=nat)) := _.
Add Zify BinOp Op_nat_sub.
#[export] Instance Op_nat_int_div : BinOp (int_div (A:=nat)) := _.
Add Zify BinOp Op_nat_int_div.
#[export] Instance Op_nat_modulo : BinOp (modulo (A:=nat)) := _.
Add Zify BinOp Op_nat_modulo.
#[export] Instance Op_nat_eqb : BinOp (eqb (A:=nat)) := _.
Add Zify BinOp Op_nat_eqb.
#[export] Instance Op_nat_leb : BinOp (leb (A:=nat)) := _.
Add Zify BinOp Op_nat_leb.
#[export] Instance Op_nat_ltb : BinOp (ltb (A:=nat)) := _.
Add Zify BinOp Op_nat_ltb.
#[export] Instance Op_nat_pow : BinOp (pow (A:=nat)) := _.
Add Zify BinOp Op_nat_pow.
#[export] Instance Op_nat_min : BinOp (min (A:=nat)) := _.
Add Zify BinOp Op_nat_min.
#[export] Instance Op_nat_max : BinOp (max (A:=nat)) := _.
Add Zify BinOp Op_nat_max.
#[export] Instance Op_nat_abs : UnOp (abs (A:=nat)) := { TUOp x := x ; TUOpInj _ := eq_refl }.
Add Zify UnOp Op_nat_abs.
(*
#[export] Instance Op_nat_sqrt : UnOp (sqrt (A:=nat)) := { TUOp x := x ; TUOpInj _ := eq_refl }.
Add Zify UnOp Op_nat_sqrt.
*)
#[export] Instance Op_nat_zero : CstOp (zero (A:=nat)) := _.
Add Zify CstOp Op_nat_zero.
#[export] Instance Op_nat_one : CstOp (one (A:=nat)) := { TCst := 1%Z ; TCstInj := eq_refl }.
Add Zify CstOp Op_nat_one.

#[export] Instance Op_N_add : BinOp (add (A:=N)) := _.
Add Zify BinOp Op_N_add.
#[export] Instance Op_N_mul : BinOp (mul (A:=N)) := _.
Add Zify BinOp Op_N_mul.
#[export] Instance Op_N_sub : BinOp (sub (A:=N)) := _.
Add Zify BinOp Op_N_sub.
#[export] Instance Op_N_int_div : BinOp (int_div (A:=N)) := _.
Add Zify BinOp Op_N_int_div.
#[export] Instance Op_N_modulo : BinOp (modulo (A:=N)) := _.
Add Zify BinOp Op_N_modulo.
#[export] Instance Op_N_eqb : BinOp (eqb (A:=N)) := _.
Add Zify BinOp Op_N_eqb.
#[export] Instance Op_N_leb : BinOp (leb (A:=N)) := _.
Add Zify BinOp Op_N_leb.
#[export] Instance Op_N_ltb : BinOp (ltb (A:=N)) := _.
Add Zify BinOp Op_N_ltb.
#[export] Instance Op_N_pow : BinOp (pow (A:=N)) := _.
Add Zify BinOp Op_N_pow.
#[export] Instance Op_N_min : BinOp (min (A:=N)) := _.
Add Zify BinOp Op_N_min.
#[export] Instance Op_N_max : BinOp (max (A:=N)) := _.
Add Zify BinOp Op_N_max.
#[export] Instance Op_N_abs : UnOp (abs (A:=N)) := { TUOp x := x ; TUOpInj _ := eq_refl }.
Add Zify UnOp Op_N_abs.
#[export] Instance Op_N_sqrt : UnOp (sqrt (A:=N)) := { TUOp := Z.sqrt ; TUOpInj := ltac:(intros []; reflexivity) }.
Add Zify UnOp Op_N_sqrt.
#[export] Instance Op_N_zero : CstOp (zero (A:=N)) := _.
Add Zify CstOp Op_N_zero.
#[export] Instance Op_N_one : CstOp (one (A:=N)) := { TCst := 1%Z ; TCstInj := eq_refl }.
Add Zify CstOp Op_N_one.

#[export] Instance Op_positive_add : BinOp (add (A:=positive)) := _.
Add Zify BinOp Op_positive_add.
#[export] Instance Op_positive_mul : BinOp (mul (A:=positive)) := _.
Add Zify BinOp Op_positive_mul.
#[export] Instance Op_positive_sub : BinOp (sub (A:=positive)) := _.
Add Zify BinOp Op_positive_sub.
#[export] Instance Op_positive_eqb : BinOp (eqb (A:=positive)) := _.
Add Zify BinOp Op_positive_eqb.
#[export] Instance Op_positive_leb : BinOp (leb (A:=positive)) := _.
Add Zify BinOp Op_positive_leb.
#[export] Instance Op_positive_ltb : BinOp (ltb (A:=positive)) := _.
Add Zify BinOp Op_positive_ltb.
#[export] Instance Op_positive_pow : BinOp (pow (A:=positive)) := _.
Add Zify BinOp Op_positive_pow.
#[export] Instance Op_positive_min : BinOp (min (A:=positive)) := _.
Add Zify BinOp Op_positive_min.
#[export] Instance Op_positive_max : BinOp (max (A:=positive)) := _.
Add Zify BinOp Op_positive_max.
#[export] Instance Op_positive_abs : UnOp (abs (A:=positive)) := { TUOp x := x ; TUOpInj _ := eq_refl }.
Add Zify UnOp Op_positive_abs.
#[export] Instance Op_positive_sqrt : UnOp (sqrt (A:=positive)) := { TUOp := Z.sqrt ; TUOpInj := ltac:(intros []; reflexivity) }.
Add Zify UnOp Op_positive_sqrt.
#[export] Instance Op_positive_one : CstOp (one (A:=positive)) := { TCst := 1%Z ; TCstInj := eq_refl }.
Add Zify CstOp Op_positive_one.

#[export] Instance Op_Z_add : BinOp (add (A:=Z)) := _.
Add Zify BinOp Op_Z_add.
#[export] Instance Op_Z_mul : BinOp (mul (A:=Z)) := _.
Add Zify BinOp Op_Z_mul.
#[export] Instance Op_Z_sub : BinOp (sub (A:=Z)) := _.
Add Zify BinOp Op_Z_sub.
#[export] Instance Op_Z_int_div : BinOp (int_div (A:=Z)) := _.
Add Zify BinOp Op_Z_int_div.
#[export] Instance Op_Z_modulo : BinOp (modulo (A:=Z)) := _.
Add Zify BinOp Op_Z_modulo.
#[export] Instance Op_Z_eqb : BinOp (eqb (A:=Z)) := _.
Add Zify BinOp Op_Z_eqb.
#[export] Instance Op_Z_leb : BinOp (leb (A:=Z)) := _.
Add Zify BinOp Op_Z_leb.
#[export] Instance Op_Z_ltb : BinOp (ltb (A:=Z)) := _.
Add Zify BinOp Op_Z_ltb.
#[export] Instance Op_Z_pow : BinOp (pow (A:=Z)) := { TBOp := Z.pow ; TBOpInj := ltac:(intros [] []; reflexivity) }.
Add Zify BinOp Op_Z_pow.
#[export] Instance Op_Z_min : BinOp (min (A:=Z)) := _.
Add Zify BinOp Op_Z_min.
#[export] Instance Op_Z_max : BinOp (max (A:=Z)) := _.
Add Zify BinOp Op_Z_max.
#[export] Instance Op_Z_opp : UnOp (opp (A:=Z)) := _.
Add Zify UnOp Op_Z_opp.
#[export] Instance Op_Z_abs : UnOp (abs (A:=Z)) := { TUOp := Z.abs ; TUOpInj _ := eq_refl }.
Add Zify UnOp Op_Z_abs.
#[export] Instance Op_Z_sqrt : UnOp (sqrt (A:=Z)) := { TUOp := Z.sqrt ; TUOpInj _ := eq_refl }.
Add Zify UnOp Op_Z_sqrt.
#[export] Instance Op_Z_zero : CstOp (zero (A:=Z)) := _.
Add Zify CstOp Op_Z_zero.
#[export] Instance Op_Z_one : CstOp (one (A:=Z)) := { TCst := 1%Z ; TCstInj := eq_refl }.
Add Zify CstOp Op_Z_one.
