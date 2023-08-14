From Coq.Floats Require Import FloatOps.
From NeuralNetInterp.Util.Arith.Flocq Require Import Hints.Core.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.

Definition fold_B2Prim x : SF2Prim (B2SF x) = B2Prim x := eq_refl.

#[local] Existing Instances Hprec.
#[local] Notation eta x := (fst x, snd x).

Lemma frshiftexp_equiv_eta x : let (m, e) := eta (PrimFloat.frshiftexp x) in
                               (Prim2B m, BinInt.Z.sub (Uint63.to_Z e) shift) = Bfrexp (Prim2B x).
Proof. generalize (frshiftexp_equiv x); destruct PrimFloat.frshiftexp; exact id. Qed.
Lemma fst_frshiftexp_equiv x : Prim2B (fst (PrimFloat.frshiftexp x)) = fst (Bfrexp (Prim2B x)).
Proof. now rewrite <- frshiftexp_equiv_eta. Qed.
Lemma snd_frshiftexp_equiv x : BinInt.Z.sub (Uint63.to_Z (snd (PrimFloat.frshiftexp x))) shift = snd (Bfrexp (Prim2B x)).
Proof. now rewrite <- frshiftexp_equiv_eta. Qed.

Lemma frexp_equiv_eta x : let (m, e) := eta (FloatOps.Z.frexp x) in
                               (Prim2B m, e) = Bfrexp (Prim2B x).
Proof. generalize (frexp_equiv x); destruct FloatOps.Z.frexp; exact id. Qed.
Lemma fst_frexp_equiv x : Prim2B (fst (FloatOps.Z.frexp x)) = fst (Bfrexp (Prim2B x)).
Proof. now rewrite <- frexp_equiv_eta. Qed.
Lemma snd_frexp_equiv x : snd (FloatOps.Z.frexp x) = snd (Bfrexp (Prim2B x)).
Proof. now rewrite <- frexp_equiv_eta. Qed.

#[export]
  Hint Rewrite
  is_finite_equiv
  get_sign_equiv
  is_nan_equiv
  abs_equiv
  opp_equiv
  eqb_equiv
  ltb_equiv
  leb_equiv
  is_zero_equiv
  is_infinity_equiv
  normfr_mantissa_equiv
  compare_equiv
  ulp_equiv
  next_up_equiv
  next_down_equiv
  sqrt_equiv
  sub_equiv
  mul_equiv
  add_equiv
  div_equiv
  ldexp_equiv
  fst_frexp_equiv
  snd_frexp_equiv
  of_int63_equiv
  ldshiftexp_equiv
  fst_frshiftexp_equiv
  snd_frshiftexp_equiv

  binary_normalize_equiv

  fold_B2Prim

  Prim2B_B2Prim
  : prim2b.

#[export]
  Hint Rewrite <-
  is_finite_equiv
  get_sign_equiv
  is_nan_equiv
  abs_equiv
  opp_equiv
  eqb_equiv
  ltb_equiv
  leb_equiv
  is_zero_equiv
  is_infinity_equiv
  normfr_mantissa_equiv
  compare_equiv
  ulp_equiv
  next_up_equiv
  next_down_equiv
  sqrt_equiv
  sub_equiv
  mul_equiv
  add_equiv
  div_equiv
  ldexp_equiv
  frexp_equiv_eta
  of_int63_equiv
  ldshiftexp_equiv
  frshiftexp_equiv_eta

  binary_normalize_equiv

  : b2prim.

#[export]
  Hint Rewrite

  fold_B2Prim

  Prim2B_B2Prim
  : b2prim.
