From Coq.Floats Require Import FloatOps.
From NeuralNetInterp.Util.Arith.Flocq Require Import Hints.Core.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.

Definition fold_B2Prim x : SF2Prim (B2SF x) = B2Prim x := eq_refl.

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
  of_int63_equiv
  (*ldshiftexp_equiv*)
  (*frshiftexp_equiv*)

  binary_normalize_equiv

  fold_B2Prim

  Prim2B_B2Prim
  : prim2b.
