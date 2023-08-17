From Coq Require Import Reals PArith ZArith.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util Require Import Arith.Classes Arith.Instances.
From NeuralNetInterp.Util.Arith.Flocq Require Import Definitions.

#[export] Instance coer_float_binary_float : has_coer float _ := Prim2B.
#[export] Instance coer_binary_float_R {prec emax} : has_coer (binary_float prec emax) R := B2R.
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion Prim2B : float >-> binary_float.
#[export] Set Warnings Append "-ambiguous-paths".
#[export] Coercion B2R : binary_float >-> R.
#[export] Set Warnings Append "ambiguous-paths".
#[local] Set Warnings Append "unsupported-attributes".
#[export] Hint Extern 10 (has_coer_from _ float ?B) => check_unify_has_coer_from (binary_float prec emax) : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to _ ?A (binary_float _ _)) => check_unify_has_coer_to float : typeclass_instances.
#[export] Hint Extern 10 (has_coer_from _ (binary_float _ _) ?B) => check_unify_has_coer_from R : typeclass_instances.
#[export] Hint Extern 10 (has_coer_to _ ?A R) => check_unify_has_coer_to (binary_float prec emax) : typeclass_instances.
#[export] Instance binary_float_has_leb {prec emax} : has_leb (binary_float prec emax) := Bleb.
#[export] Instance binary_float_has_ltb {prec emax} : has_ltb (binary_float prec emax) := Bltb.
#[export] Instance binary_float_has_opp {prec emax} : has_opp (binary_float prec emax) := Bopp.
#[export] Instance binary_float_has_abs {prec emax} : has_abs (binary_float prec emax) := Babs.
#[export] Instance binary_float_has_sqrt {prec emax prec_gt_0_ prec_lt_emax_} : has_sqrt (binary_float prec emax) := @Bsqrt prec emax prec_gt_0_ prec_lt_emax_ mode_NE (* to match with prim float *).
#[export] Instance binary_float_has_add {prec emax prec_gt_0_ prec_lt_emax_} : has_add (binary_float prec emax) := @Bplus prec emax prec_gt_0_ prec_lt_emax_ mode_NE.
#[export] Instance binary_float_has_sub {prec emax prec_gt_0_ prec_lt_emax_} : has_sub (binary_float prec emax) := @Bminus prec emax prec_gt_0_ prec_lt_emax_ mode_NE.
#[export] Instance binary_float_has_mul {prec emax prec_gt_0_ prec_lt_emax_} : has_mul (binary_float prec emax) := @Bmult prec emax prec_gt_0_ prec_lt_emax_ mode_NE.
#[export] Instance binary_float_has_div {prec emax prec_gt_0_ prec_lt_emax_} : has_div (binary_float prec emax) := @Bdiv prec emax prec_gt_0_ prec_lt_emax_ mode_NE.
#[export] Instance binary_float_has_zero {prec emax} : has_zero (binary_float prec emax) := B754_zero false.
#[export] Instance binary_float_has_one {prec emax prec_gt_0_ prec_lt_emax_} : has_one (binary_float prec emax) := @Bone prec emax prec_gt_0_ prec_lt_emax_.
#[export] Existing Instances Hprec Hmax.
#[export] Instance binary_float_has_exp : has_exp (binary_float _ _) := Bexp.
#[export] Instance binary_float_has_ln : has_ln (binary_float _ _) := Bln.
#[export] Instance binary_float_has_is_nan {prec emax} : has_is_nan (binary_float prec emax) := BinarySingleNaN.is_nan.
#[export] Instance binary_float_has_nan {prec emax} : has_nan (binary_float prec emax) := B754_nan.
#[export] Instance binary_float_has_is_infinity {prec emax} : has_is_infinity (binary_float prec emax)
  := fun x => match x with
              | B754_infinity _ => true
              | _ => false
              end.
#[export] Instance binary_float_has_infinity {prec emax} : has_infinity (binary_float prec emax) := B754_infinity.
#[export] Instance binary_float_has_get_sign {prec emax} : has_get_sign (binary_float prec emax) := Bsign.

Module Float.
  Module IEEE754Eq.
    #[export] Instance binary_float_has_eqb {prec emax} : has_eqb (binary_float prec emax) := @Beqb prec emax.
  End IEEE754Eq.

  Module Leibniz.
    #[export] Instance binary_float_has_eqb {prec emax} : has_eqb (binary_float prec emax)
    := fun x y => match x, y with
                  | B754_zero sx, B754_zero sy => Bool.eqb sx sy
                  | B754_infinity sx, B754_infinity sy => Bool.eqb sx sy
                  | B754_nan, B754_nan => true
                  | B754_finite sx mx ex _, B754_finite sy my ey _
                    => Bool.eqb sx sy && Pos.eqb mx my && Z.eqb ex ey
                  | B754_zero _, _
                  | B754_infinity _, _
                  | B754_nan, _
                  | B754_finite _ _ _ _, _
                    => false
                  end%bool.
  End Leibniz.
End Float.

Module Truncating.
  #[export] Instance coer_Z_binary_float {prec emax Hprec Hmax} : has_coer Z (binary_float prec emax) := fun z => binary_normalize prec emax Hprec Hmax mode_NE z 0 false.
  #[export] Hint Extern 10 (has_coer_from Z ?B) => check_unify_has_coer_from (binary_float prec emax) : typeclass_instances.
  #[export] Hint Extern 10 (has_coer_to ?A (binary_float _ _)) => check_unify_has_coer_to Z : typeclass_instances.
End Truncating.
