From Coq Require Import Qabs Qround Sint63 Uint63 NArith PArith ZArith QArith Floats Morphisms.
From Coq.Reals Require Import Reals.
From Coq Require Import Lra Lia Eqdep_dec.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util Require Import Default.
From NeuralNetInterp.Util.Arith Require Import QArith FloatArith.Definitions FloatArith.Proofs Flocq.Notations.
From NeuralNetInterp.Util.Arith.Flocq.Hints Require Export Core Prim2B.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose.
#[local] Open Scope bool_scope.
#[local] Set Implicit Arguments.
#[local] Existing Instances Hmax Hprec.
#[local] Notation binary_float := (binary_float prec emax).
#[local] Notation binary_normalize := (binary_normalize prec emax Hprec Hmax mode_NE).
#[local] Open Scope float_scope.
#[local] Open Scope binary_float_scope.
#[local] Coercion Prim2B : float >-> binary_float.
Notation "'∞'" := (B754_infinity false) : binary_float_scope.
Notation "'-∞'" := (B754_infinity true) : binary_float_scope.
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion N.of_nat : nat >-> N.
#[local] Coercion inject_Z : Z >-> Q.
#[local] Coercion Z.pos : positive >-> Z.
#[local] Coercion is_true : bool >-> Sortclass.

(*
  Definition of_sint63 (x : int) : float
    := if Sint63.ltb x 0
       then (-(PrimFloat.of_uint63 (-x)%sint63))%float
       else PrimFloat.of_uint63 x.

  (** Fused multiply and add [x * y + z] with a single round *)
  Definition SFfma prec emax (x y z : spec_float) : spec_float
    := match x, y, z with
       | S754_nan, _, _
       | _, S754_nan, _
       | _, _, S754_nan
         => S754_nan
       | S754_infinity _, _, _
       | _, S754_infinity _, _
         => SFadd prec emax (SFmul prec emax x y) z
       | _, _, S754_infinity _
         => z
       | S754_zero _, _, _
       | _, S754_zero _, _
         => SFadd prec emax (SFmul prec emax x y) z
       | _, _, S754_zero _
         (* in most cases, we could do [SFadd (SFmul x y) z] here too,
            but in the case where [x] and [y] are subnormal, [x * y]
            underflows to [-0] in [SFmul], and [z] is [+0], we should
            keep the sign of [x * y] rather than re-rounding by adding
            with [z] *)
         => SFmul prec emax x y
       | S754_finite sx mx ex, S754_finite sy my ey, S754_finite sz mz ez
         => let xy := S754_finite (xorb sx sy) (mx * my) (ex + ey) (* no rounding *) in
            SFadd prec emax xy z
       end.
  Definition SF64fma := SFfma prec emax.
 *)
Definition Bfmaf (x y z : binary_float) : binary_float
  := (x * y + z).

Notation mul_2p m e
  := (if Z.ltb e 0
      then Qdiv m (inject_Z (2^(-e)))
      else Qmult m (inject_Z (2^e)))
       (only parsing).
Notation div_2p m e
  := (if Z.ltb e 0
      then Qmult m (inject_Z (2^(-e)))
      else Qdiv m (inject_Z (2^e)))
       (only parsing).

Definition Bto_Q_cps (x : binary_float) {T} (on_nan : T) (on_pinf : T) (on_ninf : T) (on_nzero : T) (on_Q : Q -> T) : T
  := match x with
     | B754_nan => on_nan
     | B754_zero is_neg
       => if is_neg
          then on_nzero
          else on_Q 0%Q
     | B754_infinity false => on_pinf
     | B754_infinity true => on_ninf
     | B754_finite is_neg m e _
       => let z := mul_2p m e in
          on_Q (if is_neg then -z else z)%Q
     end.

Definition Bto_Q
  {nan : with_default "nan" Q 0%Q}
  {pinf : with_default "+∞" Q (2^emax)%Z}
  {ninf : with_default "-∞" Q (-(2^emax))%Z}
  {nzero : with_default "-0" Q (-1 / (2^(-(emin-1)))%Z)%Q}
  (x : binary_float)
  : Q
  := Bto_Q_cps x nan pinf ninf nzero (fun x => x).

(*
  Definition Z2SF (z : Z) : spec_float := binary_normalize prec emax z 0 false.
  Definition Q2SF (q : Q) : spec_float
    := let is_neg := negb (Qle_bool 0 q) in
       let q := Qred (Qabs q) in
       (* goal: Z.log2 (Z.pos m) should be at least prec, ideally exactly prec, and m * 2^e should be close to q *)
       let e := (Z.log2 (Qnum q) - Z.log2 (Qden q) - prec - 1)%Z in
       let q := div_2p q e in
       let m := Qround q in
       let m' := if is_neg then Z.opp m else m in
       binary_normalize prec emax m e is_neg.
  Definition of_Q (q : Q) : binary_float := SF2Prim (Q2SF q).
 *)

Definition Bof_Z (z : Z) : binary_float := binary_normalize z 0 false.
(*Definition Bof_N (n : N) : binary_float := Bof_Z n.
  Definition of_nat (n : nat) : binary_float := of_N n.*)
Definition Bto_Z_cps (x : binary_float) {T} (on_nan : T) (on_pinf : T) (on_ninf : T) (on_nzero : T) (on_Z : Z -> T) : T
  := Bto_Q_cps x on_nan on_pinf on_ninf on_nzero (fun q => on_Z (Qround q)).
Definition Bto_Z
  {nan : with_default "nan" Z 0%Z}
  {pinf : with_default "+∞" Z (2^emax)%Z}
  {ninf : with_default "-∞" Z (-(2^emax))%Z}
  {nzero : with_default "-0" Z 0%Z}
  (x : binary_float)
  : Z
  := Bto_Z_cps x nan pinf ninf nzero (fun x => x).

(** Translating from https://stackoverflow.com/a/40519989/377022 with GPT-4
<<<
#include <stdint.h> // for uint32_t
#include <string.h> // for memcpy
#include <math.h>   // for frexpf, ldexpf, isinf, nextafterf

#define PORTABLE (1) // 0=bit-manipulation of 'float', 1= math library functions

uint32_t float_as_uint32 (float a)
{
    uint32_t r;
    memcpy (&r, &a, sizeof r);
    return r;
}

float uint32_as_float (uint32_t a)
{
    float r;
    memcpy (&r, &a, sizeof r);
    return r;
}

/* Compute log(a) with extended precision, returned as a double-float value
   loghi:loglo. Maximum relative error: 8.5626e-10.
*/
void my_logf_ext (float a, float *loghi, float *loglo)
{
    const float LOG2_HI =  6.93147182e-1f; //  0x1.62e430p-1
    const float LOG2_LO = -1.90465421e-9f; // -0x1.05c610p-29
    const float SQRT_HALF = 0.70710678f;
    float m, r, i, s, t, p, qhi, qlo;
    int e;

    /* Reduce argument to m in [sqrt(0.5), sqrt(2.0)] */
#if PORTABLE
    m = frexpf (a, &e);
    if (m < SQRT_HALF) {
        m = m + m;
        e = e - 1;
    }
    i = (float)e;
#else // PORTABLE
    const float POW_TWO_M23 = 1.19209290e-7f; // 0x1.0p-23
    const float POW_TWO_P23 = 8388608.0f; // 0x1.0p+23
    const float FP32_MIN_NORM = 1.175494351e-38f; // 0x1.0p-126
    i = 0.0f;
    /* fix up denormal inputs */
    if (a < FP32_MIN_NORM){
        a = a * POW_TWO_P23;
        i = -23.0f;
    }
    e = (float_as_uint32 (a) - float_as_uint32 (SQRT_HALF)) & 0xff800000;
    m = uint32_as_float (float_as_uint32 (a) - e);
    i = fmaf ((float)e, POW_TWO_M23, i);
#endif // PORTABLE
    /* Compute q = (m-1)/(m+1) as a double-float qhi:qlo */
    p = m + 1.0f;
    m = m - 1.0f;
    r = 1.0f / p;
    qhi = r * m;
    qlo = r * fmaf (qhi, -m, fmaf (qhi, -2.0f, m));
    /* Approximate atanh(q), q in [sqrt(0.5)-1, sqrt(2)-1] */
    s = qhi * qhi;
    r =             0.1293334961f;  // 0x1.08c000p-3
    r = fmaf (r, s, 0.1419928074f); // 0x1.22cd9cp-3
    r = fmaf (r, s, 0.2000148296f); // 0x1.99a162p-3
    r = fmaf (r, s, 0.3333332539f); // 0x1.555550p-2
    t = fmaf (qhi, qlo + qlo, fmaf (qhi, qhi, -s)); // s:t = (qhi:qlo)**2
    p = s * qhi;
    t = fmaf (s, qlo, fmaf (t, qhi, fmaf (s, qhi, -p))); // p:t = (qhi:qlo)**3
    s = fmaf (r, p, fmaf (r, t, qlo));
    r = 2 * qhi;
    /* log(a) = 2 * atanh(q) + i * log(2) */
    t = fmaf ( LOG2_HI, i, r);
    p = fmaf (-LOG2_HI, i, t);
    s = fmaf ( LOG2_LO, i, fmaf (2.f, s, r - p));
    *loghi = p = t + s;    // normalize double-float result
    *loglo = (t - p) + s;
}
 *)
(** Compute log(a) with extended precision, returned as a double-float value
      loghi:loglo. Maximum relative error: 8.5626e-10.
 *)
Definition Bln_ext (a : binary_float) : binary_float * binary_float
  := let LOG2_HI   := Prim2B   0x1.62e430p-1 (* 6.93147182e-1f*) in
     let LOG2_LO   := Prim2B (-0x1.05c610p-29) (* -1.90465421e-9f *) in
     let SQRT_HALF := Prim2B   0x1.6a09e65dc27dfp-1 (* 0.70710678f *) in
     (* Reduce argument to m in [sqrt(0.5), sqrt(2.0)] *)
     let '(m, e) := Bfrexp a in
     let '(m, e) := if m <? SQRT_HALF
                    then (m + m, (e - 1)%Z)
                    else (m, e) in
     let i := Bof_Z e in
     (* Compute q = (m-1)/(m+1) as a double-float qhi:qlo *)
     let p := m + Prim2B 1.0 in
     let m := m - Prim2B 1.0 in
     let r := Prim2B 1.0 / p in
     let qhi := r * m in
     let qlo := r * Bfmaf qhi (-m) (Bfmaf qhi (-2.0) m) in
     (* Approximate atanh(q), q in [sqrt(0.5)-1, sqrt(2)-1] *)
     let s := qhi * qhi in
     let r :=          0x1.08c000p-3 in (* 0.1293334961f *)
     let r := Bfmaf r s 0x1.22cd9cp-3 in (* 0.1419928074f *)
     let r := Bfmaf r s 0x1.99a162p-3 in (* 0.2000148296f *)
     let r := Bfmaf r s 0x1.555550p-2 in (* 0.3333332539f *)
     let t := Bfmaf qhi (qlo + qlo) (Bfmaf qhi qhi (-s)) in (* s:t = (qhi:qlo)**2 *)
     let p := s * qhi in
     let t := Bfmaf s qlo (Bfmaf t qhi (Bfmaf s qhi (-p))) in (* p:t = (qhi:qlo)**3 *)
     let s := Bfmaf r p (Bfmaf r t qlo) in
     let r := 2 * qhi in
     (* log(a) = 2 * atanh(q) + i * log(2) *)
     let t := Bfmaf   LOG2_HI  i r in
     let p := Bfmaf (-LOG2_HI) i t in
     let s := Bfmaf ( LOG2_LO) i (Bfmaf 2.0 s (r - p)) in
     let p := t + s in
     let loghi := p in (* normalize double-float result *)
     let loglo := (t - p) + s in
     (loghi, loglo).

Definition Bln (a : binary_float) : binary_float := fst (Bln_ext a).

(**
<<<
/* Compute exponential base e. No checking for underflow and overflow. Maximum
   ulp error = 0.86565
*/
float my_expf_unchecked (float a)
{
    float f, j, r;
    int i;

    // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = fmaf (1.442695f, a, 12582912.f) - 12582912.f; // 0x1.715476p0, 0x1.8p23
    f = fmaf (j, -6.93145752e-1f, a); // -0x1.62e400p-1  // log_2_hi
    f = fmaf (j, -1.42860677e-6f, f); // -0x1.7f7d1cp-20 // log_2_lo
    i = (int)j;
    // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
    r =             1.37805939e-3f;  // 0x1.694000p-10
    r = fmaf (r, f, 8.37312452e-3f); // 0x1.125edcp-7
    r = fmaf (r, f, 4.16695364e-2f); // 0x1.555b5ap-5
    r = fmaf (r, f, 1.66664720e-1f); // 0x1.555450p-3
    r = fmaf (r, f, 4.99999851e-1f); // 0x1.fffff6p-2
    r = fmaf (r, f, 1.00000000e+0f); // 0x1.000000p+0
    r = fmaf (r, f, 1.00000000e+0f); // 0x1.000000p+0
    // exp(a) = 2**i * r
#if PORTABLE
    r = ldexpf (r, i);
#else // PORTABLE
    float s, t;
    uint32_t ia = (i > 0) ? 0u : 0x83000000u;
    s = uint32_as_float (0x7f000000u + ia);
    t = uint32_as_float (((uint32_t)i << 23) - ia);
    r = r * s;
    r = r * t;
#endif // PORTABLE
    return r;
}
 *)
(** Compute exponential base e. No checking for underflow and overflow. Maximum
      ulp error = 0.86565
 *)
Definition Bexp (a : binary_float) : binary_float :=
  (* exp(a) = 2**i * exp(f); i = rintf (a / log(2)) *)
  let big := 0x1.8p0 * Bof_Z (2^prec) in
  let j := Bfmaf 0x1.715476p0 a big - big in (* 1.442695f, 12582912f *)
  let f := Bfmaf j (-0x1.62e400p-1)  a in (* -6.93145752e-1f *) (* log_2_hi *)
  let f := Bfmaf j (-0x1.7f7d1cp-20) f in (* -1.42860677e-6f *) (* log_2_lo *)
  match Bto_Z_cps j (inr B754_nan) (inr ∞) (inr (-∞)) (inr (1:binary_float)) (@inl _ _) with (* inl i or inr return *)
  | inr out_of_bounds => out_of_bounds
  | inl i =>
      (* approximate r = exp(f) on interval [-log(2)/2, +log(2)/2] *)
      let r :=          0x1.694000p-10 in (* 1.37805939e-3f *)
      let r := Bfmaf r f 0x1.125edcp-7  in (* 8.37312452e-3f *)
      let r := Bfmaf r f 0x1.555b5ap-5  in (* 4.16695364e-2f *)
      let r := Bfmaf r f 0x1.555450p-3  in (* 1.66664720e-1f *)
      let r := Bfmaf r f 0x1.fffff6p-2  in (* 4.99999851e-1f *)
      let r := Bfmaf r f 0x1.000000p+0  in (* 1.00000000e+0f *)
      let r := Bfmaf r f 0x1.000000p+0  in (* 1.00000000e+0f *)
      (* exp(a) = 2**i * r *)
      let r := Bldexp mode_NE r i in
      r
  end.

(**
<<<
/* a**b = exp (b * log (a)), where a > 0, and log(a) is computed with extended
   precision as a double-float. Maxiumum error found across 2**42 test cases:
   1.97302 ulp @ (0.71162397, -256.672424).
*/
float my_powf_core (float a, float b)
{
    const float LET MAX_IEEE754_FLT := uint32_as_float (0x7f7fffff) in
    const float LET EXP_OVFL_BOUND := 88.7228394f in (* 0x1.62e430p+6f in *)
    const float LET EXP_OVFL_UNFL_F := 104.0f in
    const float LET MY_INF_F := uint32_as_float (0x7f800000) in
    float lhi, llo, thi, tlo, phi, plo, r in

    /* compute lhi:llo = log(a) */
    my_logf_ext (a, &lhi, &llo) in
    /* compute phi:plo = b * log(a) */
    let thi := lhi * b in
    if (fabsf (thi) > EXP_OVFL_UNFL_F) { (* definitely overflow / underflow *)
        let r := (thi < 0.0f) ? 0.0f : MY_INF_F in
    } else {
        let tlo := fmaf (lhi, b, -thi) in
        let tlo := fmaf (llo, b, +tlo) in
        /* normalize intermediate result thi:tlo, giving final result phi:plo */
#if FAST_FADD_RZ
        let phi := __fadd_rz (thi, tlo) in(* avoid premature ovfl in exp() computation *)
#else (* FAST_FADD_RZ *)
        let phi := thi + tlo in
        if (phi == EXP_OVFL_BOUND){(* avoid premature ovfl in exp() computation *)
#if PORTABLE
            let phi := nextafterf (phi, 0.0f) in
#else (* PORTABLE *)
            let phi := uint32_as_float (float_as_uint32 (phi) - 1) in
#endif (* PORTABLE *)
        }
#endif (* FAST_FADD_RZ *)
        let plo := (thi - phi) + tlo in
        /* exp'(x) = exp(x); exp(x+y) = exp(x) + exp(x) * y, for |y| << |x| */
        let r := my_expf_unchecked (phi) in
        /* prevent generation of NaN during interpolation due to r = INF */
        if (fabsf (r) <= MAX_IEEE754_FLT) {
            let r := fmaf (plo, r, r) in
        }
    }
    return r in
}

float my_powf (float a, float b)
{
    const float LET MY_INF_F := uint32_as_float (0x7f800000) in
    const float LET MY_NAN_F := uint32_as_float (0xffc00000) in
    int expo_odd_int in
    float r in

    /* special case handling per ISO C specification */
    let expo_odd_int := fmaf (-2.0f, floorf (0.5f * b), b) == 1.0f in
    if ((a == 1.0f) || (b == 0.0f)) {
        let r := 1.0f in
    } else if (isnan (a) || isnan (b)) {
        let r := a + b in  (* convert SNaN to QNanN or trigger exception *)
    } else if (isinf (b)) {
        let r := ((fabsf (a) < 1.0f) != (b < 0.0f)) ? 0.0f :  MY_INF_F in
        if (a == -1.0f) let r := 1.0f in
    } else if (isinf (a)) {
        let r := (b < 0.0f) ? 0.0f : MY_INF_F in
        if ((a < 0.0f) && expo_odd_int) let r := -r in
    } else if (a == 0.0f) {
        let r := (expo_odd_int) ? (a + a) : 0.0f in
        if (b < 0.0f) let r := copysignf (MY_INF_F, r) in
    } else if ((a < 0.0f) && (b != floorf (b))) {
        let r := MY_NAN_F in
    } else {
        let r := my_powf_core (fabsf (a), b) in
        if ((a < 0.0f) && expo_odd_int) {
            let r := -r in
        }
    }
    return r in
}
>>>
 *)

Lemma Bto_Q_cps_distr {A B} (f : A -> B) {x on_nan on_pinf on_ninf on_nzero on_Q}
  : f (Bto_Q_cps x on_nan on_pinf on_ninf on_nzero on_Q) = Bto_Q_cps x (f on_nan) (f on_pinf) (f on_ninf) (f on_nzero) (fun z => f (on_Q z)).
Proof. cbv [Bto_Q_cps]; break_innermost_match; reflexivity. Qed.

Lemma Bto_Z_cps_distr {A B} (f : A -> B) {x on_nan on_pinf on_ninf on_nzero on_Z}
  : f (Bto_Z_cps x on_nan on_pinf on_ninf on_nzero on_Z) = Bto_Z_cps x (f on_nan) (f on_pinf) (f on_ninf) (f on_nzero) (fun z => f (on_Z z)).
Proof. cbv [Bto_Z_cps]; rewrite (Bto_Q_cps_distr f); reflexivity. Qed.

Lemma to_Q_cps_equiv {x A on_nan on_pinf on_ninf on_nzero on_Q}
  : @PrimFloat.to_Q_cps x A on_nan on_pinf on_ninf on_nzero on_Q = Bto_Q_cps x on_nan on_pinf on_ninf on_nzero on_Q.
Proof.
  cbv [PrimFloat.to_Q_cps Bto_Q_cps Prim2B SF2B]; break_innermost_match.
  generalize (Prim2SF_valid x).
  break_innermost_match; reflexivity.
Qed.
#[export] Hint Rewrite @to_Q_cps_equiv : prim2b.

Lemma to_Z_cps_equiv {x A on_nan on_pinf on_ninf on_nzero on_Z}
  : @PrimFloat.to_Z_cps x A on_nan on_pinf on_ninf on_nzero on_Z = Bto_Z_cps x on_nan on_pinf on_ninf on_nzero on_Z.
Proof. now cbv [PrimFloat.to_Z_cps Bto_Z_cps]; autorewrite with prim2b. Qed.
#[export] Hint Rewrite @to_Z_cps_equiv : prim2b.

#[export] Instance Bto_Q_cps_Proper
  : Proper (eq ==> forall_relation (fun T => eq ==> eq ==> eq ==> eq ==> (pointwise_relation _ eq) ==> eq)) (@Bto_Q_cps).
Proof.
  repeat intro; subst; cbv [Bto_Q_cps]; break_innermost_match; eauto.
Qed.

#[export] Instance Bto_Z_cps_Proper
  : Proper (eq ==> forall_relation (fun T => eq ==> eq ==> eq ==> eq ==> (pointwise_relation _ eq) ==> eq)) (@Bto_Z_cps).
Proof.
  repeat intro; cbv [Bto_Z_cps]; apply Bto_Q_cps_Proper; eauto.
  intro; eauto.
Qed.

Lemma fmaf_equiv x y z : Prim2B (PrimFloat.fmaf x y z) = Bfmaf x y z.
Proof. cbv [Bfmaf PrimFloat.fmaf]; autorewrite with prim2b; reflexivity. Qed.
#[export] Hint Rewrite fmaf_equiv : prim2b.

Lemma of_Z_equiv x : Prim2B (PrimFloat.of_Z x) = Bof_Z x.
Proof. now cbv [PrimFloat.of_Z Bof_Z PrimFloat.Z2SF]; autorewrite with prim2b. Qed.
#[export] Hint Rewrite of_Z_equiv : prim2b.

Lemma ln_ext_equiv x
  : (let '(x, y) := PrimFloat.ln_ext x in (Prim2B x, Prim2B y)) = Bln_ext x.
Proof.
  cbv beta delta [Bln_ext PrimFloat.ln_ext]; autorewrite with prim2b.
  repeat first [ rewrite ltb_equiv
               | match goal with
                 | [ |- context[Z.frexp ?x] ]
                   => unique pose proof (frexp_equiv x)
                 | [ |- (let '(a, b) := let c := ?d in @?e c in @?f a b) = (let c' := ?d' in @?e' c') ]
                   => let c := fresh c in
                      let c' := fresh c' in
                      set (c' := d') in *;
                      set (c := d) in *;
                      change ((let '(a, b) := e c in f a b) = e' c');
                      cbv beta;
                      first [ cut (c = c')
                            | cut (Prim2B c = c') ];
                      [ intros <- | first [ reflexivity | shelve ] ]
                 | [ |- (let '(a, b) := let '(c0, c1) := ?d in @?e c0 c1 in @?f a b) = (let '(c0', c1') := ?d' in @?e' c0' c1') ]
                   => let c0 := fresh c0 in
                      let c1 := fresh c1 in
                      let c0' := fresh c0' in
                      let c1' := fresh c1' in
                      (destruct d as [c0 c1] eqn:?, d' as [c0' c1'] eqn:?);
                      first [ cut (c0 = c0')
                            | cut (Prim2B c0 = c0') ];
                      [ intros <- | first [ reflexivity | shelve ] ];
                      first [ cut (c1 = c1')
                            | cut (Prim2B c1 = c1') ];
                      [ intros <- | first [ reflexivity | shelve ] ]
                 end ].
  reflexivity.
  Unshelve.
  all: repeat first [ progress subst
                    | reflexivity
                    | progress destruct_head'_and
                    | match goal with
                      | [ H : (_, _) = (_, _) |- _ ] => inversion H; clear H
                      end
                    | progress autorewrite with prim2b
                    | match goal with
                      | [ |- Prim2B ?x = ?y ] => subst x y
                      end
                    | break_innermost_match_hyps_step
                    | match goal with
                      | [ H : ?x = (_, _) |- _ ] => rewrite (surjective_pairing x) in H; inversion H; clear H
                      end ].
Qed.
#[export] Hint Rewrite ln_ext_equiv : prim2b.

Lemma ln_equiv x : Prim2B (PrimFloat.ln x) = Bln x.
Proof.
  cbv [Bln PrimFloat.ln]; rewrite <- ln_ext_equiv; break_innermost_match; reflexivity.
Qed.
#[export] Hint Rewrite ln_equiv : prim2b.

Lemma exp_equiv x : Prim2B (PrimFloat.exp x) = Bexp x.
Proof.
  cbv [Bexp PrimFloat.exp].
  repeat first [ match goal with
                 | [ |- context[match Bto_Z_cps ?x ?on_nan ?on_pinf ?on_ninf ?on_nzero ?on_Z with inl i => @?L i | inr j => @?R j end] ]
                   => rewrite (@Bto_Z_cps_distr _ _ (fun v => match v with inl i => L i | inr j => R j end) x on_nan on_pinf on_ninf on_nzero on_Z)
                 | [ |- context[Prim2B (Bto_Z_cps ?x ?on_nan ?on_pinf ?on_ninf ?on_nzero ?on_Z)] ]
                   => rewrite (@Bto_Z_cps_distr _ _ Prim2B x on_nan on_pinf on_ninf on_nzero on_Z)
                 | [ |- Bto_Z_cps _ _ _ _ _ _ = Bto_Z_cps _ _ _ _ _ _ ]
                   => apply Bto_Z_cps_Proper; repeat intro
                 end
               | progress autorewrite with prim2b
               | reflexivity ].
Qed.
#[export] Hint Rewrite exp_equiv : prim2b.
