From Coq Require Import Qabs Qround Sint63 Uint63 NArith PArith ZArith QArith Floats.
From NeuralNetInterp.Util Require Import Default.
From NeuralNetInterp.Util.Arith Require Import QArith.
Local Open Scope float_scope.
Notation "'∞'" := infinity : float_scope.
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion inject_Z : Z >-> Q.
#[local] Coercion Z.pos : positive >-> Z.
#[local] Coercion is_true : bool >-> Sortclass.
Local Open Scope float_scope.
Notation "'∞'" := infinity : float_scope.

Module PrimFloat.
  Definition of_sint63 (x : int) : float
    := if Sint63.ltb x 0
       then (-(PrimFloat.of_uint63 (-x)%sint63))%float
       else PrimFloat.of_uint63 x.

  Definition to_Q_cps (x : float) {T} (on_nan : T) (on_pinf : T) (on_ninf : T) (on_nzero : T) (on_Q : Q -> T) : T
    := match Prim2SF x with
       | S754_nan => on_nan
       | S754_zero is_neg
         => if is_neg
            then on_nzero
            else on_Q 0%Q
       | S754_infinity false => on_pinf
       | S754_infinity true => on_ninf
       | S754_finite is_neg m e
         => let z := (if (e <? 0)%Z
                      then m / (2^(-e))%Z
                      else m * (2^e)%Z)%Q in
            on_Q (if is_neg then -z else z)%Q
       end.

  Definition Q2SF (q : Q) : spec_float
    := let is_neg := negb (Qle_bool 0 q) in
       let q := Qabs q in
       let '(q, e)
         := if Qle_bool q 1
            then let e := (Z.log2 (Qfloor (Qinv q)) + prec - 1)%Z in
                 (q * (2^e)%Z, (-e)%Z)%Q
            else let e := (Z.log2 (Qfloor q) + prec - 1)%Z in
                 (q / (2^e)%Z, e)%Q in
       if Qeq_bool q 0
       then S754_zero is_neg
       else
         if (e - prec <=? emax)%Z
         then
           let m := Qround q in
           if (m =? 0)%Z
           then S754_zero is_neg
           else
             let m := Z.to_pos m in
             if (emin <=? Pos.size m + e - prec)%Z
             then
               (* normal *)
               S754_finite is_neg m e
             else
               S754_zero is_neg
         else
           S754_infinity is_neg.

  Definition

  Definition of_Q (q : Q) : float

  Definition Qred_to_uint63 (num : N) (den : N) : (int * int) + float
    := let max_int := Z.to_N (Uint63.to_Z Uint63.max_int) in
       if ((num <=? max_int) && (den <=? max_int))%N
       then inl (Uint63.of_Z num, Uint63.of_Z den)
       else if (den =? 0)%N
            then inr ((if (num =? 0)%N then 0 else 1) / 0)
            else
              let '(num, den) := let q := Qred (inject_Z num / inject_Z den) in
                                 (Z.to_N (Qnum q), Npos (Qden q)) in
              if ((num <=? max_int) && (den <=? max_int))%N
              then inl (Uint63.of_Z num, Uint63.of_Z den)
              else if (max_int <? num / den)%N
                   then inr ∞
                   else let c := (1 + (N.max num den - 1) / max_int)%N in
                        let '(num, den) := (num / c, den / c)%N in
                        inl (Uint63.of_Z num, Uint63.of_Z den).

  Definition QredN_to_float (num : N) (den : N) : float
    := match Qred_to_uint63 num den with
       | inr f => f
       | inl (num, den) => PrimFloat.of_uint63 num / PrimFloat.of_uint63 den
       end.

  Definition Qred_to_float (q : Q) : float
    := if Qeq_bool q 0
       then 0
       else let num := if Qle_bool q 0
                       then Qnum (-q)
                       else Qnum q in
            QredN_to_float (Z.to_N num) (Npos (Qden q)).

  Definition of_N (v : N) : float
    := let e := N.log2 v in
       if (Uint63.to_Z Uint63.max_int <? (e + shift))%Z
       then ∞
       else ldshiftexp (QredN_to_float v (2^e)) (Uint63.of_Z (e + shift)).

  Definition of_Z (v : Z) : float
    := if (v <? 0)%Z
       then -of_N (Z.to_N (-v))
       else of_N (Z.to_N v).

  Definition of_Q_pos (q : Q) : float
    := let q := Qred q in
       let max_int := Z.to_N (Uint63.to_Z Uint63.max_int) in
       if ((Qnum q <=? max_int) && (Zpos (Qden q) <=? max_int))%Z
       then Qred_to_float q
       else let z := Qfloor q in
            let q := (q / z)%Q in
            of_Z z * Qred_to_float q.

  Definition of_Q (q : Q) : float
    := if Qeq_bool q 0
       then 0
       else if Qle_bool q 0
            then -of_Q_pos (-q)
            else of_Q_pos q.

  Definition to_Z_cps (x : float) {T} (on_nan : T) (on_pinf : T) (on_ninf : T) (on_Z : Z -> T) : T
    := match Prim2SF x with
       | S754_nan => on_nan
       | S754_zero _ => on_Z 0%Z
       | S754_infinity false => on_pinf
       | S754_infinity true => on_ninf
       | S754_finite is_neg m e
         => let z := (if (e <? 0)
                      then m / 2^(-e)
                      else m * 2^e)%Z in
            on_Z (if is_neg then -z else z)%Z
       end.

  (* float64 stores numbers 2.2E-308 to 1.7E+308, aka around 2^1024, so let's pick 2^2048 as inf *)
  Definition to_Z
    {nan : with_default "nan" Z 0%Z}
    {pinf : with_default "+∞" Z (2^2048)%Z}
    {ninf : with_default "-∞" Z (-(2^2048))%Z}
    (x : float)
    : Z
    := to_Z_cps x nan pinf ninf (fun x => x).

  Definition to_Q
    {nan : with_default "nan" Q 0%Q}
    {pinf : with_default "+∞" Q (2^2048)%Z}
    {ninf : with_default "-∞" Q (-(2^2048))%Z}
    (x : float)
    : Q
    := to_Q_cps x nan pinf ninf (fun x => x).

  #[local] Notation fmaf x y z := (x * y + z) (only parsing).

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
  Definition ln_ext (a : float) : float * float
    := let LOG2_HI   :=  0x1.62e430p-1 (* 6.93147182e-1f*) in
       let LOG2_LO   := -0x1.05c610p-29 (* -1.90465421e-9f *) in
       let SQRT_HALF :=  0x1.6a09e65dc27dfp-1 (* 0.70710678f *) in
       (* Reduce argument to m in [sqrt(0.5), sqrt(2.0)] *)
       let '(m, e) := Z.frexp a in
       let '(m, e) := if m <? SQRT_HALF
                      then (m + m, (e - 1)%Z)
                      else (m, e) in
       let i := of_Z e in
       (* Compute q = (m-1)/(m+1) as a double-float qhi:qlo *)
       let p := m + 1.0 in
       let m := m - 1.0 in
       let r := 1.0 / p in
       let qhi := r * m in
       let qlo := r * fmaf qhi (-m) (fmaf qhi (-2.0) m) in
       (* Approximate atanh(q), q in [sqrt(0.5)-1, sqrt(2)-1] *)
       let s := qhi * qhi in
       let r :=          0x1.08c000p-3 in (* 0.1293334961f *)
       let r := fmaf r s 0x1.22cd9cp-3 in (* 0.1419928074f *)
       let r := fmaf r s 0x1.99a162p-3 in (* 0.2000148296f *)
       let r := fmaf r s 0x1.555550p-2 in (* 0.3333332539f *)
       let t := fmaf qhi (qlo + qlo) (fmaf qhi qhi (-s)) in (* s:t = (qhi:qlo)**2 *)
       let p := s * qhi in
       let t := fmaf s qlo (fmaf t qhi (fmaf s qhi (-p))) in (* p:t = (qhi:qlo)**3 *)
       let s := fmaf r p (fmaf r t qlo) in
       let r := 2 * qhi in
       (* log(a) = 2 * atanh(q) + i * log(2) *)
       let t := fmaf   LOG2_HI  i r in
       let p := fmaf (-LOG2_HI) i t in
       let s := fmaf ( LOG2_LO) i (fmaf 2.0 s (r - p)) in
       let p := t + s in
       let loghi := p in (* normalize double-float result *)
       let loglo := (t - p) + s in
       (loghi, loglo).

  Definition ln (a : float) : float := fst (ln_ext a).

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
  Definition exp (a : float) : float :=
    (* exp(a) = 2**i * exp(f); i = rintf (a / log(2)) *)
    let big := 0x1.8p0 * of_Z (2^prec) in
    let j := fmaf 0x1.715476p0 a big - big in (* 1.442695f, 12582912f *)
    let f := fmaf j (-0x1.62e400p-1)  a in (* -6.93145752e-1f *) (* log_2_hi *)
    let f := fmaf j (-0x1.7f7d1cp-20) f in (* -1.42860677e-6f *) (* log_2_lo *)
    match to_Z_cps j (inr nan) (inr ∞) (inr (-∞)) (@inl _ _) with (* inl i or inr return *)
    | inr out_of_bounds => out_of_bounds
    | inl i =>
        (* approximate r = exp(f) on interval [-log(2)/2, +log(2)/2] *)
        let r :=          0x1.694000p-10 in (* 1.37805939e-3f *)
        let r := fmaf r f 0x1.125edcp-7  in (* 8.37312452e-3f *)
        let r := fmaf r f 0x1.555b5ap-5  in (* 4.16695364e-2f *)
        let r := fmaf r f 0x1.555450p-3  in (* 1.66664720e-1f *)
        let r := fmaf r f 0x1.fffff6p-2  in (* 4.99999851e-1f *)
        let r := fmaf r f 0x1.000000p+0  in (* 1.00000000e+0f *)
        let r := fmaf r f 0x1.000000p+0  in (* 1.00000000e+0f *)
        (* exp(a) = 2**i * r *)
        let r := Z.ldexp r i in
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
End PrimFloat.
