From Coq Require Import Qround Sint63 Uint63 NArith PArith ZArith QArith Floats.
Local Open Scope float_scope.
Notation "'∞'" := infinity : float_scope.
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion inject_Z : Z >-> Q.
Module PrimFloat.
  Definition of_sint63 (x : int) : float
    := if Sint63.ltb x 0
       then (-(PrimFloat.of_uint63 (-x)%sint63))%float
       else PrimFloat.of_uint63 x.

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

/* a**b = exp (b * log (a)), where a > 0, and log(a) is computed with extended
   precision as a double-float. Maxiumum error found across 2**42 test cases:
   1.97302 ulp @ (0.71162397, -256.672424).
*/
float my_powf_core (float a, float b)
{
    const float MAX_IEEE754_FLT = uint32_as_float (0x7f7fffff);
    const float EXP_OVFL_BOUND = 88.7228394f; // 0x1.62e430p+6f;
    const float EXP_OVFL_UNFL_F = 104.0f;
    const float MY_INF_F = uint32_as_float (0x7f800000);
    float lhi, llo, thi, tlo, phi, plo, r;

    /* compute lhi:llo = log(a) */
    my_logf_ext (a, &lhi, &llo);
    /* compute phi:plo = b * log(a) */
    thi = lhi * b;
    if (fabsf (thi) > EXP_OVFL_UNFL_F) { // definitely overflow / underflow
        r = (thi < 0.0f) ? 0.0f : MY_INF_F;
    } else {
        tlo = fmaf (lhi, b, -thi);
        tlo = fmaf (llo, b, +tlo);
        /* normalize intermediate result thi:tlo, giving final result phi:plo */
#if FAST_FADD_RZ
        phi = __fadd_rz (thi, tlo);// avoid premature ovfl in exp() computation
#else // FAST_FADD_RZ
        phi = thi + tlo;
        if (phi == EXP_OVFL_BOUND){// avoid premature ovfl in exp() computation
#if PORTABLE
            phi = nextafterf (phi, 0.0f);
#else // PORTABLE
            phi = uint32_as_float (float_as_uint32 (phi) - 1);
#endif // PORTABLE
        }
#endif // FAST_FADD_RZ
        plo = (thi - phi) + tlo;
        /* exp'(x) = exp(x); exp(x+y) = exp(x) + exp(x) * y, for |y| << |x| */
        r = my_expf_unchecked (phi);
        /* prevent generation of NaN during interpolation due to r = INF */
        if (fabsf (r) <= MAX_IEEE754_FLT) {
            r = fmaf (plo, r, r);
        }
    }
    return r;
}

float my_powf (float a, float b)
{
    const float MY_INF_F = uint32_as_float (0x7f800000);
    const float MY_NAN_F = uint32_as_float (0xffc00000);
    int expo_odd_int;
    float r;

    /* special case handling per ISO C specification */
    expo_odd_int = fmaf (-2.0f, floorf (0.5f * b), b) == 1.0f;
    if ((a == 1.0f) || (b == 0.0f)) {
        r = 1.0f;
    } else if (isnan (a) || isnan (b)) {
        r = a + b;  // convert SNaN to QNanN or trigger exception
    } else if (isinf (b)) {
        r = ((fabsf (a) < 1.0f) != (b < 0.0f)) ? 0.0f :  MY_INF_F;
        if (a == -1.0f) r = 1.0f;
    } else if (isinf (a)) {
        r = (b < 0.0f) ? 0.0f : MY_INF_F;
        if ((a < 0.0f) && expo_odd_int) r = -r;
    } else if (a == 0.0f) {
        r = (expo_odd_int) ? (a + a) : 0.0f;
        if (b < 0.0f) r = copysignf (MY_INF_F, r);
    } else if ((a < 0.0f) && (b != floorf (b))) {
        r = MY_NAN_F;
    } else {
        r = my_powf_core (fabsf (a), b);
        if ((a < 0.0f) && expo_odd_int) {
            r = -r;
        }
    }
    return r;
}
>>>
*)
End PrimFloat.
