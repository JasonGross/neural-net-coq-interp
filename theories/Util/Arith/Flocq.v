From Coq.Reals Require Import Reals.
From Coq Require Import Lra Lia.
From Coq.Floats Require Import SpecFloat PrimFloat FloatAxioms.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead.
Local Open Scope bool_scope.

Section Binary.
  Context (prec emax : Z).
  Context (prec_gt_0_ : Prec_gt_0 prec).
  Context (prec_lt_emax_ : Prec_lt_emax prec emax).

  Notation emin := (emin prec emax).
  Notation fexp := (fexp prec emax).

  Theorem Bltb_correct_full (x y : binary_float prec emax)
    : Bltb x y = if is_finite x && is_finite y
                 then Rlt_bool (B2R x) (B2R y)
                 else if is_nan x || is_nan y
                      then false
                      else if negb (Bool.eqb (Bsign x) (Bsign y))
                           then Bsign x (* true iff x is neg *)
                           else
                             if negb (is_finite x) && negb (is_finite y)
                             then false
                             else xorb
                                    (Bsign x) (* negb iff both are neg *)
                                    (is_finite x) (* true iff x is finite, y is ∞ *).
  Proof.
    pose proof (Bltb_correct prec emax x y) as H.
    cbv [andb orb Bool.eqb negb xorb]; break_innermost_match.
    all: repeat specialize (H eq_refl).
    all: repeat match goal with H : false = true -> _ |- _ => clear H end.
    all: destruct_head'_and.
    all: try assumption.
    all: cbv [is_finite is_nan Bsign] in *.
    all: break_innermost_match_hyps; subst; try congruence.
    all: try reflexivity.
  Qed.

  Theorem B2R_Bminus m x y
    : B2R (Bminus m x y)
      = if is_finite x && is_finite y
        then if Rlt_bool (Rabs (round radix2 fexp (round_mode m) (B2R x - B2R y))) (bpow radix2 emax)
             then round radix2 fexp (round_mode m) (B2R x - B2R y)
             else SF2R radix2 (binary_overflow prec emax m (Bsign x))
        else 0%R.
  Proof using Type.
    pose proof (Bminus_correct prec emax prec_gt_0_ prec_lt_emax_ m x y) as H.
    cbv [andb]; break_innermost_match.
    all: repeat specialize (H eq_refl).
    all: repeat match goal with H : false = true -> _ |- _ => clear H end.
    all: destruct_head'_and.
    all: try assumption.
    all: try now rewrite <- SF2R_B2SF; congruence.
    all: cbv [is_finite] in *.
    all: break_innermost_match_hyps; try congruence.
    all: try reflexivity.
    all: cbv [Bminus]; break_innermost_match; reflexivity.
  Qed.

  Theorem B2R_Bdiv m x y
    : B2R (Bdiv m x y)
      = if negb (Req_bool (B2R y) 0%R)
        then if Rlt_bool (Rabs (round radix2 (SpecFloat.fexp prec emax) (round_mode m) (B2R x / B2R y))) (bpow radix2 emax)
             then
               round radix2 (SpecFloat.fexp prec emax) (round_mode m) (B2R x / B2R y)
             else
               SF2R radix2 (binary_overflow prec emax m (xorb (Bsign x) (Bsign y)))
        else 0%R.
  Proof using Type.
    pose proof (Bdiv_correct prec emax prec_gt_0_ prec_lt_emax_ m x y) as H.
    destruct (Req_bool_spec (B2R y) 0) as [H'|H'];
      [ clear H; rewrite H'; cbv [B2R] in H'
      | specialize (H ltac:(assumption)) ]; cbn [negb].
    all: break_innermost_match; break_innermost_match_hyps; destruct_head'_and.
    all: try assumption.
    all: try now rewrite <- SF2R_B2SF; congruence.
    all: try reflexivity.
    all: try now cbv [Bdiv]; break_innermost_match; reflexivity.
    cbv [Bdiv]; break_innermost_match; try reflexivity.
    cbv [Defs.F2R Defs.Fnum Defs.Fexp cond_Zopp] in H'.
    match goal with
    | [ H : (?x * ?y = 0)%R |- _ ]
      => assert (x = 0 \/ y = 0)%R by nra; clear H
    end.
    destruct_head'_or; break_innermost_match_hyps.
    all: lazymatch goal with
         | [ H : IZR ?x = 0%R |- _ ]
           => apply (eq_IZR x 0) in H
         | [ H : bpow ?r ?e = 0%R |- _ ]
           => pose proof (bpow_gt_0 r e)
         | _ => idtac
         end.
    all: try lia.
    lra.
  Qed.
End Binary.
