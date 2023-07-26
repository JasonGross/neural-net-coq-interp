From Coq.Reals Require Import Reals.
From Coq Require Import Lra Lia Eqdep_dec.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util.Arith Require Import FloatArith.Definitions.
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

Local Existing Instances Hprec Hmax.

Lemma fma_equiv x y z : Prim2B (PrimFloat.fmaf x y z) = Bfma mode_NE (Prim2B x) (Prim2B y) (Prim2B z).
Proof.
  cbv [PrimFloat.fmaf PrimFloat.SF64fma PrimFloat.SFfma Bfma Bfma_szero Bsign].
  cbv [Prim2B].
  repeat match goal with |- context[Prim2SF_valid ?x] => generalize (Prim2SF_valid x) end.
  destruct (Prim2SF x), (Prim2SF y), (Prim2SF z); clear x y z.
  all: try reflexivity.
  all: let check x :=
         lazymatch x with
         | S754_infinity _ => idtac
         | S754_zero _ => idtac
         | S754_nan => idtac
         end in
       match goal with
       | [ |- context[SFmul _ _ ?x ?y] ]
         => first [ check x | check y ]; cbn [SFmul]
       | _ => idtac
       end.
  all: cbn [SFadd].
  all: repeat match goal with
         | [ |- context[if ?b then ?f ?x else ?f ?y] ]
           => replace (if b then f x else f y) with (f (if b then x else y)) by now destruct b
         end.
  all: intros.
  all: repeat match goal with H : valid_binary (Prim2SF _) = true |- _ => revert H end.
  all: rewrite ?Prim2SF_SF2Prim by ((idtac + destruct_head'_bool); assumption).
  all: intros.
  all: repeat match goal with H : valid_binary (Prim2SF _) = true |- _ => revert H end.
  all: try reflexivity.
  all: repeat match goal with
         | [ H1 : ?x = true, H2 : ?x = true |- _ ]
           => assert (H1 = H2)
             by (generalize dependent x; clear; intros; subst; (idtac + symmetry); apply UIP_refl_bool);
              subst H2
         end.
  all: try reflexivity.
  all: try (cbv [Bool.eqb SF2B] in *; break_innermost_match; reflexivity).
  all: cbn [SF2B].
  all: change (SFadd prec emax) with SF64add.
  all: change (SFmul prec emax) with SF64mul.
  lazymatch goal with
  | [ |- context[SF64add (SF64mul ?x ?y) ?z] ]
    => rewrite <- (Prim2SF_SF2Prim x), <- (Prim2SF_SF2Prim y), <- (Prim2SF_SF2Prim z), <- mul_spec, <- add_spec, ?SF2Prim_Prim2SF by assumption
  end.
  all: lazymatch goal with
       | [ |- context[SF2B (Prim2SF ?x)] ]
         => let e := fresh "e" in
            intro e;
            assert (e = Prim2SF_valid _)
              by (generalize (Prim2SF_valid x);
                  clear; generalize dependent (valid_binary (Prim2SF x));
                  clear; intros; subst;
                  (idtac + symmetry); apply UIP_refl_bool);
            subst e;
            change (SF2B (Prim2SF x) (Prim2SF_valid x)) with (Prim2B x)
       | _ => idtac
       end.
  all: try rewrite add_equiv, mul_equiv.
  { cbv [Prim2B].
    repeat match goal with |- context[Prim2SF_valid ?x] => generalize (Prim2SF_valid x) end.
    rewrite ?Prim2SF_SF2Prim by ((idtac + destruct_head'_bool); assumption).
    intros.
    all: repeat match goal with
           | [ H1 : ?x = true, H2 : ?x = true |- _ ]
             => assert (H1 = H2)
               by (generalize dependent x; clear; intros; subst; (idtac + symmetry); apply UIP_refl_bool);
                subst H2
           end.
    cbv [SF2B].
    destruct_head'_bool;
      cbv [Bmult Bplus Operations.Fmult binary_normalize cond_Zopp Z.mul Z.opp].
    cbv [binary_round].
    cbn [xorb].
    all: shelve. }
  { cbv [Prim2B].



  cbv [Prim2B].
  Search Prim2B SF2Prim.
  cbv [Bplus Bmulkt


  Check
  change (SF2B (Prim2SF ?x)) with (Prim2B x (valid_binary_B2SF _))).
  Check add_equiv.
  rewrite .

  rewrite Prim2SF_SF2Prim.
  2: { Check valid_binary_B2SF.

  all: rewrite ?binary_round_aux_equiv, ?binary_normalize_equiv.
  all: repeat change (SF2Prim (B2SF ?x)) with (B2Prim x).
  all: rewrite ?Prim2SF_B2Prim.
  Check Prim2SF_valid.
  match goal with
  | [ |- context[valid_binary (Prim2SF (SF2Prim ?x))] ]
    => rewrite <- (B2SF_SF2B prec emax (Prim2SF (SF2Prim x)) (Prim2SF_valid _));
       generalize dependent (
  end.
  Print Prim2B.
  Search Prim2SF SF2B.
  Print B2SF.

  Search SF2B.
  Search (_ * _)%float.
  Search Prim2SF SF2Prim.
  rewrite SFadd_
  Check SFadd_
  cbn [SFmul].
  all: cbv [valid_binary bounded canonical_mantissa] in * |- .
  all: rewrite Bool.andb_true_iff in *.
  all: destruct_head'_and.
  all: repeat match goal with H : Zeq_bool _ _ = true |- _ => apply Zeq_bool_eq in H end.
  all: cbv [binary_normalize binary_round Operations.Fmult cond_Zopp Z.mul Z.opp Operations.Fplus Operations.Falign].
  all: destruct_head'_bool.
  all: cbn [xorb Bool.eqb].
  match goal with
  | [ |- context[shl_align_fexp ?prec ?emax ?m ?e] ]
    => replace (shl_align_fexp prec emax m e) with (m, e)
  end.
  2: { cbv [shl_align_fexp shl_align fexp] in *.
       rewrite Z.max_l.
       Search
       change digits2_pos with Pos.size in *.

       Zify.zify.
       Locate Ltac zify.
       rewrite Z.max_l.
       2: { match goal with
         | [ |- context[Z.pos (
       2: change (Z.pos (digits2_pos ?x)) with (Z.log2 (Z.pos x~0)) in *.
       2: {
         rewrite Pos2Z.inj_mul in *.
         Search Z.pos Pos.mul.

  match goal with

  Check shl_align_fexp_correct.
  cbn [Z.mul].
  all: cbv
  cbn.
  cbv [binary_normalize binary_round shl_align_fexp].
  Print shl_align.
  rewrite Prim2SF_SF2Prim.
  2: {
       Search shl_align.
       Search binary_normalize.
       cbv [Operations.Fplus Operations.Fmult Operations.Falign].
       Search binary_normalize.
       Search B2SF valid_bin
       Search Prim2SF B2Prim
  2: { Search SpecFloat.binary_normalize.

  1-2:rewrite Prim2SF_SF2Prim by (destruct_head'_bool; assumption).
  1-2:reflexivity.

  rewrote
  64: { cbv [Operations.Fmult].
        cbv [Operations.Fplus].
        cbv [Operations.Falign].
  all: cbv [Bool.eqb xorb]; break_innermost_match.
  rewrite Prim2SF_SF2Prim
  all: intros.
  rewrite Prim2SF_SF2Prim.

  cbv [Prim
  destruct (Prim2SF
  cbv [SF2B].
  break_innermost_match_step.

  Print Prim2B.
  Search Prim2B SF2Prim.
