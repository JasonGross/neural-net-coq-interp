From Coq Require Import Morphisms Lia Qround Sint63 Uint63 NArith PArith ZArith QArith Floats.
From NeuralNetInterp.Util Require Import Default.
From NeuralNetInterp.Util.Arith Require Import FloatArith.Definitions.
From NeuralNetInterp.Util.Tactics Require Import DestructHead BreakMatch.
Local Open Scope float_scope.
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion inject_Z : Z >-> Q.
#[local] Coercion Z.pos : positive >-> Z.
#[local] Coercion is_true : bool >-> Sortclass.

Create HintDb float_spec discriminated.
Create HintDb float_spec_rev discriminated.

Module PrimFloat.
  Export FloatArith.Definitions.PrimFloat.

  #[local] Notation eta2 x := (fst x, snd x).
  Lemma fst_frshiftexp_spec : forall f, Prim2SF (fst (frshiftexp f)) = fst (SFfrexp prec emax (Prim2SF f)).
  Proof. intro f; generalize (frshiftexp_spec f); destruct frshiftexp; intros <-; reflexivity. Qed.
  Lemma snd_frshiftexp_spec : forall f, to_Z (snd (frshiftexp f)) = (snd (SFfrexp prec emax (Prim2SF f)) + shift)%Z.
  Proof. intro f; generalize (frshiftexp_spec f); destruct frshiftexp; intros <-; cbn [snd]; lia. Qed.
  Lemma frshiftexp_spec_eta : forall f, let (m,e) := eta2 (frshiftexp f) in (Prim2SF m, ((to_Z e) - shift)%Z) = SFfrexp prec emax (Prim2SF f).
  Proof. intro f; generalize (frshiftexp_spec f); destruct frshiftexp; intros <-; reflexivity. Qed.

  #[export] Hint Rewrite
    opp_spec
    abs_spec

    eqb_spec
    ltb_spec
    leb_spec

    compare_spec

    classify_spec
    mul_spec
    add_spec
    sub_spec
    div_spec
    sqrt_spec

    of_uint63_spec
    normfr_mantissa_spec

    fst_frshiftexp_spec
    snd_frshiftexp_spec
    ldshiftexp_spec

    next_up_spec
    next_down_spec
  : float_spec.
  #[export] Hint Rewrite <-
    opp_spec
    abs_spec

    eqb_spec
    ltb_spec
    leb_spec

    compare_spec

    classify_spec
    mul_spec
    add_spec
    sub_spec
    div_spec
    sqrt_spec

    of_uint63_spec
    normfr_mantissa_spec

    frshiftexp_spec_eta
    ldshiftexp_spec

    next_up_spec
    next_down_spec
  : float_spec_rev.

  #[export] Instance of_Q_pos_Proper : Proper (Qeq ==> eq) of_Q_pos.
  Proof.
    intros x y H.
    cbv [of_Q_pos].
    rewrite !(Qred_complete _ _ H); reflexivity.
  Qed.

  Lemma of_to_Q f {nan pinf ninf}
    : of_Q (@to_Q nan pinf ninf f)
      = if is_nan f
        then of_Q nan
        else if is_infinity f
             then if f <? 0 then of_Q ninf else of_Q pinf
             else if is_zero f
                  then abs f
                  else f.
  Proof.
    cbv [to_Q to_Q_cps is_infinity is_nan is_zero].
    pose proof (SF2Prim_Prim2SF f) as H.
    destruct (Prim2SF f) eqn:H'; subst.
    { destruct_head'_bool; vm_compute; reflexivity. }
    { destruct_head'_bool; cbv -[of_Q]; reflexivity. }
    { cbv -[of_Q]; reflexivity. }
    { repeat autorewrite with float_spec.
      rewrite !H'.
      vm_compute Prim2SF.
      cbv [SFabs SFeqb SFltb SFcompare].
      rewrite !Z.compare_refl.
      rewrite Pos.compare_cont_refl.
      cbv [CompOpp].
      etransitivity;
        [
        | let s := lazymatch goal with s : bool |- _ => s end in
          pose s as sv; destruct s; cbn [negb];
          let sv := (eval cbv in sv) in
          set (s := sv); reflexivity ].
      apply Prim2SF_inj; rewrite H'.
      cbv [of_Q SF2Prim Qeq_bool Qle_bool Zeq_bool].
      destruct_head'_bool; break_innermost_match.
      all: cbn [Qnum Qden Qopp Qdiv] in *.
      all: rewrite ?Z.mul_0_l, ?Z.mul_1_r, ?Z.ltb_lt, ?Z.compare_eq_iff, ?Z.compare_lt_iff, ?Z.compare_ge_iff, ?Z.compare_gt_iff, ?Z.leb_le in *.
      all: cbv [Qnum Qdiv Qinv inject_Z Qmult Qden] in *.
      all: break_innermost_match_hyps.
      all: rewrite ?Z.mul_1_r in *.
      all: rewrite ?Z.pow_eq_0_iff in *.
      all: destruct_head'_or; try lia.
      all: rewrite ?Qopp_opp.
      all: rewrite ?Z.opp_neg_pos, ?Z.opp_nonpos_nonneg in *.
      all: cbv [inject_Z Qmult].
      all: rewrite ?Pos.mul_1_l.
      Set Printing All.
      Search (-_ <= 0)%Z.
      pose proof (fun q => Qopp_opp q).
      Search (- - _)%Q.
      rewrite Q_op
      rewrite  in *.
      Z.compare_lt

        pose Z.compare_eq_iff.
        Search (_^_ = 0)%Z iff.
        all: match goal with
             | [ H : (

        Search (2^_ <> 0)%Z.
        all: rewrite ?Z.mul_0_r in *.
        cbn [Z.opp Z.compare Z.leb] in *.
        rewrite  in *.
        match goal with
        | [ H : (
        Search (2^_ = 0)%Z.
        cbn in *.
        al
      cbn [Qnum
      Print Qeq_bool.

      Search Z.ldexp.
           all: apply f_equal.
           all: apply f_equal3.

      cbn [SFeqb].
      rewrite Prim2SF_SF2Prim.
      2: { cbv [SpecFloat.valid_binary bounded].
      Search Prim2SF SF2Prim.
    Search Prim2SF.
      then @to_Q nan pinf ninf (of_Q f) = if is_nan f
                                     then nan
                                     else
                                       if is_infinity f
                                       then if f <? 0 then ninf else pinf
                                       else f.
  Lemma to_of_Q f {nan pinf ninf}
    : @to_Q nan pinf ninf (of_Q f) = if is_nan f
                                     then nan
                                     else
                                       if is_infinity f
                                       then if f <? 0 then ninf else pinf
                                       else f.

End PrimFloat.
Export (hints) PrimFloat.
