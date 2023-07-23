From Coq Require Import Lqa Qabs Qround Morphisms Lia Qround Sint63 Uint63 NArith PArith ZArith QArith Floats.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel.
From NeuralNetInterp.Util.Arith Require Import FloatArith.Definitions QArith.
From NeuralNetInterp.Util.Tactics Require Import DestructHead BreakMatch UniquePose.
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
  Lemma snd_frshiftexp_spec : forall f, Uint63.to_Z (snd (frshiftexp f)) = (snd (SFfrexp prec emax (Prim2SF f)) + shift)%Z.
  Proof. intro f; generalize (frshiftexp_spec f); destruct frshiftexp; intros <-; cbn [snd]; lia. Qed.
  Lemma frshiftexp_spec_eta : forall f, let (m,e) := eta2 (frshiftexp f) in (Prim2SF m, ((Uint63.to_Z e) - shift)%Z) = SFfrexp prec emax (Prim2SF f).
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

  #[export] Instance Q2SF_Proper : Proper (Qeq ==> eq) Q2SF.
  Proof.
    intros x y H.
    cbv [Q2SF]; rewrite H.
    repeat match goal with
           | [ H : ?x == ?y, H' : context[?x] |- _ ] => rewrite !H in H'; rewrite ?H' in *
           | _ => break_innermost_match_step
           | [ H : ?x == ?y |- context[?x] ] => rewrite H
           | [ H : ?x == ?y |- _ ] => is_var x; clear x H
           end.
    all: reflexivity.
  Qed.


  #[export] Instance of_Q_Proper : Proper (Qeq ==> eq) of_Q.
  Proof. cbv [of_Q]; now intros ?? ->. Qed.

  Local Ltac make_cases _ :=
    lazymatch goal with
    | [ |- (?P /\ ?Q) \/ _ ]
      => let H := fresh in
         assert (H : P \/ ~P) by lra;
         destruct H as [H|H];
         [ left; split; [ assumption | ]
         | right; make_cases () ]
    | _ => idtac
    end.
  Lemma Q2SF_classify q
    : (q == 0 /\ Q2SF q = S754_zero false)
      \/ ((((-1) / (2^(-(emin-1)))%Z) <= q < 0)%Q /\ Q2SF q = S754_zero true)
      \/ ((q <= -(2^emax)%Z)%Q /\ Q2SF q = S754_infinity true)
      \/ (((2^emax)%Z <= q)%Q /\ Q2SF q = S754_infinity false)
      \/ ((-(2^emax)%Z < q < (2^emax)%Z)%Q /\ forall m e, Q2SF q = S754_finite (negb (Qle_bool 0 q)) m e).
  Proof.
    make_cases ().
    { rewrite H; reflexivity. }
    all: cbv [Q2SF]; rewrite ?Qabs_alt; break_innermost_match.
    all: rewrite ?Qeq_bool_iff, ?Qle_bool_iff in *.
    all: vm_compute emin in *.
    all: vm_compute emax in *.
    all: cbv [prec emax] in *.
    all: repeat first [ progress destruct_head'_and
                      | match goal with
                        | [ H : (_ * _ == 0)%Q |- _ ] => apply Qmult_integral in H; destruct H
                        | [ H : Qle_bool ?x ?y = false |- _ ]
                          => pose proof (@Qle_bool_iff x y); rewrite H in *; clear H
                        | [ H : Qeq_bool ?x ?y = false |- _ ] => apply Qeq_bool_neq in H
                        | [ H : false = true <-> ?P |- _ ]
                          => assert (~P) by (rewrite <- H; congruence); clear H
                        | [ H : ~?x <= ?y |- _ ] => assert (y < x) by lra; clear H
                        | [ H : inject_Z ?x == 0 |- _ ]
                          => assert (x = 0)%Z by (now apply inject_Z_injective); clear H
                        | [ H : (_^_ = 0)%Z |- _ ]
                          => rewrite Z.pow_eq_0_iff in H
                        | [ H : (Z.log2 ?x + _ - _ < 0)%Z |- _ ]
                          => clear -H; pose proof (Z.log2_nonneg x); lia
                        | [ H : Z.leb _ _ = true |- _ ] => rewrite Z.leb_le in H
                        end
                      | progress destruct_head'_or
                      | lra
                      | lia
                      | reflexivity ].
    lazymatch goal with
    | [ H : ((-1)/inject_Z (2^?x) <= ?q)%Q |- _ ]
      => move H at bottom; cbv [Qle Qopp Qdiv Qnum Qden inject_Z Qmult] in H;
         let xv := (eval cbv in x) in
         change x with xv in H;
         change (Qinv ((2^xv) # 1)) with (1 # (Z.to_pos (2^xv))) in H;
         cbv beta iota in H
    end.
  Admitted.

  Lemma to_of_Q' q {nan pinf ninf nzero}
    : @to_Q nan pinf ninf nzero (of_Q q)
      == if (Qle_bool ((-1) / (2^(-(emin-1)))%Z) q) && negb (Qle_bool 0 q)
         then nzero
         else if Qle_bool q (-(2^emax)%Z)
              then ninf
              else if Qle_bool (2^emax)%Z q
                   then pinf
                   else q.
  Proof.
    cbv [to_Q to_Q_cps of_Q].
    rewrite Prim2SF_SF2Prim.
    pose proof (Q2SF_classify q); repeat (destruct_head'_or; destruct_head'_and).
    all: generalize dependent (Q2SF q); intros; subst.
    { match goal with
      | [ H : q == _ |- _ ] => rewrite !H; cbv -["=="]; lra
      end. }
    all: cbv [andb negb]; break_innermost_match.
    all: repeat match goal with
           | [ H : Qle_bool _ _ = true |- _ ] => rewrite Qle_bool_iff in H
           | [ H : Qeq_bool _ _ = true |- _ ] => rewrite Qeq_bool_iff in H
           | [ H : Qle_bool ?x ?y = false |- _ ]
             => pose proof (@Qle_bool_iff x y); rewrite H in *; clear H
           | [ H : Qeq_bool ?x ?y = false |- _ ] => apply Qeq_bool_neq in H
           | [ H : false = true <-> ?P |- _ ]
             => assert (~P) by (rewrite <- H; congruence); clear H
           | [ H : ~?x <= ?y |- _ ] => assert (y < x) by (clear -H; lra); clear H
           end.
    all: try lra.
    all: repeat match goal with
           | [ H : ?q <= ?x, H' : ?y <= ?q |- _ ]
             => unique assert (y <= x) by lra
           | [ H : ?q < ?x, H' : ?y <= ?q |- _ ]
             => unique assert (y < x) by lra
           end.
    all: try match goal with
           | [ H : _ |- _ ]
             => exfalso; revert H; clear;
                lazymatch goal with
                | [ H : _ |- _ ] => fail
                | _ => idtac
                end;
                vm_compute; congruence
           | [ H : forall m e, _ = _ :> spec_float |- _ ]
             => exfalso; clear -H; epose (H _ _); discriminate
           end.
    3: {
    2: {
    vm_compute in H2.
    vm_compute in H1, H2; congruence.
                assert (H : y < x) by lra;
                vm_compute in H; specialize (H eq_refl); exfalso; now apply H
           end.
    match goal with
    | [ H : ?q < ?x, H' : ?y <= ?q |- _ ]
      => let H := fresh in
         assert (H : y < x) by lra;
         vm_compute in H; specialize (H eq_refl); exfalso; now apply H
    end.
    vm_compute in H0.
    vm_compute Z.pow in *.


    rewrite !H; cbv -["=="]; lra.
    vm_compute negb.
    rewrite Bool.andb_false_r.
    vm_compute.
    all: match goal with H : Q2SF q = _ |- _ =>
    cbv [Q2SF andb negb]; break_innermost_match.
    all: rewrite ?Qle_bool_iff in *.
    all: rewrite ?Qabs_pos in * by assumption.
    all: rewrite ?Qeq_bool_iff in *.

    2: { assumption.
        if is_nan f
        then of_Q nan
        else if is_infinity f
             then if f <? 0 then of_Q ninf else of_Q pinf
             else if is_zero f && get_sign f
                  then of_Q nzero
                  else f.

  Lemma of_to_Q' f {nan pinf ninf nzero}
    : of_Q (@to_Q nan pinf ninf nzero f)
      = if is_nan f
        then of_Q nan
        else if is_infinity f
             then if f <? 0 then of_Q ninf else of_Q pinf
             else if is_zero f && get_sign f
                  then of_Q nzero
                  else f.
  Proof.
    cbv [to_Q to_Q_cps is_infinity is_nan is_zero].
    pose proof (SF2Prim_Prim2SF f) as H.
    destruct (Prim2SF f) eqn:H'; subst.
    1-3: destruct_head'_bool; cbv -[of_Q]; reflexivity.
    { cbv [get_sign is_zero zero].
      repeat autorewrite with float_spec.
      rewrite !H'.
      vm_compute (Prim2SF 0).
      vm_compute (Prim2SF ∞).
      cbv [SFabs SFeqb SFltb SFcompare].
      autorewrite with float_spec.
      rewrite !Z.compare_refl.
      rewrite Pos.compare_cont_refl.
      cbv [CompOpp].
      etransitivity;
        [
        | let s := lazymatch goal with s : bool |- _ => s end in
          pose s as sv; destruct s; cbn [negb];
          rewrite !H', ?Bool.andb_false_l;
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
