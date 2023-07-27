From Coq.Reals Require Import Reals.
From Coq Require Import Lra Lia Eqdep_dec.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util.Arith Require Import FloatArith.Definitions.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose.
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
    cbv [Operations.Fmult].
    cbv [Bplus].
    cbv [Bmult].
    set (pf := proj1 _); clearbody pf; revert pf.
    cbv [cond_Zopp Z.opp Z.mul].
    cbv [binary_normalize].
    repeat match goal with
           | [ |- context[match (match ?b with true => ?x | false => ?y end) with Z0 => ?Z | Z.pos p => @?P p | Z.neg n => @?N n end] ]
             => replace (match (match b with true => x | false => y end) with Z0 => Z | Z.pos p => P p | Z.neg n => N n end)
               with (if b then match x with Z0 => Z | Z.pos p => P p | Z.neg n => N n end else match y with Z0 => Z | Z.pos p => P p | Z.neg n => N n end)
               by now destruct b
           | [ |- context[if ?s then if ?s' then ?a else ?b else if ?s' then ?b else ?a] ]
             => replace (if s then if s' then a else b else if s' then b else a)
               with (if xorb s s' then b else a)
               by now destruct s, s'
           end.
    repeat match goal with
           | [ |- context[?v] ]
             => lazymatch v with
                | (if ?b
                   then ?f (?g ?true ?x ?y) (proj1 (?h ?true' ?z ?w))
                   else ?f (?g ?false ?x ?y) (proj1 (?h ?false' ?z ?w)))
                  => replace v
                    with (f (g (if b then true else false) x y) (proj1 (h (if b then true' else false') z w)))
                    by now destruct b
                end
           | [ |- context[if ?b then true else false] ]
             => replace (if b then true else false) with b by now destruct b
           | [ |- context[if ?b then false else true] ]
             => replace (if b then false else true) with (negb b) by now destruct b
           end.
    set (pf' := proj1 _); clearbody pf'; revert pf'.
    cbv [binary_round] in *.
    cbv [shl_align_fexp shl_align].
    repeat match goal with
           | [ |- context[match (match ?b with Z.neg n => @?N n | Z0 => ?Z | Z.pos p => @?P p end) with pair A B => @?PP A B end] ]
             => replace (match (match b with Z.neg n => @N n | Z0 => Z | Z.pos p => @P p end) with pair A B => @PP A B end)
               with (PP (match b with Z.neg n => fst (N n) | Z0 => fst Z | Z.pos p => fst (P p) end)
                       (match b with Z.neg n => snd (N n) | Z0 => snd Z | Z.pos p => snd (P p) end))
               by now destruct b
           end.
    cbn [fst snd].
    cbv [Bool.eqb].
    cbv [valid_binary bounded canonical_mantissa] in * |- .
    rewrite Bool.andb_true_iff, Z.leb_le in *.
    destruct_head'_and.
    repeat match goal with H : Zeq_bool _ _ = true |- _ => apply Zeq_bool_eq in H end.
    repeat match goal with
           | [ |- context[?v] ]
             => lazymatch v with
                | (if ?b
                   then ?f (?g ?true ?x ?y) (proj1 (?h ?true' ?z ?w))
                   else ?f (?g ?false ?x ?y) (proj1 (?h ?false' ?z ?w)))
                  => replace v
                    with (f (g (if b then true else false) x y) (proj1 (h (if b then true' else false') z w)))
                    by now destruct b
                end
           | [ |- context[if ?b then true else false] ]
             => replace (if b then true else false) with b by now destruct b
           | [ |- context[if ?b then false else true] ]
             => replace (if b then false else true) with (negb b) by now destruct b
           end.
    match goal with
    | [ |- context[fexp ?prec ?emax (?m + ?e)] ]
      => assert (0 <= fexp prec emax (m + e) - e)%Z
    end.
    { cbv [fexp] in *.
      all: change digits2_pos with Pos.size in *.
      repeat match goal with H : context[Z.max] |- _ => revert H end.
      all: repeat match goal with
             | [ |- context[Z.pos (Pos.size ?x)] ]
               => let k := fresh in
                  set (k := Z.pos (Pos.size x)) in *;
                  let H' := fresh in
                  assert (H' : (k = Z.log2 (Z.pos x) + 1)%Z)
                    by (clear; subst k; cbn; break_innermost_match; cbn; lia);
                  first [ clearbody k; subst k
                        | rewrite ?H' in *; clear H'; subst k ]
             | [ |- context[Z.log2 (Z.pos (?x * ?y))] ]
               => change (Z.log2 (Z.pos (x * y))) with (Z.log2 (Z.pos x * Z.pos y)) in *;
                  pose proof (Z.log2_mul_below (Z.pos x) (Z.pos y) ltac:(lia) ltac:(lia));
                  pose proof (Z.log2_mul_above (Z.pos x) (Z.pos y) ltac:(lia) ltac:(lia))
             | [ |- context[Z.log2 ?x] ]
               => unique pose proof (Z.log2_nonneg x)
             end.
      repeat apply Z.max_case_strong.
      all: intros.
      all: subst.
      all: repeat match goal with
             | [ H : (Z.log2 ?x + ?y + ?z - ?w = ?z)%Z |- _ ]
               => assert (Z.log2 x = w - y)%Z by lia; clear H
             | [ H : Z.log2 _ = _ |- _ ] => progress rewrite ?H in *
             | [ H : (?x <= ?y)%Z, H' : (?y <= ?x + 1)%Z |- _ ]
               => assert (y = x \/ y = x + 1)%Z by lia; clear H H'
             | _ => progress destruct_head'_or
             end.
      all: try lia.
      all: try (vm_compute emin in *; vm_compute emax in *; cbv [prec emax] in *; lia). }
    repeat match goal with
           | [ |- context[?v] ]
             => lazymatch v with
                | match (fexp ?prec ?emax (?m + ?e) - ?e)%Z with Z.neg n => @?N n | Z0 => ?z | Z.pos _ => ?z end
                  => replace v with z
                    by (destruct (fexp prec emax (m + e) - e)%Z eqn:?; try lia)
                end
           end.
    intros.
    repeat match goal with
           | [ H1 : ?x = true, H2 : ?x = true |- _ ]
             => assert (H1 = H2)
               by (clear; generalize dependent x; clear; intros; subst; (idtac + symmetry); apply UIP_refl_bool);
                subst H2
           end.
    break_innermost_match; try reflexivity.
    cbv [binary_round_aux binary_fit_aux binary_overflow overflow_to_inf SF2B] in *.
    repeat (break_innermost_match_hyps_step; []).
    repeat match goal with
           | [ H : context[match ?x with _ => _ end] |- _ ]
             => destruct x eqn:?; subst; try (break_innermost_match_hyps; congruence); []
           end.
    repeat match goal with
           | [ H : S754_zero _ = S754_zero _ |- _ ] => inversion H; clear H
           | [ H : B754_zero _ = B754_zero _ |- _ ] => inversion H; clear H
           end.
    subst.
    repeat first [ progress subst
                 | match goal with
                   | [ H : (_, _) = (_, _) |- _ ]
                     => pose proof (f_equal (@fst _ _) H);
                        pose proof (f_equal (@snd _ _) H);
                        clear H; cbn [fst snd] in *
                   | [ H : context[shr_fexp] |- _ ]
                     => rewrite shr_fexp_truncate in H by (cbv [choice_mode Round.cond_incr]; break_innermost_match; (exact _ + lia))
                   end
                 | break_innermost_match_hyps_step ].
    rewrite ?loc_of_shr_record_of_loc, ?shr_m_shr_record_of_loc in *.
    cbv [Round.truncate] in *.
    rewrite <- Digits.Zdigits2_Zdigits in *.
    cbv [Zdigits2] in *.
    break_innermost_match_hyps; rewrite ?Z.ltb_lt, ?Z.ltb_ge in *.
    cbv [Round.truncate_aux] in *.
    all: repeat first [ progress subst
                      | match goal with
                        | [ H : (_, _) = (_, _) |- _ ]
                          => pose proof (f_equal (@fst _ _) H);
                             pose proof (f_equal (@snd _ _) H);
                             clear H; cbn [fst snd] in *
                        | [ H : context[shr_fexp] |- _ ]
                          => rewrite shr_fexp_truncate in H by (cbv [choice_mode Round.cond_incr]; break_innermost_match; (exact _ + lia))
                        end
                      | break_innermost_match_hyps_step ].


    all: cbv [Round.truncate] in *.
    all: rewrite <- ?Digits.Zdigits2_Zdigits in *.
    all: cbv [Zdigits2] in *.
    2: { cbv [choice_mode Round.cond_incr] in *.

         rewrite shr_fexp_truncate in Heqp. by (cbv [choice_mode Round.cond_incr]; break_innermost_match; (exact _ + lia))
    Search Digits.Zdigits.
    cbv [Digits.Zdigits].
    Print Round.truncate.
    Search Round.truncate.
    match goal with
    | [ H : context[shr_fexp] |- _ ]
      => rewrite shr_fexp_truncate in H; try exact _
    end.
    2: { cbv [choice_mode Round.cond_incr].
    Print choice_mode.
    Search shr_record_of_loc.
    cbv [shr_record_of_loc] in *.
    2: exact _.
    Search shr_fexp.
    cbv [shr_fexp] in *.
    cbv [shr] in *.
    cbv [Zdigits2] in *.
Search
    match goal with
           | [ H : (0 <= ?x)%Z, H' : context[?v] |- _ ]
             => lazymatch v with
                | match ?x with Z.neg n => @?N n | Z0 => ?z | Z.pos _ => ?z end
                  => replace v with z
                    by (destruct x eqn:?; try lia)
                end
           end.
    cbv [loc_of_shr_record] in *.
    cbv [choice_mode] in *.
    cbv [Round.cond_incr] in *.
    cbv [shr_fexp] in *.
    cbv [shr_record_of_loc] in *.
    cbv [shr] in *.
    cbn in *.
    destruct_head shr_record.
    break_innermost_match_hyps_step; try congruence.
    2: { break_innermost_match_hyps_step.
    break_innermost_match_hyps_step
    break_innermost_match_hyps; try congruence.
    2: {
    cbn in *.
    brea
    break_match_hyps.
    break_innermost_match; try reflexivity.
    2: {

    2: { match goal with
           | [ H : (Z.log2 ?x + ?y + ?z - ?w <= ?k)%Z |- _ ]
             => assert (Z.log2 x <= k + w - z - y)%Z by lia; clear H
           end.
              match goal with
              | [ H : (?x <= ?y)%Z, H' : (?y <= ?x + 1)%Z |- _ ]
                => assert (y = x \/ y = x + 1)%Z by lia; clear H H'
              end.
              destruct_head'_or.
              all: vm_compute emin in *; vm_compute emax in *; cbv [prec emax] in *.
              all: try lia.
              all: rewrite H in *.
              all: repeat match goal with H : _ |- _ => progress ring_simplify in H end.
              lazymatch goal with
              | [ H : (?x <= - ?y - ?z - ?w)%Z |- _ ]
                => assert (y + z <= -w - x)%Z by lia; clear H
              end.
              all: repeat match goal with H : _ |- _ => progress ring_simplify in H end.
              clear H3 H4 H5.

              all: ring_sim
              change (53-1)%Z with 52%Z in *.
              change (52+52)%Z with (104%Z) in *.
              change (104+1)%Z with 105%Z in *.
              cbv [emax] in *.
              ring_simplify in H9.
              ring_simplify in H0.
              ring_simplify in H2.
              assert (
              cbn in *.

              nia.
         all: try lia.
         match goal with
           2: {
         end.
         intros.
         ring_simplify.


         Zify.zify.

         lia.
    cbv [fexp].
    break_innermost_match; intros.
    all: repeat match goal with
           | [ H1 : ?x = true, H2 : ?x = true |- _ ]
             => assert (H1 = H2)
               by (clear; generalize dependent x; clear; intros; subst; (idtac + symmetry); apply UIP_refl_bool);
                subst H2
           end.
    all: try congruence.
    all: repeat match goal with H : valid_binary _ = true |- _ => revert H end.
    all: rewrite ?Zpower.shift_pos_correct in *.
    all: cbv [fexp] in *.
    all: repeat match goal with H : context[Z.max] |- _ => revert H; apply Z.max_case_strong end.
    all: repeat match goal with
           | [ |- context[(?x + ?y - ?z - ?y)%Z] ]
             => replace (x + y - z - y)%Z with (x - z)%Z by lia
           end.
    all: try (intros; break_match; reflexivity).
    all: try lia.
    all: intros.
    all: repeat match goal with
           | [ H : (?x + ?y - ?z = ?y)%Z |- _ ]
             => assert (x = z) by lia; clear H; subst
           end.
    all: change digits2_pos with Pos.size in *.
    all: repeat match goal with
           | [ H : context[Z.pos (Pos.size ?x)] |- _ ]
             => let k := fresh in
                set (k := Z.pos (Pos.size x)) in *;
                let H' := fresh in
                assert (H' : (k = Z.log2 (Z.pos x) + 1)%Z)
                  by (clear; subst k; cbn; break_innermost_match; cbn; lia);
                first [ clearbody k; subst k
                      | rewrite ?H' in *; clear H'; subst k ]
           end.
    all: repeat match goal with
           | [ H : (Z.log2 ?x + 1 = ?y)%Z |- _ ]
             => assert (Z.log2 x = (y - 1))%Z by lia; clear H
           | [ H : Z.log2 ?x = _, H' : context[Z.log2 ?x] |- _ ]
             => rewrite H in H'
           end.
    all: repeat match goal with
           | [ H : context[Z.log2 (Z.pos (?x * ?y))] |- _ ]
             => change (Z.log2 (Z.pos (x * y))) with (Z.log2 (Z.pos x * Z.pos y)) in *;
                pose proof (Z.log2_mul_below (Z.pos x) (Z.pos y) ltac:(lia) ltac:(lia));
                pose proof (Z.log2_mul_above (Z.pos x) (Z.pos y) ltac:(lia) ltac:(lia))
           end.
    all: repeat match goal with
           | [ H : (Z.log2 ?x + 1 - ?z = ?y)%Z |- _ ]
             => assert (Z.log2 x = (y + z - 1))%Z by lia; clear H
           | [ H : (?y = Z.log2 ?x + 1)%Z |- _ ]
             => assert (Z.log2 x = (y - 1))%Z by lia; clear H
           | [ H : Z.log2 ?x = _, H' : context[Z.log2 ?x] |- _ ]
             => rewrite H in H'
           end.
    all: repeat match goal with
           | [ H : context[Z.log2 ?x] |- _ ]
             => unique pose proof (Z.log2_nonneg x)
           end.
    all: try (vm_compute emin in *; cbv [prec] in *; lia).
    all: repeat match goal with
           | [ H : (?y - 1 + ?x <= ?z + ?y - 1)%Z |- _ ]
             => assert (x <= z)%Z by lia; clear H
           | [ H : (?x + ?y - 1 <= ?y - 1 + ?z + ?w)%Z |- _ ]
             => assert (x <= z + w)%Z by lia; clear H
           | [ H : (?x + ?y - 1 <= ?z + (?y - 1) + ?w)%Z |- _ ]
             => assert (x <= z + w)%Z by lia; clear H
           | [ H : (?x + (?y - 1) <= ?z + ?y - 1)%Z |- _ ]
             => assert (x <= z)%Z by lia; clear H
           | [ H : (Z.log2 ?x <= 0)%Z |- _ ]
             => assert ((Z.log2 x = 0)%Z) by lia; clear H
           | [ H : (Z.log2 ?x <= Z.neg _)%Z |- _ ]
             => exfalso; clear -H; pose proof (Z.log2_nonneg x); lia
           end.
    all: rewrite ?Z.log2_null in *.
    all: repeat first [ progress subst
                      | rewrite Z.sub_diag in *
                      | match goal with
                        | [ H : (Z.pos ?x <= 1)%Z |- _ ]
                          => assert (x = 1%positive) by lia; clear H
                        | [ H : context[(?x - (?y + ?x))%Z] |- _ ]
                          => replace (x - (y + x))%Z with (-y)%Z in * by lia
                        | [ H : context[(?x - (?x + ?y))%Z] |- _ ]
                          => replace (x - (x + y))%Z with (-y)%Z in * by lia
                        | [ H : Z.opp ?x = ?y |- _ ]
                          => is_var x; assert (x = Z.opp y) by lia; clear H
                        end ].
    all: cbn [Z.opp] in *.
    all: try (break_match; try reflexivity; []).
    all: cbv [SF2B] in *; break_innermost_match_hyps; try congruence.
    all: match goal with H : B754_zero _ = B754_zero _ |- _ => inversion H; clear H end.
    all: subst.
    all: rewrite ?Pos.mul_1_r in *.
    all: rewrite ?Pos.mul_1_l in *.
    all: rewrite ?Z.add_0_l in *.
    all: rewrite ?Z.add_0_r in *.
    all: rewrite ?Z.mul_1_r in *.
    all: rewrite ?Z.mul_1_l in *.
    all: exfalso.

    all: cbv [binary_round_aux binary_fit_aux binary_overflow overflow_to_inf] in *; break_innermost_match_hyps; try congruence.
    all: rewrite ?Z.leb_le, ?Z.leb_gt in *.
    all: match goal with H : S754_zero _ = S754_zero _ |- _ => inversion H; clear H end.
    all: subst.
    all: cbv [shr_fexp shr fexp Zdigits2 shr_record_of_loc] in *.
    cbv [loc_of_shr_record choice_mode Round.cond_incr] in *.
    all: repeat match goal with
           | [ H : context[match ?b with Z.neg n => (@?N n, @?N' n) | Z0 => (?Z, ?Z') | Z.pos p => (@?P p, @?P' p) end] |- _ ]
             => replace (match b with Z.neg n => (@N n, @N' n) | Z0 => (Z, Z') | Z.pos p => (@P p, @P' p) end)
               with (match b with Z.neg n => N n | Z0 => Z | Z.pos p => P p end,
                      match b with Z.neg n => N' n | Z0 => Z' | Z.pos p => P' p end)
               in *
                 by now destruct b
           end.
    all: change digits2_pos with Pos.size in *.
    all: repeat match goal with
           | [ H : context[Z.pos (Pos.size ?x)] |- _ ]
             => let k := fresh in
                set (k := Z.pos (Pos.size x)) in *;
                let H' := fresh in
                assert (H' : (k = Z.log2 (Z.pos x) + 1)%Z)
                  by (clear; subst k; cbn; break_innermost_match; cbn; lia);
                first [ clearbody k; subst k
                      | rewrite ?H' in *; clear H'; subst k ]
           end.
    all: repeat match goal with
           | [ H : (_, _) = (_, _) |- _ ]
             => pose proof (f_equal (@fst _ _) H);
                pose proof (f_equal (@snd _ _) H);
                clear H; cbn [fst snd] in *
           end.
    all: repeat match goal with H : context[Z.max] |- _ => revert H end.
    all: repeat apply Z.max_case_strong.
    all: intros.
    all: repeat first [ progress cbn [SpecFloat.shr_m] in *
                      | progress subst
                      | congruence
                      | break_innermost_match_hyps_step ].
    all: try lia.
    all: change digits2_pos with Pos.size in *.
    all: repeat match goal with
           | [ H : context[Z.pos (Pos.size ?x)] |- _ ]
             => let k := fresh in
                set (k := Z.pos (Pos.size x)) in *;
                let H' := fresh in
                assert (H' : (k = Z.log2 (Z.pos x) + 1)%Z)
                  by (clear; subst k; cbn; break_innermost_match; cbn; lia);
                first [ clearbody k; subst k
                      | rewrite ?H' in *; clear H'; subst k ]
           end.
    all: cbn [shr_m] in *.
    all: try lia.
    all: rewrite ?Pos.mul_1_r, ?Pos.mul_1_l, ?Z.add_0_l, ?Z.add_0_r, ?Z.mul_1_r, ?Z.mul_1_l in *.
    all: repeat match goal with
           | [ H : (?y - 1 + ?x <= ?z + ?y - 1)%Z |- _ ]
             => assert (x <= z)%Z by lia; clear H
           | [ H : (?x + ?y - 1 <= ?y - 1 + ?z + ?w)%Z |- _ ]
             => assert (x <= z + w)%Z by lia; clear H
           | [ H : (?x + ?y - 1 <= ?z + (?y - 1) + ?w)%Z |- _ ]
             => assert (x <= z + w)%Z by lia; clear H
           | [ H : (?x + (?y - 1) <= ?z + ?y - 1)%Z |- _ ]
             => assert (x <= z)%Z by lia; clear H
           | [ H : (?y <= ?x + ?y - ?z)%Z |- _ ]
             => assert (z <= x)%Z by lia; clear H
           | [ H : (?x + ?y - ?z - (?x + ?y) = ?w)%Z |- _ ]
             => assert (z = -w)%Z by lia; clear H
           | [ H : (?x + ?y + ?z - ?w - ?z = 0)%Z |- _ ]
             => assert (x = w - y)%Z by lia; clear H
           | [ H : (?x <= ?y - ?z + ?z + (?w + ?x) - ?y)%Z |- _ ]
             => assert (0 <= w)%Z by lia; clear H
           | [ H : (Z.log2 ?x <= 0)%Z |- _ ]
             => assert ((Z.log2 x = 0)%Z) by lia; clear H
           | [ H : (Z.log2 ?x <= Z.neg _)%Z |- _ ]
             => exfalso; clear -H; pose proof (Z.log2_nonneg x); lia
           | [ H : Z.log2 ?x = _, H' : context[Z.log2 ?x] |- _ ]
             => rewrite H in H'
           | [ H : ?x = ?x |- _ ] => clear H
           | _ => progress change (Z.log2 1) with 0%Z in *
           | [ H : (?x <= ?x)%Z |- _ ] => clear H
           end.
    all: cbn [Z.opp] in *.
    all: try (vm_compute emin in *; cbv [prec] in *; lia).
    move p at bottom.
    cbn in Heqz0.
    inversion Heqz0; clear Heqz0; subst.

    Search iter_pos.
    cbv [shr_1] in *.
    inversion H12; subst; clear H12.
    vm_compute in H2, H, H4, H1, H9.
    repeat match goal with H : _ |- _ => ring_simplify in H end.
    cbv [
    rewrite H14 in *.
    repeat match goal with H : ?x = ?x |- _ => clear H
           end.
    repeat match goal with H : _ |- _ => ring_simplify in H end.
    match goal with
    end.
    match goal with
    | [ H : (?x - ?y + 1 = 0)%Z |- _ ] => assert (x = y - 1)%Z by lia; clear H
    end.

    all:
    all: rewrite
    ring_simplify in Heqz3.
    cbv [prec Z.opp] in Heqz3.
    inversion Heqz3; subst.
    rewrite H13 in *.
    2: {
    cbv [shr_m shr_record_of_loc] in *.
    lia.
                      match b with Z.neg n => (@N n, @N' n) | Z0 => (Z, Z') | Z.pos p => (@P p, @P' p) end)
                       PP (match b with Z.neg n => fst (N n) | Z0 => fst Z | Z.pos p => fst (P p) end)
                       (match b with Z.neg n => snd (N n) | Z0 => snd Z | Z.pos p => snd (P p) end))
               by now destruct b


    all: break_innermost_match_hyps.
    all: try lia.
    Print Zdigits2.
    5: {
    cbv [overflow_to_inf] in *.
    a


    ring_simplify in H7.
    match goal with

    end.
    all: repeat match goal with
           | [ H : Z.log2 ?x = _, H' : context[Z.log2 ?x] |- _ ]
             => rewrite H in H'


    all: cbv [binary_round_aux] in *; break_innermost_match_hyps.
    Print binary_round_aux.
    Check Z.log2_pos.
    lazymatch goal with
    | [ H : context[Z.log2 (Z.pos ?x)] |- _ ]
      => unique pose proof (Z.log2_pos (Z.pos x) ltac:(clear; lia))
    end.
           | [ H : context[Z.pos (Pos.size ?x)] |- _ ]
             => replace (Z.pos (Pos.size x)) with (Z.log2 (Z.pos x) + 1)%Z in H
                 by (clear; cbn; break_innermost_match; cbn; lia)
           end.
    match goal with
    end.

    all: try lia.

    Search Z.log2 Z.mul.


    Search Pos.size.
    Search Pos.size Pos.mul.
    break_innermost_match.
    all: cbv [valid_binary bounded canonical_mantissa].
    all:

    Print emin.
    all: rewrite ?Z
    all: match goal with
    rewrite Heqb.

    lazymatch goal with
    | [ |- context[if ?b then ?f _ else ?g _] ]
      => idtac f g
    end.
        e
                           if b then x else x') (if b then y else y')) by now destruct b
           end.
    match goal with
    end.
    Check binary_round_aux_equiv.

    pose proof binary_round_aux_correct.
    destruct_head'_bool;
      cbv [Bmult Bplus Operations.Fmult binary_normalize cond_Zopp Z.mul Z.opp].
    cbv [binary_round].
    cbn [xorb].
    all: shelve. }
  { rewrite binary_normalize_equiv.
    change (SF2Prim (B2SF ?x)) with (B2Prim x).
    rewrite Prim2B_B2Prim.
    cbv [Operations.Fplus Operations.Falign Operations.Fmult].
    repeat match goal with
           | [ |- context[match (if ?b then (?x, ?y) else (?x', ?y')) with pair A B => @?P A B end] ]
             => replace (match (if b then (x, y) else (x', y')) with pair A B => P A B end)
               with (P (if b then x else x') (if b then y else y')) by now destruct b
           end.
    cbv [shl_align].
    destruct Z.leb eqn:H'.
    all: rewrite ?Z.leb_le, ?Z.leb_gt in H'.
    all: rewrite ?Z.min_l, ?Z.min_r by lia.
    all: rewrite ?Z.sub_diag.
    all: cbn [fst snd].
    all: match goal with
         | [ |- binary_normalize _ _ _ _ _ ?m ?e ?z = binary_normalize _ _ _ _ _ ?m' ?e' ?z' ]
           => unshelve ((tryif constr_eq m m' then idtac else replace m with m' by shelve);
                        (tryif constr_eq e e' then idtac else replace e with e' by shelve);
                        (tryif constr_eq z z'
                          then idtac
                          else
                            cut (m' = 0%Z -> z = z');
                         [ cbv [binary_normalize]; break_innermost_match; try reflexivity; try now intros -> | ]))
         end.
    all: cbv [radix2 radix_val Z.pow] in *.
    all: break_innermost_match; try lia.
    all: cbn [fst snd].
    all: rewrite ?Zpower.shift_pos_correct.
    all: repeat first [ match goal with
                        | [ H : (?x - ?y = 0)%Z |- _ ]
                          => is_var y; assert (y = x) by lia; clear H; subst
                        | [ H : (?x - ?y = ?w)%Z, H' : context[(?y - ?x)%Z] |- _ ]
                          => replace (y - x)%Z with (Z.opp w) in * by lia
                        | [ H : Z.neg _ = Z.neg _ |- _ ] => inversion H; clear H
                        end
                      | progress subst
                      | progress cbn [Z.opp] in *
                      | rewrite Z.sub_diag in * ].
    all: destruct_head'_bool; cbv [cond_Zopp xorb Z.mul Z.opp Bool.eqb] in *; break_innermost_match; try lia. }
    { destruct_head'_bool; cbv [cond_Zopp xorb Z.mul Z.opp radix2 radix_val]; break_innermost_match; cbn [fst snd]; try reflexivity; try lia.

      all: change (Z.neg (Zpower.shift_pos ?x ?y)) with (Z.opp (Z.pos (Zpower.shift_pos x y))) in *; rewrite ?Zpower.shift_pos_correct.
      all: cbn [Z.opp] in *.
      all: rewrite ?Z.sub_diag in *.
      all: try lia. }
    { destruct_head'_bool; try reflexivity; cbn.
      all: cbv [Z.pow].
      all: break_innermost_match; try lia. }
    { break_innermost_match; try lia.
      all: try reflexivity.
      Search Zpower.shift_pos.
      Search (
      repeat match goal with
             end.

      all: match goal with
           | [ H : (2^?e)%Z = _ |- _ ] => destruct e eqn:?; cbn in H; try lia
           end.
      zify.
      lia.
      destruct
    f_equal.

    2: intros ->.
                replace m with m';
              [ replace e with e';

    Print binary_normalize.
    all: f_equal.

    2: { destruct_head'_bool; try reflexivity; cbn.
    break_innermost_match; try lia.
    Search (?x - ?x)%Z.
    Search (_ <=? _)%Z false iff.
    break_innermost_match_step.
    Print B2Prim.
    Search Prim2B SF2Prim.
    Search Z.min
    Check Operations.Falign_spec.
    cbv [Operations.Falign].
    destruct_head'_bool; cbn [xorb];
      cbv [Bmult Bplus Operations.Fmult Operations.Fplus Operations.Falign binary_normalize cond_Zopp Z.mul Z.opp].



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
