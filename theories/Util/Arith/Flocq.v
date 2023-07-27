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
  intros Hz Hy Hx; revert Hz Hy Hx.
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
  | [ |- context[SF64mul ?x ?y] ]
    => rewrite <- (Prim2SF_SF2Prim x), <- (Prim2SF_SF2Prim y), <- mul_spec, ?SF2Prim_Prim2SF by assumption
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
  all: rewrite ?add_equiv, ?mul_equiv, ?binary_normalize_equiv.
  all: try change (SF2Prim (B2SF ?x)) with (B2Prim x).
  all: rewrite ?Prim2B_B2Prim.
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
    reflexivity. }
  { cbv [Operations.Fplus Operations.Falign Operations.Fmult].
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
Qed.
