From Coq Require Import Zify ZifyUint63 Lia ZArith NArith Uint63 String.
From NeuralNetInterp.Util Require Import Default Pointed.
From NeuralNetInterp.Util.List Require Import NthError.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor.

#[local] Open Scope core_scope.

Lemma raw_get_arange_app {start stop step i}
  : @arange start stop step i = start + RawIndex.tl i * step.
Proof. reflexivity. Qed.

Lemma raw_get_arange {start stop step i}
  : raw_get (@arange start stop step) i = start + RawIndex.tl i * step.
Proof. cbv [raw_get]; apply raw_get_arange_app. Qed.

Lemma get_arange {start stop step i}
  : get (@arange start stop step) i = start + (Index.tl i mod (1 + (stop - start - 1) / step))%uint63 * step.
Proof. cbv [get]; rewrite raw_get_arange; reflexivity. Qed.

Lemma raw_get_reshape_all_app {r s A t idx}
  : @reshape_all r s A t idx = t (RawIndex.unreshape s idx).
Proof. reflexivity. Qed.

Lemma raw_get_reshape_all {r s A t idx}
  : raw_get (@reshape_all r s A t) idx = raw_get t (RawIndex.unreshape s idx).
Proof. cbv [raw_get]; apply raw_get_reshape_all_app. Qed.

Lemma get_reshape_all {r s A t idx}
  : get (@reshape_all r s A t) idx = t.[RawIndex.unreshape' s (Uint63.to_Z (Index.tl idx mod Uint63.of_Z (Shape.reshape' s)))]%raw_tensor.
Proof. cbv [get]; rewrite raw_get_reshape_all; reflexivity. Qed.

Lemma raw_get_nth_error_Shape_Tuple_to_list'_ntupleify'_init'_app
  : forall {r A A' B1 B2 f g k ai b s' i j},
    List.nth_error
      (@Shape.Tuple.to_list'
         r (fun _ => A) B2 A' (Shape.repeat s' r) g k
         (@ntupleify' r (Shape.repeat s' r) A B1 B2 f
            (@Shape.Tuple.init' r _ B1 (Shape.repeat s' r) ai b)
            (RawIndex.unreshape' (Shape.repeat s' r) i))) j
    = if j <? r
      then Some (g s' (ai s' (tt, Uint63.of_Z (i // Z.pow (Uint63.to_Z s') (r - S j) mod Uint63.to_Z s'))))
      else List.nth_error (k (f b)) (j - r).
Proof.
  cbv beta.
  induction r as [|r IH]; cbn [Shape.Tuple.to_list' ntupleify' Shape.repeat RawIndex.unreshape' Shape.Tuple.init']; intros.
  all: cbv [RawIndex.hd RawIndex.tl RawIndex.snoc Shape.hd Shape.snoc Shape.tl RawIndex.nil Shape.nil Classes.ltb nat_has_ltb] in *; cbn [fst snd].
  { break_innermost_match; try lia.
    f_equal; lia. }
  rewrite IH.
  break_innermost_match; try lia.
  all: try assert (j = r) by lia; subst.
  all: rewrite ?Nat.sub_diag, ?Z.sub_diag, ?Z.pow_0_r; cbn [List.nth_error fst snd]; cbv [raw_get].
  all: cbv [Classes.int_div Z_has_int_div]; rewrite ?Z.div_1_r.
  all: lazymatch goal with
       | [ H : (?x <? S _)%nat = false |- _ ] => is_var x; destruct x; [ lia | ]
       | _ => idtac
       end.
  all: rewrite ?Nat.sub_succ, ?Nat.sub_succ_l by lia; cbn [List.nth_error snd].
  all: try reflexivity.
  rewrite Zdiv_Zdiv, <- Z.pow_succ_r by lia.
  repeat (f_equal; []).
  lia.
Qed.

Lemma raw_get_cartesian_exp_app {s A defaultA t n idx}
  (i:=RawIndex.tl (RawIndex.hd idx))
  (j:=RawIndex.tl idx)
  : @cartesian_exp s A defaultA t n idx
    = if n =? 0
      then point
      else t (tt,
               Uint63.of_Z
                 ((Uint63.to_Z i // (Uint63.to_Z s ^ (Uint63.to_Z (n - (j mod n) - 1)))%Z)
                    mod Uint63.to_Z s)).
Proof.
  subst i j; destruct idx as [[[] i] j].
  cbv [cartesian_exp Shape.Tuple.init cartesian_nprod raw_get Shape.Tuple.nth_default ntupleify Shape.Tuple.to_list List.nth_default].
  rewrite raw_get_reshape_all_app.
  cbv [RawIndex.unreshape].
  cbv [Uint63.coer_int_N'].
  setoid_rewrite raw_get_nth_error_Shape_Tuple_to_list'_ntupleify'_init'_app.
  repeat rewrite ?List.nth_error_nil, ?N2Nat.id, ?Z2N.id, ?Nat2N.inj_succ, ?N2Z.inj_succ, ?nat_N_Z, ?Nat2Z.id, ?Z2Nat.id, ?of_to_Z, ?Z_N_nat by lia.
  cbv [RawIndex.item RawIndex.snoc RawIndex.tl]; cbn [fst snd].
  cbv [Classes.ltb nat_has_ltb Classes.eqb Classes.zero int_has_eqb int_has_zero] in *.
  break_innermost_match.
  all: try (exfalso; lia).
  all: try reflexivity.
  repeat (f_equal; []).
  nia.
Qed.

Lemma raw_get_cartesian_exp {s A defaultA t n idx}
  (i:=RawIndex.tl (RawIndex.hd idx))
  (j:=RawIndex.tl idx)
  : raw_get (@cartesian_exp s A defaultA t n) idx
    = if n =? 0
      then point
      else t.[[Uint63.of_Z
                ((Uint63.to_Z i // (Uint63.to_Z s ^ (Uint63.to_Z (n - (j mod n) - 1)))%Z)
                   mod Uint63.to_Z s)]]%raw_tensor.
Proof. cbv [raw_get]; rewrite raw_get_cartesian_exp_app; reflexivity. Qed.

Lemma get_cartesian_exp {s A defaultA t n idx}
  (i:=Index.tl (Index.hd idx))
  (j:=Index.tl idx)
  : get (@cartesian_exp s A defaultA t n) idx
    = if n =? 0
      then point
      else t.[[Uint63.of_Z
                 ((Uint63.to_Z (i mod s ^ Z.to_N (Uint63.to_Z n))
                     // (Uint63.to_Z s ^ (Uint63.to_Z (n - (j mod n) - 1)))%Z)
                    mod Uint63.to_Z s)]]%raw_tensor.
Proof.
  cbv [get adjust_indices_for Index.map2 adjust_index_for Index.tl Index.snoc]; cbn [snd]; rewrite raw_get_cartesian_exp.
  subst i j; cbv [RawIndex.nil Index.nil RawIndex.snoc Index.snoc Index.tl RawIndex.tl RawIndex.hd Index.hd Shape.snoc Uint63.coer_int_N']; cbn [fst snd].
  cbv [Classes.modulo Uint63.int_has_modulo].
  replace ((snd idx mod n) mod n)%uint63 with (snd idx mod n)%uint63; [ reflexivity | ].
  destruct idx as [[[] i] j]; cbv in *; clear.
  assert ((j mod n) / n = 0)%uint63 by nia.
  nia.
Qed.
