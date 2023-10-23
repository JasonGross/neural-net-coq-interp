From Coq Require Import Field Zify PreOmega ZifyUint63 Qreals Lqa Lra Reals Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util.Tactics Require Import IsFloat IsUint63 BreakMatch DestructHead Head UniquePose SplitInContext.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63 Wf_Uint63.Instances Wf_Uint63.Proofs SolveProperEqRel Default.
From NeuralNetInterp.Util.Arith Require Import Classes Instances Instances.Reals Classes.Laws Instances.Laws FloatArith.Definitions Reals.Definitions Instances.Reals.Laws Reals.Proofs Reals.Instances.
From NeuralNetInterp.Torch Require Import Tensor.Instances Slicing.Instances Tensor.Proofs.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model Heuristics TheoremStatement Model.Instances Model.ExtraComputations Model.ExtraComputations.Flocqify Model.ExtraComputations.Instances Model.Flocqify Model.Rify.
From NeuralNetInterp.Util.Compat Require Import RIneq.
Import LoopNotation.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import logit_delta.

From NeuralNetInterp.Util.Arith Require Import Flocq.Hints.Prim2B.
Import Instances.Uint63.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From Flocq Require Import Raux.
From Interval.Missing Require Import Stdlib.
From Coquelicot Require Import Rcomplements.
From NeuralNetInterp.Util.Arith Require Import Flocq Flocq.Instances Flocq.Notations Flocq.Definitions.
Local Open Scope core_scope.

#[export] Existing Instance reflexive_eq_dom_reflexive.
#[local] Hint Opaque Model.logit_delta logits all_tokens AllLogits.all_tokens_logits
  Classes.abs Classes.leb Classes.sub Classes.pow Classes.sqrt
  R_has_sqrt
  Rabs Rle_bool Rminus
  N.pow Z.of_N
  PrimFloat.to_Q_cps PrimFloat.to_Q
  Q2R
  of_Z
  B2R SF2B SF2Prim prec emax : rewrite.
#[local] Hint Constants Opaque : typeclass_instances.
#[local] Typeclasses Transparent
  relation Hetero.relation fst snd
  Classes.max Instances.Uint63.max has_default_max_leb
  Classes.leb Instances.Uint63.int_has_leb
  Classes.one
  Shape ShapeType Shape.nil Shape.snoc Shape.app Shape.broadcast2 Shape.broadcast3 Shape.map Shape.map2 Shape.map3 Shape.hd Shape.tl Shape.ones Shape.repeat Shape.ShapeType.one
  RawIndex RawIndexType RawIndex.nil RawIndex.snoc RawIndex.app RawIndex.hd RawIndex.tl
  Index IndexType Index.nil Index.snoc Index.app Index.hd Index.tl
  SliceIndex.transfer_shape SliceIndex.SliceIndexType.transfer_shape Slice.start Slice.stop Slice.step Slice.Concrete.length Slice.norm_concretize SliceIndex.SliceIndexType.transfer_shape_single_index
  inject_int int_has_one
  Rank Nat.radd
  Classes.add Classes.div Classes.sub Classes.mul
  has_add_with has_div_by has_sub_with has_mul_with
  tensor tensor_of_rank
  Tensor.eqfR
  raw_get
  with_default
.
#[export] Hint Transparent PrimInt63.leb : typeclass_instances.
#[local] Set Keyed Unification.

Ltac specialize_step_with f i' :=
  let f' := fresh f in
  rename f into f';
  pose (f' i') as f;
  change (f' i') with f;
  repeat match goal with
    | [ H := context[f' i'] |- _ ] => change (f' i') with f in (value of H)
    | [ H : context[f' i'] |- _ ] => change (f' i') with f in H
    end;
  cbv [f'] in f; clear f'.
Ltac specialize_step i' :=
  match goal with
  | [ |- context[?f i'] ] => is_var f; specialize_step_with f i'
  | [ H := context[?f i'] |- _ ] => is_var f; specialize_step_with f i'
  | [ H : context[?f i'] |- _ ] => is_var f; specialize_step_with f i'
  end.

#[local] Ltac saturate_mod_pos_bound _ :=
  repeat match goal with
    | [ |- context[(?x mod ?y)%Z] ] => unique pose proof (Z.mod_pos_bound x y ltac:(cbv -[Z.lt]; clear; lia))
    | [ H : context[(?x mod ?y)%Z] |- _ ] => unique pose proof (Z.mod_pos_bound x y ltac:(cbv -[Z.lt]; clear; lia))
    end.

#[local] Ltac rewrite_mod_mod_small_by_lia _ :=
  repeat match goal with
    | [ H : (0 <= ?x mod ?y < ?y)%Z |- context[((?x mod ?y) mod ?z)%Z] ]
      => rewrite (Z.mod_small (x mod y)%Z z)
        by (revert H; generalize (x mod y)%Z; clear; cbv -[Z.le Z.lt]; lia)
    | [ H : (0 <= ?x mod ?y < ?y)%Z, H' : context[((?x mod ?y) mod ?z)%Z] |- _ ]
      => rewrite (Z.mod_small (x mod y)%Z z) in H'
          by (revert H; generalize (x mod y)%Z; clear; cbv -[Z.le Z.lt]; lia)
    end.

#[local] Ltac arg_equiv_side _ :=
  cbv [Classes.max Classes.min Classes.ltb Classes.leb Classes.eqb
         has_default_max_leb has_default_min_leb
         Uint63.max Uint63.min
         R_has_leb R_has_ltb R_has_min int_has_leb int_has_ltb Rmin];
  clear; intros;
  (idtac + instantiate (1:=Rle_bool));
  break_innermost_match;
  rewrite <- ?not_true_iff_false, ?Rle_bool_iff, ?Rlt_bool_iff in *;
  try lia; try lra.
#[local] Ltac handle_argminmax_in H :=
  match type of H with
  | context[@Reduction.argmax ?A ?ltbA ?start ?stop ?step ?f]
    => let am := fresh "v" in
       let Hv := fresh in
       remember (@Reduction.argmax A ltbA start stop step f) as am eqn:Hv in *;
       symmetry in Hv;
       apply Reduction.argmax_spec in Hv
  | context[@Reduction.argmin ?A ?lebA ?start ?stop ?step ?f]
    => let am := fresh "v" in
       let Hv := fresh in
       remember (@Reduction.argmin A lebA start stop step f) as am eqn:Hv in *;
       symmetry in Hv;
       apply Reduction.argmin_spec in Hv
  end;
  change (1%uint63 =? 0) with false in *;
  cbv beta iota in *.
#[local] Ltac handle_argminmax_step _ :=
  match goal with H : _ |- _ => handle_argminmax_in H end.
#[local] Ltac handle_argminmax _ := repeat handle_argminmax_step ().

From Coq.Compat Require Import AdmitAxiom.

Theorem good_accuracy : TheoremStatement.Accuracy.best (* (abs (real_accuracy - expected_accuracy) <? error)%float = true *).
Proof.
  cbv [true_accuracy].
  cbv [Classes.abs Classes.leb Classes.ltb Classes.sub item float_has_abs float_has_sub].
  cbv [item raw_get].
  (* convert from prim float to flocq *)
  rewrite leb_equiv, abs_equiv, sub_equiv, div_equiv.
  let lem := constr:(Model.acc_fn_equiv
                       (use_checkpoint2:=false)
                       logits_all_tokens
                       (logits (use_checkpoint:=false) (all_tokens (use_checkpoint:=false)))
                       all_tokens (all_tokens (use_checkpoint:=false))
                       (Model.logits_equiv (Model.all_tokens_Proper _ _ I))
                       (Model.all_tokens_Proper _ _ I)
                       tt) in
  rewrite lem.
  (* Now I'd like to convert to R, but this means I need to prove lack of exceptions, I think *)
  (* So for now instead I'm playing around without really knowing what
  I'm doing. My first attempt failed when I cound't even turn [abs (x
  / y - z) < w] into [abs (x - z * y) < w*y]... *)
  pose proof (Rify.Model.acc_fn_equiv_bounded (use_checkpoint1:=false) (use_checkpoint2:=false) tt) as lem.
  cbv beta iota zeta in lem.
  change Babs with (Classes.abs : has_abs (binary_float prec emax)); cbv beta.
  change (Bminus mode_NE) with (Classes.sub : has_sub (binary_float prec emax)); cbv beta.
  change (Bdiv mode_NE) with (Classes.div : has_div (binary_float prec emax)); cbv beta.
  change (Bleb) with (Classes.leb : has_leb (binary_float prec emax)); cbv beta.
  lazymatch goal with
  | [ H : (abs (?y - B2R ?x) <=? ?err) = true |- (abs (?x' / ?e - ?one) <=? ?err') = true ]
    => first [ constr_eq x x' | fail 1 x "≠" x' ];
       cut ((abs ((y:R) / (e:R) - (one:R)) <=? ((err':R) + (err:R) / (e:R))) = true);
       cbv beta; [ | clear H ];
       [ revert H; generalize x y err err' e one | ]
  end.
  { cbv [Classes.abs Classes.leb Classes.sub Classes.div
           R_has_abs R_has_leb R_has_sub R_has_div
           binary_float_has_abs binary_float_has_leb binary_float_has_sub binary_float_has_div].
    intros.
    rewrite Bleb_correct_full, B2R_Babs, is_finite_Babs, B2R_Bminus, is_nan_Babs, Bsign_Babs, B2R_Bdiv.
    admit. (* XXX FIXME TODO rounding error *)  }

  (** change to multiplicative *)
  lazymatch goal with
  | [ |- (abs (?x / ?y - ?z) <=? ?w) = true ]
    => cut ((abs (x - y * z) <=? w * y) = true);
       [ generalize x z w; intros *; assert (0 < y)%R
       | ]
  end.
  { vm_compute; lra. }
  { cbv [Classes.abs Classes.mul Classes.sub Classes.div Classes.leb Classes.zero
           R_has_abs R_has_mul R_has_sub R_has_div R_has_leb R_has_zero] in *.
    match goal with
    | [ |- Rle_bool ?x ?y = true -> Rle_bool ?z ?w = true ]
      => cut (Rle x y -> Rle z w);
         [ destruct (Rle_bool_spec x y), (Rle_bool_spec z w); intros; auto; lra
         | ]
    end.
    intro H'; apply Rabs_le_inv in H'.
    apply Rabs_le.
    split; destruct H' as [H'0 H'1].
    all: eapply Rmult_le_reg_r; [ eassumption | ].
    all: repeat match goal with H : _ |- _ => move H at bottom; progress field_simplify in H; [ | nra .. ] end.
    all: field_simplify; nra. }

  (** Introduce the computation *)
  pose proof (@Model.logit_delta_equiv_bounded false _) as lem.
  cbv beta iota zeta in *.
  pose proof logit_delta_eq as lemeq.
  cbv [logit_delta_float_gen] in lemeq.
  setoid_rewrite <- Model.logit_delta_equiv in lem.
  setoid_rewrite <- lemeq in lem; clear lemeq.
  all: try now apply Model.all_tokens_Proper.
  all: try (intro; rewrite AllLogits.all_tokens_logits_eq).
  all: try (apply (Model.logits_Proper_dep _ _ Model.Rf); repeat intro; subst; try exact I; try reflexivity).
  all: lazymatch goal with
       | [ |- Model.R ?x ?y ]
         => let x := head x in
            let y := head y in
            cbv [x y]
       | _ => idtac
       end.
  all: do 2 lazymatch goal with
         | [ |- Model.R (if ?x then _ else _) (if ?y then _ else _) ]
           => let x := head x in
              let y := head y in
              cbv [x y]
         | _ => idtac
         end.
  all: autorewrite with prim2b; try reflexivity.
  all: break_innermost_match; try reflexivity.
  all: [ > ].
  set (logits := logits all_tokens) in *.
  set (all_tokens := all_tokens) in *.
  revert lem.
  cbv beta iota delta [Model.logit_delta acc_fn PArray.maybe_checkpoint] in *.
  Optimize.lift_lets ().
  all: repeat match goal with H := [ _ ] : N |- _ => subst H end.
  Optimize.subst_cleanup ().
  Optimize.subst_local_cleanup ().

  (** working on goal *)
  cbv [Accuracy.error expected_accuracy total_rounding_error logit_delta_rounding_error logit_delta].
  cbv [Prim2B SF2B].
  repeat (set (k := Prim2SF _) at 1; vm_compute in k; subst k; cbv beta iota).
  cbv beta iota.
  cbv [Classes.abs Classes.mul Classes.div R_has_div Classes.add R_has_add R_has_mul R_has_abs Classes.leb R_has_leb Classes.sub R_has_sub mean reduce_axis_m1 reduce_axis_m1' map Reduction.mean int_has_sub coer coer_trans coer_binary_float_R coer_float_binary_float Truncating.coer_Z_float B2R Defs.F2R Defs.Fnum Defs.Fexp cond_Zopp bpow Zaux.radix_val Zaux.radix2 Prim2B].
  set (k := PrimFloat.of_Z _); vm_compute in k; subst k.
  set (k := of_Z _); vm_compute in k; subst k.
  set (k := Prim2SF _); vm_compute in k; subst k.
  cbv [SF2B].
  repeat (set (k := Z.pow_pos _ _); vm_compute in k; subst k).
  vm_compute PrimFloat.to_Q.
  cbv [reshape_snoc_split reshape_all Shape.broadcast2 RawIndex.unreshape Shape.map2 RawIndex.unreshape' RawIndex.curry_radd raw_get RawIndex.item RawIndex.combine_radd RawIndex.tl Shape.snoc RawIndex.snoc Shape.tl Shape.nil RawIndex.nil]; cbn [fst snd].
  vm_compute Uint63.max.
  intro H'; apply Rle_bool_true.
  repeat match goal with
         | [ H : Rle_bool ?x ?y = true |- _ ]
           => let H' := fresh in
              assert (H' : Rle x y) by (now destruct (Rle_bool_spec x y));
              clear H; rename H' into H
         end.
  revert H'.
  rewrite 2 Rabs_le_between'.
  repeat match goal with
         | [ |- context[(IZR ?n * / IZR (Zpos ?d))%R] ]
           => change ((IZR n * / IZR (Zpos d))%R) with (Q2R (n # d))
         | [ |- context[(Q2R ?x / Q2R ?y)%R] ]
           => rewrite <- (Qreals.Q2R_div x y) by Lqa.lra
         | [ |- context[(Q2R ?x + Q2R ?y)%R] ]
           => rewrite <- (Qreals.Q2R_plus x y)
         | [ |- context[(Q2R ?x * Q2R ?y)%R] ]
           => rewrite <- (Qreals.Q2R_mult x y)
         | [ |- context[(Q2R ?x - Q2R ?y)%R] ]
           => rewrite <- (Qreals.Q2R_minus x y)
         | [ |- context[(- Q2R ?x)%R] ]
           => rewrite <- (Qreals.Q2R_opp x)
         | [ |- context[Q2R ?x] ]
           => let x' := (eval vm_compute in x) in
              progress change x with x'
         | [ |- context[Q2R ?x] ]
           => progress (erewrite (Qreals.Qeq_eqR x)
                  by (symmetry; etransitivity; [ | apply Qred_correct ]; vm_compute Qred; reflexivity))
         | [ |- (Rabs ?x <= ?y)%R ]
           => cut (-y <= x <= y)%R;
              [ generalize y x; clear; cbv [Rabs]; intros; destruct Rcase_abs; lra | ]
         | [ |- (?x <= ?y - ?z <= ?w)%R ]
           => cut (x + z <= y <= w + z)%R;
              [ generalize x y z w; clear; intros; lra | ]
         | [ |- (?x <= ?y / ?z <= ?w)%R ]
           => cut (x * z <= y <= w * z)%R;
              [ generalize x y w; clear; intros; nra | ]
         end.
  cbv [coer_tensor map] in *.

  (** In the more general case, this theorem will classify which inputs are correct and which are incorrect *)
  match goal with
  | [ |- context[Reduction.sum _ _ _ ?f] ]
    => set (res' := f)
  end.
  subst res; cbv beta in *.
  rename res' into res.
  vm_compute lift_coer_has_zero in *.
  intro H'.
  assert (Hres : pointwise_relation _ eq res (fun _ => 1%R)).
  2: { clearbody res; clear -Hres.
       rewrite (Reduction.sum_Proper _ _ Hres).
       rewrite (Reduction.sum_const_mul_step1 (coerN:=IZR))
         by (cbv [Classes.zero coer Classes.add Classes.mul R_has_mul R_has_add R_has_zero]; intros;
             rewrite ?N2Z.inj_succ, ?N2Z.inj_0, ?Q2RAux.Q2R_inject_Z, ?succ_IZR;
             lra).
       cbv [coer coer_trans Q2R_coer inject_Z_coer Uint63.coer_int_N'].
       vm_compute Z.of_N.
       cbv -[Rle IZR].
       lra. }

  intro i.
  subst res; cbv beta.
  all: cbv [Classes.one Classes.zero lift_coer_has_one lift_coer_has_zero Z_has_zero Z_has_one of_bool map map2] in *.
  match goal with
  | [ |- (if ?x then ?t else ?f) = ?t' ]
    => cut (x = true); [ intros ->; vm_compute; lra | ]
  end.
  apply eqb_complete.
  set (i' := of_Z _).

  (** reason about min *)
  cbv [item raw_get] in *.

  cbv [Classes.sub tensor_sub R_has_sub] in *.
  cbv [Tensor.min Tensor.max max_dim_m1 min_dim_m1 argmax_dim_m1 reduce_axis_m1 reduce_axis_m1' gather_dim_m1
         Tensor.map raw_get Shape.keepdim Nat.radd Shape.reshape' Shape.reduce Shape.app
         reshape_snoc_split reshape_all Shape.broadcast2 RawIndex.unreshape Shape.map2 RawIndex.unreshape' RawIndex.curry_radd raw_get RawIndex.item RawIndex.combine_radd RawIndex.tl Shape.snoc RawIndex.snoc Shape.tl Shape.nil RawIndex.nil
         Tensor.get Tensor.raw_get SliceIndex.slice SliceIndex.SliceIndexType.slice RawIndex.hd Slice.invert_index RawIndex.tl RawIndex.snoc RawIndex.nil adjust_index_for Shape.tl Shape.hd Shape.nil Shape.snoc FancyIndex.slice reshape_app_combine' RawIndex.uncurry_radd FancyIndex.slice_ map_dep RawIndex.split_radd Nat.radd FancyIndex.broadcast map2 FancyIndex.FancyIndexType.broadcast map2' repeat' Tensor.map Tensor.map' map3] in *;
    cbn [fst snd] in *.
  repeat match goal with H := fun x => coer (?f x) |- _ => first [ is_var f | is_const f ]; subst H end.
  repeat match goal with H := ?f |- _ => first [ is_var f | is_const f | is_constructor f ]; subst H end.
  clearbody logits.


  assert (Hbounds : forall j, ((0 <=? all_tokens j) && (all_tokens j <? Uint63.of_Z cfg.d_vocab))%uint63 = true).
  { clear; subst all_tokens.
    cbv [get raw_get all_tokens].
    intro j.
    rewrite ?PArray.maybe_checkpoint_correct.
    rewrite raw_get_cartesian_exp_app, raw_get_arange_app.
    cbv [RawIndex.tl RawIndex.hd]; cbn [snd fst].
    vm_compute point.
    break_innermost_match; [ reflexivity | ].
    revert j.
    repeat match goal with
           | [ |- context[?x] ]
             => lazymatch type of x with
                | PrimInt63.int => idtac
                | Z => idtac
                | bool => idtac
                | positive => idtac
                end;
                let v := (eval cbv in x) in
                progress change x with v
           end.
    cbv [Classes.pow Classes.mul Classes.add Classes.zero Classes.one Classes.leb Classes.ltb int_has_add Classes.int_div Z_has_int_div int_has_mul].
    intros; rewrite Bool.andb_true_iff; split.
    { lia. }
    { lia. } }
  clearbody all_tokens.


  RawIndex.curry_lets ().
  repeat specialize_step i'.
  do 4 (set (k := of_Z _) in *; vm_compute in k; let kv := (eval cbv delta [k] in k) in is_uint63  kv; subst k).
  vm_compute Uint63.max in *.
  change pred_logits with (predicted_logits i') in *; clear pred_logits.
  repeat match goal with
         | [ H := fun x (_ : ?T) => @?f x |- _ ]
           => let H' := fresh in
              rename H into H';
              pose f as H;
              change (fun x (_ : T) => H x) in (value of H');
              subst H'; cbv beta in *
         end.
  change true_maximum with (indices_of_max i'); clear true_maximum.
  destruct_head'_and.
  lazymatch goal with
  | [ H : ?lower <= Reduction.min _ _ _ (fun i => min_incorrect_logit _) |- _ ]
    => assert (lower <= min_incorrect_logit i');
       [ erewrite !@Reduction.argmin_min_equiv in *; try typeclasses eauto;
         [ | now arg_equiv_side () .. ];
         handle_argminmax_in H;
         let H' := lazymatch goal with H' : ex _ |- _ => H' end in
         destruct H' as [? [? H']];
         unshelve (let pf := open_constr:(_) in
                   specialize (H' i' (Z.to_nat (Uint63.to_Z i')) pf));
         [ cbv [Reduction.in_bounds_alt_at]; clear;
           rewrite ?nat_N_Z, ?Z2Nat.id, ?of_to_Z by lia;
           cbv [Classes.add Classes.mul Classes.zero Classes.max int_has_add Classes.one Classes.eqb Uint63.max int_has_eqb int_has_one int_has_mul int_has_zero has_default_max_leb Classes.leb int_has_leb Uint63.leb] in *;
           match goal with
           | [ |- context[?v] ]
             => lazymatch v with context[i'] => fail | context[if _ then _ else _] => idtac end;
                lazymatch type of v with
                | Z => idtac
                | nat => idtac
                | int => idtac
                | N => idtac
                | bool => idtac
                end;
                let v' := (eval vm_compute in v) in
                progress change v with v'
           end;
           try (break_innermost_match; lia)
         | cbv [Classes.leb R_has_leb is_true] in H';
           rewrite !Rle_bool_iff in H';
           etransitivity; [ eassumption | etransitivity; [ apply H' | apply Req_le, f_equal, f_equal ] ];
           subst i';
           rewrite ?of_Z_spec;
           saturate_mod_pos_bound ();
           rewrite_mod_mod_small_by_lia ();
           try reflexivity ]
       | clear H ]
  end.
  all: [ > ].
  match goal with H : Reduction.min _ _ _ _ <= _ |- _ => clear H end.
  move min_incorrect_logit at bottom.
  cbv [inject_int] in *.
  specialize_step i'.
  specialize_step i'.
  cbv [Classes.modulo int_has_modulo Classes.max Uint63.max has_default_max_leb Classes.leb int_has_leb Uint63.leb] in *.
  set (i'' := (i' mod _)%uint63) in *.
  assert (i' = i'') by (clear; subst i' i''; try nia).
  clearbody i''; subst i''.
  move indices_of_max at bottom.
  subst min_incorrect_logit.
  lazymatch goal with
  | [ H : ?lower <= Reduction.min _ _ _ (fun i => ?f _) |- ?iv = indices_of_max i' ]
    => assert (lower <= f iv);
       [ erewrite !@Reduction.argmin_min_equiv in *; try typeclasses eauto;
         [ | now arg_equiv_side () .. ];
         handle_argminmax_in H;
         let H' := lazymatch goal with H' : ex _ |- _ => H' end in
         destruct H' as [? [? H']];
         let Hv := fresh in
         let iv' := fresh iv in
         remember iv as iv' eqn:Hv in *;
         subst iv;
         handle_argminmax_in Hv;
         let Hv := lazymatch goal with H : ex _ |- _ => H end in
         let n := fresh in
         destruct Hv as [n Hv];
         unshelve (let pf := open_constr:(_) in
                   specialize (H' iv' n pf));
         [ subst iv'; now apply Hv
         | cbv [Classes.leb R_has_leb is_true] in H';
           rewrite !Rle_bool_iff in H';
           etransitivity; [ eassumption | apply H' ] ]
       | clear H ]
  end.
  subst logits_above_correct0; cbv beta in *.
  break_innermost_match_hyps; cbv [Classes.eqb int_has_eqb] in *.
  { lia. }
  exfalso.
  clear bigger_than_anything.
  specialize_step i'.
  specialize_step i'.
  specialize_step i'.
  specialize_step i'.
  subst logits_above_correct; cbv beta in *.
  subst correct_logits.
  subst pred_tokens indices_of_max.
  unshelve erewrite !@Reduction.argmax_max_equiv in *; try typeclasses eauto;
    [ | now arg_equiv_side () .. ].
  handle_argminmax ().
  setoid_rewrite Bool.andb_true_iff in Hbounds.
  split_and.
  destruct_head'_ex; destruct_head'_and.
  repeat match goal with
         | [ H : Reduction.in_bounds_alt_at ?start ?stop ?step ?v _ |- _ ]
           => unique assert ((start <=? v) = true /\ (v <? stop) = true);
              [ generalize (@Reduction.in_bounds_alt_bounded start stop step v (ex_intro _ _ H));
                generalize v;
                vm_compute; lia
              | ]
         end.
  destruct_head'_and.
  cbv [Classes.leb Classes.ltb int_has_leb int_has_ltb R_has_ltb] in *.
  cbv [Classes.zero int_has_zero] in *.
  repeat match goal with
         | [ H0 : forall j0, (0 <=? ?f j0)%uint63 = true, H1 : forall j, (?f j <? ?m)%uint63 = true, H' : context[?f ?j'] |- _ ]
           => progress (try unique pose proof (H0 j'); try unique pose proof (H1 j'))
         | [ H0 : (0 <=? ?x)%uint63 = true, H1 : (?x <? ?m)%uint63 = true, H' : context[(?x mod ?m)%uint63] |- _ ]
           => replace (x mod m)%uint63 with x in *
               by (revert H0 H1; generalize x; cbv; clear; nia)
         end.
  repeat match goal with
         | [ H : (0 <=? ?x)%uint63 = true, H' : (?x <? ?b)%uint63 = true |- _ ]
           => is_uint63 b;
              unique assert (Reduction.in_bounds_alt_at 0 b 1 x (Z.to_nat (Uint63.to_Z x)))
                by (rewrite Reduction.in_bounds_alt_at_step1_small_iff by reflexivity;
                    repeat apply conj; try assumption; lia)
         end.
  repeat match goal with
         | [ H : Reduction.in_bounds_alt_at ?start ?stop ?step _ _, H' : forall j n', Reduction.in_bounds_alt_at ?start ?stop ?step j n' -> _ |- _ ]
           => unique pose proof (H' _ _ H)
         end.
  repeat match goal with
         | [ H : (?x <? ?x)%R = true \/ _ |- _ ] => clear H
         | [ H : (?x <? ?x)%uint63 = true \/ _ |- _ ] => clear H
         end.
  rewrite Rlt_bool_iff in *.
  destruct_head'_or; destruct_head'_and.
  all: lra.
  all: fail. (* no goals left *)
Abort. (* so slow :-( *)
