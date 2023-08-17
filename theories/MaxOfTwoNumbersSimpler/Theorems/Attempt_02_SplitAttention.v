From Coq Require Import Reals Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util.Tactics Require Import IsFloat IsUint63.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63 Wf_Uint63.Instances SolveProperEqRel Default.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.Definitions Reals.Definitions.
From NeuralNetInterp.Torch Require Import Tensor.Instances Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model Heuristics TheoremStatement Model.Instances Model.Flocqify Model.Rify.
Import LoopNotation.
(*From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import AllLogits.*)
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Local Ltac let_bind_1 _ :=
  match goal with
  | [ |- context G[let n := ?v in @?f n] ]
    => let n' := fresh n in
       set (n' := v); let G := context G[f n'] in change G; cbv beta
  end.

Local Ltac let_bind_hyp _ :=
  match goal with
  | [ H := context G[let n := ?v in @?f n] |- _ ]
    => let n' := fresh n in
       set (n' := v) in (value of H); let G := context G[f n'] in change G in (value of H); cbv beta in H
  end.

Local Ltac let_bind _ := repeat first [ progress cbv beta iota in * | let_bind_1 () | let_bind_hyp () ].

Local Ltac let_bind_subst_shape _ :=
  let_bind ();
  repeat match goal with H : Shape _ |- _ => subst H end;
  repeat match goal with H : forall b : bool, Shape _ |- _ => subst H end;
  repeat match goal with H := ?x |- _ => is_var x; subst H end;
  repeat match goal with H := ?x, H' := ?y |- _ => constr_eq x y; change H' with H in *; clear H' end.

#[export] Existing Instance reflexive_eq_dom_reflexive.
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
Local Ltac Proper_Tensor_eqf_t_step _
  := match goal with
     | [ |- Proper (_ ==> _) (fun x => _) ]
       => intros ???
     | [ |- (_ ==> _)%signature (fun x => _) (fun y => _) ]
       => intros ???
     | [ |- Tensor.eqfR _ (fun x => _) (fun y => _) ]
       => intro
     | [ |- ?R (match ?x with _ => _ end) (match ?y with _ => _ end) ]
       => tryif constr_eq x y
         then destruct x eqn:?; subst
         else destruct x eqn:?, y eqn:?; subst
     | [ H : Tensor.eqfR _ ?x ?y |- _ ]
       => move H at top;
          first [ is_var x; generalize dependent x; intros x H; setoid_rewrite H; clear H x; intros
                | is_var y; generalize dependent y; intros y H; setoid_rewrite <- H; clear H y; intros ]
     | [ |- ?R ?x ?x ] => reflexivity
     end.
Local Ltac Proper_Tensor_eqf_t _ := repeat Proper_Tensor_eqf_t_step ().
#[export] Hint Extern 1 (Proper (_ ==> _) (fun _ => _)) => progress Proper_Tensor_eqf_t () : typeclass_instances.
Local Ltac setoid_rewrite_in_body rev R lem H :=
  let rewrite_or_error rev lem :=
    let rew := match rev with
               | true => fun _ => rewrite <- lem
               | false => fun _ => rewrite -> lem
               end (*_ := (rewrite lem) (* first [ rewrite lem | setoid_rewrite lem ] *) *) in
    tryif rew ()
    then idtac
    else (match goal with
          | [ |- ?G ]
            => let lemo := open_constr:(lem) in
               let T := type of lemo in
               idtac "Could not rewrite" lem ":" T "in goal" G
          end;
          rew ()) in
  let ty := open_constr:(_) in
  let H' := fresh H in
  rename H into H';
  evar (H : ty);
  let lemH := fresh in
  assert (lemH : R H' H) by (subst H'; rewrite_or_error rev lem; subst H; reflexivity);
  let rec do_rewrite _
    := lazymatch goal with
       | [ H'' := context[H'] |- _ ]
         => setoid_rewrite_in_body false R lemH H'';
            do_rewrite ()
       | _ => idtac
       end in
  do_rewrite ();
  lazymatch goal with
  | [ |- context[H'] ] => rewrite_or_error false lemH
  | _ => idtac
  end;
  clear lemH; clear H'.
Tactic Notation "setoid_rewrite" "(" uconstr(R) ")" uconstr(lem) "in" "(" "value" "of" hyp(H) ")" := setoid_rewrite_in_body false R lem H.
Tactic Notation "setoid_rewrite" "(" uconstr(R) ")" "<-" uconstr(lem) "in" "(" "value" "of" hyp(H) ")" := setoid_rewrite_in_body true R lem H.

#[local] Set Warnings Append "undeclared-scope".
From Interval Require Import Tactic_float.
#[local] Set Warnings Append "+undeclared-scope".

From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From Flocq Require Import Raux.
From NeuralNetInterp.Util.Arith Require Import Flocq Flocq.Instances Flocq.Notations.

Theorem good_accuracy : TheoremStatement.Accuracy.best (* (abs (real_accuracy - expected_accuracy) <? error)%float = true *).
Proof.
  cbv [real_accuracy].
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
  | [ H : (abs (B2R ?x - ?y) <=? ?err)%core = true |- (abs (?x' / ?e - ?one) <=? ?err')%core = true ]
    => first [ constr_eq x x' | fail 1 x "≠" x' ];
       cut ((abs ((y:R) / (e:R) - (one:R)) <=? ((err':R) + (err:R) / (e:R))) = true)%core;
       cbv beta; [ | clear H ];
       [ revert H; generalize x y err err' e one | ]
  end.
  { cbv [Classes.abs Classes.leb Classes.sub Classes.div
           R_has_abs R_has_leb R_has_sub R_has_div
           binary_float_has_abs binary_float_has_leb binary_float_has_sub binary_float_has_div].
    intros.
    rewrite Bleb_correct_full, B2R_Babs, is_finite_Babs, B2R_Bminus, is_nan_Babs, Bsign_Babs, B2R_Bdiv.
    admit. (* XXX FIXME TODO rounding error *)  }
  set (m := acc_fn _ _).
  set (m' := m).
  assert (m' = raw_get m) by (clearbody m; reflexivity).
  clearbody m'; subst m' m.
  cbv beta iota delta [acc_fn]; let_bind_subst_shape ().
  cbv beta iota delta [logits HookedTransformer.logits HookedTransformer.HookedTransformer.logits] in *; let_bind_subst_shape ().
  cbv beta iota delta [coer_blocks_params cfg.blocks_params] in *.
  cbv beta iota delta [HookedTransformer.blocks_cps HookedTransformer.HookedTransformer.blocks_cps fold_right Datatypes.length List.firstn HookedTransformer.HookedTransformer.blocks HookedTransformer.blocks List.map] in *; let_bind_subst_shape ().
  vm_compute Shape.tl in *.
  vm_compute of_Z in *.
  vm_compute SliceIndex.transfer_shape in *.
  vm_compute Shape.app in *.
  vm_compute Shape.broadcast2 in *.
  cbv beta iota delta [TransformerBlock.attn_only_out] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.attn_out] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.z] in *; let_bind_subst_shape ().
  set (v := Attention.v _ _ _) in *.
  set (pattern := Attention.pattern _ _ _ _ _ _) in *.
  cbv beta iota delta [HookedTransformer.HookedTransformer.unembed HookedTransformer.unembed] in *; let_bind_subst_shape ().
  cbv beta iota delta [Unembed.forward] in *; let_bind_subst_shape ().
  cbv beta iota delta [all_tokens] in true_maximum; let_bind_subst_shape ().
  cbv beta iota delta [Model.all_tokens] in *; let_bind_subst_shape ().
  cbv beta iota delta [HookedTransformer.HookedTransformer.ln_final HookedTransformer.ln_final] in *; let_bind_subst_shape ().
  cbv beta iota delta [cfg.normalization_type TransformerBlock.ln1] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.pattern Attention.masked_attn_scores Attention.apply_causal_mask Attention.attn_scores] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.v Attention.q Attention.k] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.einsum_input TransformerBlock.add_head_dimension TransformerBlock.value_input TransformerBlock.query_input TransformerBlock.key_input] in *; let_bind_subst_shape ().
  cbv beta iota delta [softmax_dim_m1] in *; let_bind_subst_shape ().
  cbv beta iota delta [HookedTransformer.Attention.masked_attn_scores Attention.apply_causal_mask HookedTransformer.Attention.attn_scores] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.q Attention.k Attention.v] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.einsum_input] in *; let_bind_subst_shape ().
  cbv beta iota delta [HookedTransformer.HookedTransformer.resid_postembed HookedTransformer.HookedTransformer.embed HookedTransformer.HookedTransformer.pos_embed HookedTransformer.Embed.forward HookedTransformer.PosEmbed.forward] in *; let_bind_subst_shape ().

  cbv beta iota in *.

  cbv beta iota delta [of_bool map map2] in res.
  move true_maximum at bottom.
  cbv beta iota delta [reduce_axis_m1 reduce_axis_m1' reshape_snoc_split RawIndex.curry_radd RawIndex.combine_radd map RawIndex] in true_maximum.
  cbv beta iota delta [Reduction.max max has_default_max_leb leb] in true_maximum.
  rename all_tokens1 into all_tokens.
  cbv -[PrimInt63.leb all_tokens] in true_maximum.
  move out at bottom.
  cbv beta iota delta [HookedTransformer.Unembed.forward] in *.
  cbv [map' map2' reshape_app_combine reshape_app_combine' RawIndex.uncurry_radd map reshape_app_split reshape_app_split' RawIndex.curry_radd map2 raw_get get reduce_axis_m1 reduce_axis_m1' map RawIndex.split_radd RawIndex.combine_radd RawIndex.snoc RawIndex.nil reshape_snoc_split reshape_snoc_combine map' broadcast broadcast' repeat' repeat Nat.radd Shape.nil] in *.
  pose (fun i => res (tt, i)) as res'.
  set (me := mean).
  replace (me res tt) with (me (fun i => res' (RawIndex.tl i)) tt); subst me.
  2: { subst res'; cbv [mean]; set (k:=Reduction.mean); clearbody res k; clear.
       cbv -[of_Z to_Z Z.modulo Z.opp Z.mul]; reflexivity. }
  subst res.
  cbv beta in *.
  Ltac strip_one_tt H :=
    let Hv := (eval cbv delta [H] in H) in
    lazymatch Hv with
    | context[?f (tt, _)]
      => lazymatch Hv with
         | context C[f]
           => let f' := fresh f in
              rename f into f';
              pose (fun i => f' (tt, i)) as f;
              let T := type of f' in
              let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
              let C := context C[fun i : T => f (snd i)] in
              change C in (value of H); cbv [f'] in f; clear f'; cbn [snd] in H, f
         end
    end.
  Ltac strip_two_tt H :=
    let Hv := (eval cbv delta [H] in H) in
    lazymatch Hv with
    | context[?f ((tt, _), _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i j => f' ((tt, i), j)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         repeat lazymatch (eval cbv delta [H] in H) with
           | context C[f']
             => let C := context C[fun i : T => f (snd (fst i)) (snd i)] in
                progress change C in (value of H); cbn [fst snd] in H
           end;
         clear f'
    end.
  Ltac strip_three_tt H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f (((tt, _), _), _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i j k => f' (((tt, i), j), k)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         lazymatch Hv with
         | context C[f']
           => let C := context C[fun i : T => f (snd (fst (fst i))) (snd (fst i)) (snd i)] in
              progress change C in (value of H); cbn [fst snd] in H
         end;
         clear f'
    end.
  Ltac strip_three_tt' H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f (((tt, _), _), _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i j k => f' (((tt, i), j), k)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         idtac f;
         repeat match goal with
           | [ H' := context C[f'] |- _ ]
             => let body := (eval cbv delta [H'] in H') in
                lazymatch body with context[f' (((tt, _), _), _)] => idtac end;
                let C := context C[fun i : T => f (snd (fst (fst i))) (snd (fst i)) (snd i)] in
                progress change C in (value of H'); cbn [fst snd] in H'
           end;
         clear f'
    end.
  Ltac strip_two_tt' H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f ((tt, _), _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i j => f' ((tt, i), j)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         idtac f;
         repeat match goal with
           | [ H' := context C[f'] |- _ ]
             => let body := (eval cbv delta [H'] in H') in
                lazymatch body with context[f' ((tt, _), _)] => idtac end;
                let C := context C[fun i : T => f (snd (fst i)) (snd i)] in
                progress change C in (value of H'); cbn [fst snd] in H'
           end;
         clear f'
    end.
  Ltac strip_one_tt' H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f (tt, _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i => f' (tt, i)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         idtac f;
         repeat match goal with
           | [ H' := context C[f'] |- _ ]
             => let body := (eval cbv delta [H'] in H') in
                lazymatch body with context[f' (tt, _)] => idtac end;
                let C := context C[fun i : T => f (snd i)] in
                progress change C in (value of H'); cbn [fst snd] in H'
           end;
         clear f'
    end.
  Ltac move_int_const_early_2 H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f _ ?x]
      => is_uint63 x;
         let f' := fresh f in
         rename f into f';
         pose (fun a b => f' b a) as f;
         repeat match goal with
           | [ H' := context[f'] |- _ ]
             => assert_fails constr_eq H' f;
                change f' with (fun a b => f b a) in H'
           end;
         cbv [f'] in f; clear f'; cbv beta in *
    end.
  Ltac strip_const H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f ?x]
      => is_uint63 x; is_var f;
         let f' := fresh f in
         rename f into f';
         set (f := f' x) in *;
         cbv [f'] in f; clear f'
    end.
  Ltac strip_four_tt H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f ((((tt, _), _), _), _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i j k l => f' ((((tt, i), j), k), l)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         lazymatch Hv with
         | context C[f']
           => let C := context C[fun i : T => f (snd (fst (fst (fst i)))) (snd (fst (fst i))) (snd (fst i)) (snd i)] in
              progress change C in (value of H); cbn [fst snd] in H
         end;
         clear f'
    end.
  Ltac strip_four_tt' H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f ((((tt, _), _), _), _)]
      => let f' := fresh f in
         rename f into f';
         let Hv := (eval cbv delta [H] in H) in
         pose (fun i j k l => f' ((((tt, i), j), k), l)) as f;
         let T := type of f' in
         let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
         cbv [f'] in f;
         repeat match goal with
           | [ H' := context C[f'] |- _ ]
             => idtac H';
                let C := context C[fun i : T => f (snd (fst (fst (fst i)))) (snd (fst (fst i))) (snd (fst i)) (snd i)] in
                progress change C in (value of H'); cbn [fst snd] in H'
           end;
         clear f'
    end.
  Ltac move_int_const_early_3 H :=
    let Hv := (eval cbv delta [H] in H) in
    match Hv with
    | context[?f _ _ ?x]
      => is_uint63 x;
         let f' := fresh f in
         rename f into f';
         pose (fun c a b => f' a b c) as f;
         change f' with (fun a b c => f c a b) in H;
         cbv [f'] in f; clear f'; cbv beta in *
    end.
  repeat match goal with H : _ |- _ => strip_one_tt H end.
  move all_tokens at bottom.
  repeat match goal with H : _ |- _ => strip_two_tt H end.
  cbv -[logits0] in pred_logits.
  subst logits0; cbv beta in *.
  cbv [get raw_get Classes.add tensor_add map2 R_has_add] in pred_logits.
  repeat match goal with H : _ |- _ => strip_three_tt H end.
  cbv [tensor_add Classes.add Classes.div tensor_div_by map2 R_has_add R_has_div] in *.
  subst residual0.
  cbv beta in *.
  repeat match goal with H : _ |- _ => strip_three_tt H end.
  move attn_out at bottom.
  move_int_const_early_2 pred_logits.
  strip_const pred_logits.
  move_int_const_early_2 attn_out.
  strip_const attn_out.
  subst attn_out.
  subst out.
  cbv beta in *.
  repeat match goal with H : _ |- _ => strip_four_tt' H end.
  repeat match goal with H : _ |- _ => strip_two_tt' H end.
  repeat match goal with H : _ |- _ => strip_one_tt' H end.
  cbv [RawIndex.hd RawIndex.nil Shape.nil] in *.
  cbn [fst snd] in *.

  move_int_const_early_3 pred_logits.
  strip_const pred_logits.
  move_int_const_early_3 pattern.
  strip_const pattern.
  subst sum_exp_t.
  move_int_const_early_3 pattern.
  strip_const pattern.
  cbv [Nat.radd] in *.
  move qk at bottom.
  subst qk.
  cbv [where_] in *.
  cbv [map3] in *.
  repeat match goal with H : _ |- _ => strip_three_tt' H end.
  move_int_const_early_2 exp_t.
  vm_compute in mask.
  cbv [SliceIndex.slice SliceIndex.SliceIndexType.slice Slice.invert_index RawIndex.tl RawIndex.hd SliceIndex.transfer_shape RawIndex.snoc RawIndex.nil] in *.
  cbn [fst snd] in *.
  strip_two_tt exp_t.
  strip_const exp_t.
  set (s2 := Reduction.sum 0 2 1) in *.
  set (s1 := Reduction.sum 0 1 1) in *.
  cbv -[Rplus] in s2, s1.
  subst s1 s2; cbv beta in *.
  move_int_const_early_2 pred_logits.
  move_int_const_early_2 pred_logits.
  strip_const pred_logits.
  move_int_const_early_3 pred_logits.
  strip_const pred_logits.
  move_int_const_early_2 pred_logits.
  move_int_const_early_2 pattern.
  strip_const pattern.
  move_int_const_early_2 pattern.
  move residual at bottom.
  cbv [Classes.exp R_has_exp] in *.
  cbv [FancyIndex.slice reshape_app_combine' FancyIndex.slice_ RawIndex.uncurry_radd RawIndex.split_radd map_dep SliceIndex.slice Shape.snoc Shape.nil map2 SliceIndex.SliceIndexType.slice repeat' FancyIndex.FancyIndexType.broadcast map Shape.tl Shape.hd adjust_index_for Slice.invert_index RawIndex.tl FancyIndex.broadcast] in *.
  cbn [fst snd] in *.
  repeat match goal with H : _ |- _ => strip_two_tt' H end.
  move_int_const_early_2 true_maximum.
  cbv [cartesian_exp cartesian_nprod Shape.Tuple.init] in *.
  cbv [Uint63.coer_int_N'] in *.
  vm_compute of_Z in all_tokens.
  vm_compute Z.to_N in all_tokens.
  vm_compute N.to_nat in all_tokens.
  cbv [reshape_all ntupleify Shape.Tuple.init' ntupleify' Shape.repeat RawIndex.unreshape RawIndex.unreshape' Shape.Tuple.nth_default Shape.Tuple.to_list Shape.Tuple.to_list' RawIndex.snoc RawIndex.nil RawIndex.hd RawIndex.tl RawIndex.item Shape.tl Shape.hd Shape.snoc Shape.nil get raw_get] in all_tokens.
  cbn [fst snd] in *.
  repeat (set (k := to_Z _) in (value of all_tokens) at 1; vm_compute in k; subst k).
  cbv [Classes.int_div Z_has_int_div] in *.
  strip_one_tt' all_tokens.
  cbv [arange] in all_toks0.
  cbv [RawIndex.item RawIndex.tl] in *.
  cbn [fst snd] in *.
  subst all_toks0.
  cbv beta in *.
  vm_compute in mask.
  move true_maximum at bottom.
  subst pos_embed0; cbv beta in *.
  Ltac change01 f :=
    let f' := fresh f in
    rename f into f';
    pose (fun i => if (i =? 0)%uint63 then f' 0%uint63 else f' 1%uint63) as f;
    repeat match goal with
      | [ H := context[f'] |- _ ]
        => assert_fails constr_eq H f;
           progress (change (f' 0%uint63) with (f 0%uint63) in (value of H);
                     change (f' 1%uint63) with (f 1%uint63) in (value of H))
      end;
    match goal with
    | [ H := context[f'] |- _ ]
      => assert_fails constr_eq H f;
         fail 1 H
    | _ => idtac
    end;
    subst f'; cbv beta in f.
  change01 pattern.
  change01 exp_t.
  change01 v.
  change01 residual.
  move_int_const_early_2 residual.
  change01 embed0.
  change01 all_tokens.
  change01 mask.
  vm_compute in mask.
  vm_compute Z.to_nat in all_tokens.
  cbv [nth_default nth_error] in all_tokens.
  subst embed0; cbv beta iota in *.
  cbv [Uint63.eqb] in *.
  HERE
    (*
  move residual at bottom.
  change01 residual.
         progress (change (f' 0%uint63) with (f 0%uint63) in (value of H);
  rename
  match goal with
  | [ H := context[?f 0%uint63] |- _ ]
    => is_var f;
       lazymatch goal with
       | [ f' := fun x => if (x =? 0)%uint63 then f else _ ] =>
  vm_compute to_Z in all_tokens.

  cbv -[to_Z Z.to_nat all_toks0] in all_tokens.
  strip_two_tt'
  Set Printing All.
  move_int_const_early_3 pattern.

  move_int_const_early_3 pred_logits.

  unfold Reduction.sum in (va
  subst mask3; cbv beta iota in *.
  cbv
  cbv [ones tril] in *.
  cbv [mask]
  strip_three_tt' exp_t.


  strip_three_tt pred_logits.
  subst logits0.
  cbv beta iota delta [map' reshape_app_combine reshape_app_combine' RawIndex.uncurry_radd map] in *.


  (*
  cbv -[map2' Reduction.sum L0_attn_W_O
          Instances.binary_float_has_add prec emax Hprec Hmax Instances.binary_float_has_mul Classes.add Classes.mul Classes.zero Classes.coer Instances.coer_float_binary_float
          map2' raw_get v pattern L0_attn_W_O RawIndex.snoc RawIndex.nil
          Instances.binary_float_has_add prec emax Hprec Hmax Instances.binary_float_has_mul Classes.add Classes.mul Classes.zero Reduction.sum] in out.
  unfold Reduction.sum in (value of out) at 1.
  unfold Reduction.sum in (value of out) at 2.
  cbv -[map2' Reduction.sum L0_attn_W_O
          Instances.binary_float_has_add prec emax Hprec Hmax Instances.binary_float_has_mul Classes.add Classes.mul Classes.zero Classes.coer Instances.coer_float_binary_float
          map2' raw_get v pattern L0_attn_W_O RawIndex.snoc RawIndex.nil
          Instances.binary_float_has_add prec emax Hprec Hmax Instances.binary_float_has_mul Classes.add Classes.mul Classes.zero Reduction.sum] in out.
  cbv [mean Reduction.mean reduce_axis_m1 reduce_axis_m1' map item SliceIndex.slice raw_get Truncating.coer_Z_float Shape.reshape' Shape.reduce Shape.tl snd reshape_m1 reshape_snoc_split].
  vm_compute of_Z.
  vm_compute Uint63.to_Z.
  set (v_ := coer _).
  rewrite <- (Prim2B_B2Prim v_).
  set (v__ := B2Prim v_).
  subst v_.
  vm_compute in v__.
  subst v__.
  cbv [RawIndex.unreshape RawIndex.unreshape' RawIndex.curry_radd RawIndex.combine_radd RawIndex.item RawIndex.snoc RawIndex.tl RawIndex.nil snd Shape.tl].
  cbv [expected_accuracy].
  cbv [error].

  subst all_toks_c.
  cbv beta iota delta [cartesian_prod] in true_maximum.
  cbv beta iota delta [reshape_m1 modulo PrimInt63.mod Uint63.to_Z Uint63.to_Z_rec Uint63.size Uint63.is_even Uint63.is_zero Uint63.land Uint63.eqb Uint63.compare PrimInt63.lsr Z.double Z.succ_double Z.to_nat nth_default nth_error Pos.to_nat Pos.iter_op] in true_maximum.
  cbv [raw_get] in true_maximum.
  vm_compute PrimInt63.add in true_maximum.
  cbv [tupleify] in true_maximum.
  cbv [RawIndex.unreshape] in true_maximum.
  cbv [RawIndex.unreshape'] in true_maximum.
  cbv [RawIndex.item RawIndex.tl RawIndex.snoc RawIndex.nil fst snd Shape.hd RawIndex.hd Shape.tl Shape.snoc Shape.nil] in true_maximum.
  move all_toks at bottom.
  vm_compute of_Z in all_toks.
  cbv [arange] in all_toks.
  subst all_toks.
  cbv [RawIndex.item RawIndex.tl raw_get] in true_maximum.
  cbn [snd] in true_maximum.
  pose (fun idx => res (tt, idx)) as res'.
  lazymatch goal with
  | [ |- context G[res] ]
    => let G := context G[fun v : RawIndex 1 => res' (snd v)] in
       change G; cbn [snd]
  end.
  subst res; rename res' into res; cbv beta in res.
  set (res' := fun i => res _); subst res; rename res' into res; cbv beta in res.
  pose (fun i => true_maximum (@pair unit RawIndex.RawIndexType.t tt (of_Z (Z.modulo (to_Z i) (to_Z 0x1000))))) as true_maximum'.
  lazymatch (eval cbv delta [res] in res) with
  | fun i : ?ty => if ?f (true_maximum _) then ?T else ?F
    => let C := constr:(fun i : ty => if f (true_maximum' i) then T else F) in
       change C in (value of res); subst true_maximum; rename true_maximum' into true_maximum;
       cbn [snd] in res, true_maximum
  end.
  pose (fun i => pred_tokens (@pair unit RawIndex.RawIndexType.t tt (of_Z (Z.modulo (to_Z i) (to_Z 0x1000))))) as pred_tokens'.
  lazymatch (eval cbv delta [res] in res) with
  | fun i : ?ty => if ?f (pred_tokens _) ?x then ?T else ?F
    => let C := constr:(fun i : ty => if f (pred_tokens' i) x then T else F) in
       change C in (value of res); subst pred_tokens; rename pred_tokens' into pred_tokens;
       cbn [snd] in res, pred_tokens
  end.


  (* remove ln *)
  cbv [TransformerBlock.ln1 HookedTransformer.ln_final point default_Z] in *.
  repeat match goal with H := coer 0%Z |- _ => subst H end.
  repeat match goal with H := ?x, H' := ?y |- _ => constr_eq x y; change H' with H in *; clear H' end.

  cbv [SliceIndex.slice SliceIndex.SliceIndexType.slice Shape.tl Shape.hd Shape.snoc Shape.nil Slice.invert_index RawIndex.snoc RawIndex.hd adjust_index_for RawIndex.tl RawIndex Nat.radd PrimInt63.mod] in pred_logits; cbn [fst snd] in pred_logits.
  set (k := PrimInt63.mod _ _) in (value of pred_logits).
  vm_compute in k; subst k.

  cbv [SliceIndex.slice SliceIndex.SliceIndexType.slice Shape.tl Shape.hd Shape.snoc Shape.nil Slice.invert_index RawIndex.snoc RawIndex.hd adjust_index_for RawIndex.tl RawIndex Nat.radd PrimInt63.mod RawIndex.nil map raw_get map' reshape_app_combine reshape_app_combine' RawIndex.uncurry_radd RawIndex.split_radd reshape_app_split reshape_app_split' RawIndex.curry_radd RawIndex.combine_radd broadcast broadcast' repeat' reduce_axis_m1 reduce_axis_m1' map reshape_snoc_split map2'] in *; cbn [fst snd] in *.
  strip_two_tt pred_tokens.
  strip_three_tt pred_logits.
  subst logits; cbv iota beta in pred_logits.
  cbv [add] in pred_logits.
  cbv [tensor_add] in pred_logits.
  cbv [map2] in *.
  strip_three_tt pred_logits.
  move residual0 at bottom.
  subst residual0.
  change (@Classes.add _ _ _ (@tensor_add ?r ?sA ?sB ?A ?B ?C ?a)) with (@tensor_add r sA sB A B C a) in *.
  change (@Classes.div _ _ _ (@tensor_div_by ?r ?sA ?sB ?A ?B ?C ?a)) with (@tensor_div_by r sA sB A B C a) in *.
  cbv [tensor_add tensor_div_by map2] in *.
  From NeuralNetInterp.Util.Tactics Require Import IsUint63.
  strip_const pred_logits.
  move residual at bottom.
  cbv [map2'] in *.
  strip_three_tt residual1.
  move_int_const_early_2 residual1.
  strip_const residual1.
  strip_three_tt attn_out.
  move_int_const_early_2 attn_out.
  strip_const attn_out.
  move_int_const_early_3 out.
  strip_const out.
  move_int_const_early_3 out.
  strip_const out.
  move_int_const_early_2 out.
  move_int_const_early_2 out.
  strip_const out.
  move_int_const_early_2 out.
  change (@Classes.div _ _ _ (@tensor_div_by ?r ?sA ?sB ?A ?B ?C ?a)) with (@tensor_div_by r sA sB A B C a) in *.
  cbv beta iota delta [tensor_div_by map2 where_ map3 reduce_axis_m1] in *.
  strip_four_tt' pattern.
  strip_four_tt' pattern.
  strip_four_tt' exp_t.
  move pattern at bottom.
  move_int_const_early_2 pattern.
  move_int_const_early_2 pattern.
  strip_const pattern.
  strip_const pattern.
  move_int_const_early_2 pattern.
  move_int_const_early_2 pattern.
  strip_const pattern.
  strip_const pattern.
  pose (fun x => sum_exp_t x point) as sum_exp_t'.
  change sum_exp_t with (fun x (_ : RawIndexType) => sum_exp_t' x) in pattern.
  subst sum_exp_t.
  rename sum_exp_t' into sum_exp_t.
  cbv beta in *.
  set (v0 := v 0%uint63) in *.
  set (v1 := v 1%uint63) in *.
  set (pattern0 := pattern 0%uint63) in *.
  set (pattern1 := pattern 1%uint63) in *.
  cbv beta in *.
  subst v pattern.
  cbv beta iota in *.
  move v0 at bottom.
  subst attn_out.
  subst residual1.
  cbv beta in *.
  move v1 at bottom.
  strip_three_tt' pred_logits.
  move_int_const_early_2 pred_logits.
  move true_maximum at bottom.
(*
  pose (fun i => (pred_tokens (of_Z (Z.modulo (to_Z i) (to_Z 0x1000))))) as pred_tokens'.
  l
  Set Printing All.
  move pattern0 at bottom.
  move sum_exp_t at bottom.
  move sum_exp_t at bottom.


  cbv beta iota delta [Attention.v Attention.einsum_input map' reshape_app_combine] in *; let_bind_subst_shape ().

  subst out'
  strip_const out'.

         lazymatch Hv with
         | context C[f]
           => let f' := fresh f in
              rename f into f';
              pose (fun i j k => f' (((tt, i), j), k)) as f;
              let T := type of f' in
              let T := lazymatch (eval hnf in T) with ?T -> _ => T end in
              let C := context C[fun i : T => f (snd (fst (fst i))) (snd (fst i)) (snd i)] in
              change C in (value of H); cbv [f'] in f; clear f'; cbn [fst snd] in H, f
         end
    end.
  cbv [add tensor_add map2] in residual1.
  Set Printing All.
  cbv [raw_get] in pred_logits.
  cbv [reduce_axis_m1 reduce_axis_m1' map] in pred_tokens.
  let H := pred_logits in
  let H' := fresh H in
  rename H into H';


  cbv [coer_tensor_float
  cbv [PrimInt63.mod] in pred_logits.
  Set Printing All.
  Print Uint63.mod.
  cbv [Uint63.mod] in pred_logits.
  Set Printing All.
  move pre

  Set Printing All.
  Ltac push_pair_idx H :=
    lazymatch (eval cbv [H] in H) with
    | fun i : RawIndexType
*)
*)*)
Abort.
