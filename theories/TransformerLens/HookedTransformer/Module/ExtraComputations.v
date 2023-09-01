From Coq Require Vector.
From Coq Require Import Derive.
From Coq.Structures Require Import Equalities.
From Coq Require Import PreOmega Zify Lia ZifyUint63 Sint63 Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Import PrimitiveProd.
From NeuralNetInterp.Util.Tactics Require Import IsUint63 ChangeInAll ClearAll BreakMatch.
From NeuralNetInterp.Util Require Export Default Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor Slicing Tensor.Proofs Tensor.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Config HookedTransformer.Module.
Import Instances.Truncating.
#[local] Open Scope core_scope.

Module ModelComputations (cfg : Config) (Import Model : ModelSig cfg).
  Import (hints) cfg.
  Import Optimize.

  (*
  Set Printing Implicit.
  Set Printing Coercions.
  Let foo := (Unembed.forward (HookedTransformer.resid_postembed all_tokens)).[:,-1,:]%tensor : tensor _ float.
  Print foo.*)

  Notation residual_error_m1_all_tokens_gen with_checkpoint A coer_float coer_Z addA mulA
    := ((@Unembed.forward 1 [of_Z (Z.of_N (@pow N N N N_has_pow cfg.d_vocab cfg.n_ctx))] (of_Z (Z.of_N cfg.n_ctx)) A coer_float addA mulA (@HookedTransformer.resid_postembed 1 [of_Z (Z.of_N (@pow N N N N_has_pow cfg.d_vocab cfg.n_ctx))] (of_Z (Z.of_N cfg.n_ctx)) A coer_float coer_Z addA with_checkpoint (@all_tokens with_checkpoint))).[:,single_index (inject_int (-1)%sint63),:]%tensor).

  Definition residual_error_m1_all_tokens {A with_checkpoint coer_float coer_Z addA mulA}
    := residual_error_m1_all_tokens_gen with_checkpoint A coer_float coer_Z addA mulA.

  Notation residual_error_m1_all_tokens_float
    := (residual_error_m1_all_tokens_gen false float (@coer_refl float) coer_Z_float float_has_add float_has_mul).

  Definition residual_error_m1_all_tokens_concrete : PArray.concrete_tensor _ float
    := PArray.concretize residual_error_m1_all_tokens_float.

  Derive residual_error_m1_all_tokens_concrete_opt
    SuchThat (residual_error_m1_all_tokens_concrete_opt = residual_error_m1_all_tokens_concrete)
    As residual_error_m1_all_tokens_concrete_opt_eq.
  Proof.
    start_optimizing ().
    do_red ().
    red_early_layers (); red_late_layers_1 (); red_late_layers_2 ().
    cbv zeta in embed, pos_embed |- *; do_red ().
    subst_local_cleanup ().
    clear_all.
    red_ops ().
    subst embed pos_embed all_tokens0 all_toks; cbn [fst snd].
    symmetry; etransitivity.
    { repeat match goal with
             | [ |- PArray.concretize _ = _ ]
               => eapply Tensor.PArray.concretize_Proper; intro;
                  instantiate (1:=ltac:(intro)); cbv beta
             | [ |- Wf_Uint63.Reduction.sum _ _ _ _ = _ ]
               => eapply Wf_Uint63.Reduction.sum_ext;
                  [ intro; instantiate (1:=ltac:(intro)); cbv beta
                  | reflexivity
                  | repeat intro; subst; reflexivity ]
             | [ |- (?x + ?y)%float = _ ] => apply f_equal2
             | [ |- (?x * ?y)%float = _ ] => apply f_equal2
             | [ |- cfg.b_U _ = _ ] => reflexivity
             | [ |- cfg.W_U _ = _ ] => reflexivity
             | [ |- cfg.W_pos _ = _ ] => reflexivity
             end.
      rewrite raw_get_cartesian_exp_app, raw_get_arange_app.
      cbv [RawIndex.tl RawIndex.hd]; cbn [fst snd].
      cbv [Classes.zero Classes.sub Classes.mul Classes.add int_has_add int_has_mul Classes.zero Classes.one int_has_zero Classes.eqb int_has_eqb Classes.int_div Z_has_int_div].
      repeat match goal with
             | [ |- context[(?x / 1)%uint63] ]
               => replace (x / 1)%uint63 with x by lia
             | [ |- context[(?x - 0)%uint63] ]
               => replace (x - 0)%uint63 with x by lia
             | [ |- context[(0 + ?x)%uint63] ]
               => replace (0 + x)%uint63 with x by (generalize x; clear; intros; lia)
             | [ |- context[(?x * 1)%uint63] ]
               => replace (x * 1)%uint63 with x by (generalize x; clear; intros; lia)
             | [ |- context[(1 + (?x - 1))%uint63] ]
               => replace (1 + (x - 1))%uint63 with x by lia
             | [ |- context[((?x mod ?y) mod ?y)%uint63] ]
               => replace ((x mod y) mod y)%uint63 with (x mod y)%uint63
                 by (assert ((x mod y) / y = 0)%uint63; nia)
             | [ |- context[((if ?b then ?x else ?y) mod ?m)%uint63] ]
               => replace ((if b then x else y) mod m)%uint63 with (if b then x mod m else y mod m)%uint63 by (now destruct b);
                  let x' := open_constr:(_) in
                  replace (x)%uint63 with x' by (clear_all; vm_compute; reflexivity);
                  replace (x' mod m)%uint63 with x' by (clear_all; assert (x' / m = 0)%uint63; nia)
             end.
      rewrite !of_Z_spec.
      reflexivity. }
    cbv beta.
    set (denom := Z.pow _ _).
    set (d_vocab' := Z.modulo _ _).
    set (m1 := Uint63.mod _ _).

    symmetry.
    revert_lets_eq ().
    finish_optimizing ().
  Qed.
End ModelComputations.
