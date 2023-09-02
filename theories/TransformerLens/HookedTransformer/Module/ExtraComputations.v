From Coq Require Vector.
From Coq Require Import Derive.
From Coq.Structures Require Import Equalities.
From Coq Require Import PreOmega Zify Lia ZifyUint63 Sint63 Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Import PrimitiveProd.
From NeuralNetInterp.Util.Tactics Require Import IsUint63 ChangeInAll ClearAll BreakMatch.
From NeuralNetInterp.Util Require Export Default Pointed.
From NeuralNetInterp.Util Require Import Wf_Uint63.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor Slicing Tensor.Proofs Tensor.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Config HookedTransformer.Module.
Import Instances.Truncating.
#[local] Open Scope tensor_scope.
#[local] Open Scope core_scope.

Module ModelComputations (cfg : Config) (Import Model : ModelSig cfg).
  Import (hints) cfg.
  Import Optimize.

  Section generic.
    Context {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {sqrtA : has_sqrt A} {expA : has_exp A}
      {maxA : has_max A} {minA : has_min A}
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (defaultA : pointed A := @coer _ _ coerZ point).
    Let coerA' (x : float) : A := coer x.
    #[local] Coercion coerA' : float >-> A.
    #[local] Existing Instance defaultA.
    #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).

    Definition residual_error_m1 : tensor [cfg.d_vocab; cfg.d_vocab_out] A
      := let W_E : tensor _ A := cfg.W_E in
         let W_pos : tensor _ A := cfg.W_pos in
         let W_U : tensor _ A := cfg.W_U in
         let resid_postembed : tensor [cfg.d_vocab; cfg.d_model] A
           := W_E + W_pos.[slice:(-1:),:] in
         resid_postembed *m W_U.

    Definition centered_residual_error_m1 : tensor [cfg.d_vocab; cfg.d_vocab_out] A
      := let err := checkpoint residual_error_m1 in
         err - (Tensor.diagonal err).[:,None].

    Definition centered_residual_error_m1_0_min_max : A * A
      := let err := centered_residual_error_m1.[:,0] in
         let with_f f := Tensor.item (reduce_axis_m1 (keepdim:=false) f err) in
         (with_f Reduction.min, with_f Reduction.max).

    Definition centered_residual_error_m1_pos_min_max : A * A
      := let err := centered_residual_error_m1.[:,slice:(1:)] in
         let with_f f := Tensor.item (reduce_axis_m1 (keepdim:=false) f (reduce_axis_m1 (keepdim:=false) f err)) in
         (with_f Reduction.min, with_f Reduction.max).
  End generic.

  Definition centered_residual_error_m1_float : tensor _ float := centered_residual_error_m1.

  Definition centered_residual_error_m1_concrete : PArray.concrete_tensor _ float
    := PArray.concretize centered_residual_error_m1_float.

  Derive centered_residual_error_m1_concrete_opt
    SuchThat (centered_residual_error_m1_concrete_opt = centered_residual_error_m1_concrete)
    As centered_residual_error_m1_concrete_opt_eq.
  Proof.
    start_optimizing ().
    cbv [centered_residual_error_m1_float].
    cbv beta iota delta [centered_residual_error_m1 residual_error_m1].
    do_red ().
    red_early_layers (); red_late_layers_1 (); red_late_layers_2 ().
    cbv beta zeta in *; do_red (); subst_local_cleanup ().
    subst W_U W_E W_pos.
    red_ops ().
    do_red ().
    red_sum ().
    red_ops ().
    revert_lets_eq ().
    finish_optimizing ().
  Qed.
(*

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
*)
End ModelComputations.
