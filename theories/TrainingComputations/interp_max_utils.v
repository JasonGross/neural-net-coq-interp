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
Import Instances.Truncating Instances.Uint63.
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

    Definition logit_delta
      (batch : N := cfg.d_vocab ^ cfg.n_ctx)
      (all_tokens : tensor [batch; cfg.n_ctx] RawIndexType)
      (all_predicted_logits : tensor [batch; cfg.n_ctx; cfg.d_vocab_out] A)
      : A
      := let predicted_logits : tensor [_;_] A
           := all_predicted_logits.[:,-1,:] in
         let indices_of_max : tensor [_;1] RawIndexType
           := Tensor.max_dim_m1 (keepdim:=true) all_tokens in
         let correct_logits
           := gather_dim_m1 predicted_logits indices_of_max in
         let logits_above_correct
           := PArray.maybe_checkpoint (correct_logits - predicted_logits) in
         let bigger_than_anything : A
           := Tensor.item (Tensor.max logits_above_correct) in
         let logits_above_correct : tensor [cfg.d_vocab ^ cfg.n_ctx : N; cfg.d_vocab_out] A
           := fun '(((tt, b), i) as idxs)
              => if i =? Tensor.item indices_of_max.[b,0]
                 then bigger_than_anything
                 else Tensor.item logits_above_correct.[b,i] in
         let min_incorrect_logit
           := Tensor.min_dim_m1 logits_above_correct in
         Tensor.item (Tensor.min min_incorrect_logit).
  End generic.

  Definition logit_delta_float_gen : _ -> _ -> float := logit_delta.

  Definition logit_delta_float : float := logit_delta all_tokens logits_all_tokens.

  (*
  Derive logit_delta_float_opt
    SuchThat (forall all_tokens all_tokensc,
                 all_tokensc = PArray.concretize all_tokens
                 -> forall all_predicted_logits all_predicted_logitsc,
                   all_predicted_logitsc = PArray.concretize all_predicted_logits
                   -> logit_delta_float_opt all_tokensc all_predicted_logitsc = logit_delta_float all_tokens all_predicted_logits)
    As logit_delta_float_opt_eq.
  Proof.
    intros.
    cbv beta delta [logit_delta_float logit_delta].
    do_red ().
    start_optimizing ().
    cbv beta iota delta [logit_delta].
    lift_lets ().
    do_red ().
    red_early_layers (); red_late_layers_1 (); red_late_layers_2 ().
    cbv beta zeta in *; do_red (); subst_local_cleanup ().
    set (logits := HookedTransformer.logits _) in all_predicted_logits.
    assert (PArray.abstract logits_all_tokens_concrete_opt = logits).
    rewrite logits_all_tokens_concrete_opt_eq.
    cbv [logits logits_all_tokens_concrete].
    Search PArray.concretize.

    cbv [] in *.
    cbn [fst snd Shape.tl Shape.hd] in *.
    red_ops ().
    do_red ().

    cbv [logits
    revert_lets_eq ().
    finish_optimizing ().
  Qed.
  Definition logit_delta_
 *)
End ModelComputations.

Module Type ModelComputationsSig (cfg : Config) (Model : ModelSig cfg) := Nop <+ ModelComputations cfg Model.
