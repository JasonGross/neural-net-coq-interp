From Coq Require Import Morphisms RelationClasses RelationPairs.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel Option.
From NeuralNetInterp.Util.Tactics Require Import Head DestructHead.
From NeuralNetInterp.Util.List.Instances Require Import Forall2 Forall2.Map.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances HookedTransformer.Module.Instances.
From NeuralNetInterp.MaxOfTwoNumbersUndertrainedSimpler Require Import Model.
Import Dependent.ProperNotations Dependent.RelationPairsNotations.

Module Model.
  Export Model.
  Include ModelInstances cfg Model.
  Include HookedTransformer.

  Local Ltac t' :=
    try eassumption;
    auto;
    repeat intro;
    try (eapply Tensor.map_Proper_dep; try eassumption; repeat intro);
    try reflexivity;
    auto.

  (*
  #[export] Instance embed_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR) (@embed r batch pos).
  Proof. cbv [embed]; repeat intro; apply HookedTransformer.HookedTransformer.embed_Proper_dep; t'. Qed.

  #[export] Instance embed_Proper {r batch pos A coer_float}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@embed r batch pos A coer_float)
    := _.

  #[export] Instance pos_embed_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR) (@pos_embed r batch pos).
  Proof. cbv [pos_embed]; repeat intro; apply HookedTransformer.HookedTransformer.pos_embed_Proper_dep; t'. Qed.

  #[export] Instance pos_embed_Proper {r batch pos A coer_float}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@pos_embed r batch pos A coer_float)
    := _.

  #[export] Instance ln_final_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Tensor.eqfR
           ==> Tensor.eqfR)
        (@ln_final r batch pos).
  Proof. cbv [ln_final cfg.normalization_type]; repeat intro; apply HookedTransformer.HookedTransformer.ln_final_Proper_dep; t'. Qed.

  #[export] Instance ln_final_Proper {r batch pos A coer_float coerZ addA subA mulA divA sqrtA use_checkpoint}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@ln_final r batch pos A coer_float coerZ addA subA mulA divA sqrtA use_checkpoint).
  Proof. cbv [ln_final]; apply HookedTransformer.HookedTransformer.ln_final_Proper. Qed.

  #[export] Instance unembed_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Tensor.eqfR
           ==> Tensor.eqfR)
        (@unembed r batch pos).
  Proof. cbv [unembed]; repeat intro; apply HookedTransformer.HookedTransformer.unembed_Proper_dep; t'. Qed.

  #[export] Instance unembed_Proper {r batch pos A coer_float coerZ addA mulA}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@unembed r batch pos A coer_float coerZ addA mulA).
  Proof. cbv [unembed]; apply HookedTransformer.HookedTransformer.unembed_Proper. Qed.

  #[export] Instance logits_Proper {r batch pos A coer_float coerZ addA subA mulA divA sqrtA expA use_checkpoint}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@logits r batch pos A coer_float coerZ addA subA mulA divA sqrtA expA use_checkpoint)
    := HookedTransformer.HookedTransformer.logits_Proper.

  #[export] Instance masked_attn_scores_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR)
        (@masked_attn_scores r batch pos).
  Proof.
    cbv [masked_attn_scores].
    pose proof (@HookedTransformer.HookedTransformer.masked_attn_scores_Proper_dep) as H.
    repeat intro.
    repeat (let v := open_constr:(_) in specialize (H v)).
    move H at bottom.
    revert H.
    lazymatch goal with
    | [ |- ?R _ _ ?R'' ?x ?y -> ?R' (invert_Some ?x' ?i) (invert_Some ?y' ?i) ]
      => unify x x'; unify y y'; unify R'' R'; set (x'':=x); set (y'':=y);
         intro H;
         refine (@invert_Some_Proper_dep _ _ (Tensor.eqfR R') x y H i)
    end.
    Unshelve.
    all: cbv beta iota; t'.
    all: try apply HookedTransformer.coer_blocks_params_Proper_dep; t'.
    all: try (apply Dependent.reflexivity; try exact _).
    all: try apply HookedTransformer.block_params_type_genR_refl.
  Qed.

  #[export] Instance masked_attn_scores_Proper {r batch pos A coer_float coerZ addA subA mulA divA sqrtA expA use_checkpoint}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@masked_attn_scores r batch pos A coer_float coerZ addA subA mulA divA sqrtA expA use_checkpoint).
  Proof. apply masked_attn_scores_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance attn_pattern_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR)
        (@attn_pattern r batch pos).
  Proof.
    cbv [attn_pattern].
    pose proof (@HookedTransformer.HookedTransformer.attn_pattern_Proper_dep) as H.
    repeat intro.
    repeat (let v := open_constr:(_) in specialize (H v)).
    move H at bottom.
    revert H.
    lazymatch goal with
    | [ |- ?R _ _ ?R'' ?x ?y -> ?R' (invert_Some ?x' ?i) (invert_Some ?y' ?i) ]
      => unify x x'; unify y y'; unify R'' R'; set (x'':=x); set (y'':=y);
         intro H;
         refine (@invert_Some_Proper_dep _ _ (Tensor.eqfR R') x y H i)
    end.
    Unshelve.
    all: cbv beta iota; t'.
    all: try apply HookedTransformer.coer_blocks_params_Proper_dep; t'.
    all: try (apply Dependent.reflexivity; try exact _).
    all: try apply HookedTransformer.block_params_type_genR_refl.
  Qed.

  #[export] Instance attn_pattern_Proper {r batch pos A coer_float coerZ addA subA mulA divA sqrtA expA use_checkpoint}
    : Proper (Tensor.eqf ==> Tensor.eqf) (@attn_pattern r batch pos A coer_float coerZ addA subA mulA divA sqrtA expA use_checkpoint).
  Proof. apply attn_pattern_Proper_dep; repeat intro; subst; reflexivity. Qed.

  Notation model_Proper := logits_Proper (only parsing).
*)
  #[export] Instance loss_fn_Proper_dep {r batch pos return_per_token}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Tensor.eqfR
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR)
        (@loss_fn r batch pos return_per_token).
  Proof. cbv [loss_fn]; HookedTransformer.t. Qed.

  #[export] Instance loss_fn_Proper {r batch pos return_per_token A coerZ addA subA divA maxA oppA expA lnA use_checkpoint}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@loss_fn r batch pos return_per_token A coerZ addA subA divA maxA oppA expA lnA use_checkpoint).
  Proof. apply loss_fn_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance acc_fn_Proper_dep {r batch pos return_per_token}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.const eq)
           ==> Dependent.const (fun _ _ => True)
           ==> Tensor.eqfR
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR)
        (@acc_fn r batch pos return_per_token).
  Proof. cbv [acc_fn]; HookedTransformer.t; subst; reflexivity. Qed.

  #[export] Instance acc_fn_Proper {r batch pos return_per_token A coerZ addA divA ltbA use_checkpoint}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@acc_fn r batch pos return_per_token A coerZ addA divA ltbA use_checkpoint).
  Proof. apply acc_fn_Proper_dep; repeat intro; subst; reflexivity. Qed.
End Model.
Export (hints) Model.
