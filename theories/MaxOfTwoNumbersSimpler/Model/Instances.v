From Coq Require Import Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel Option.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model.

Module Model.
  Export Model.

  #[export] Instance embed_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@embed r batch pos)
    := _.
  #[export] Instance pos_embed_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@pos_embed r batch pos)
    := _.
  #[export] Instance ln_final_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@ln_final r batch pos)
    := _.
  #[export] Instance unembed_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@unembed r batch pos)
    := _.
  #[export] Instance logits_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@logits r batch pos)
    := _.
  #[export] Instance masked_attn_scores_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@masked_attn_scores r batch pos).
  Proof.
    cbv [masked_attn_scores].
    lazymatch goal with
    | [ |- Proper (?R ==> _) (fun tokens => Option.invert_Some (?f ?n tokens)) ]
      => pose proof (_ : Proper (eq ==> R ==> _) f) as H;
         specialize (H n n eq_refl)
    end.
    intros x y Ht; specialize (H _ _ Ht).
    refine (Option.invert_Some_Proper _ _ H).
  Qed.
  #[export] Instance attn_pattern_Proper {r batch pos} : Proper (Tensor.eqf ==> Tensor.eqf) (@attn_pattern r batch pos).
  Proof.
    cbv [attn_pattern].
    lazymatch goal with
    | [ |- Proper (?R ==> _) (fun tokens => Option.invert_Some (?f ?n tokens)) ]
      => pose proof (_ : Proper (eq ==> R ==> _) f) as H;
         specialize (H n n eq_refl)
    end.
    intros x y Ht; specialize (H _ _ Ht).
    refine (Option.invert_Some_Proper _ _ H).
  Qed.

  Notation model_Proper := logits_Proper (only parsing).

  #[export] Instance loss_fn_Proper {r batch return_per_token}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@loss_fn r batch return_per_token).
  Proof. cbv [loss_fn]; HookedTransformer.t. Qed.

  #[export] Instance acc_fn_Proper {r batch return_per_token}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@acc_fn r batch return_per_token).
  Proof. cbv [acc_fn]; HookedTransformer.t. Qed.
End Model.
Export (hints) Model.
