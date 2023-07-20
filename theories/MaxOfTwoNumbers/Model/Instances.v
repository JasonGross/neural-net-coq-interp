From Coq Require Import Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default (* Pointed PArray PArray.Instances Wf_Uint63.Instances List Notations Arith.Classes Arith.Instances Bool*) SolveProperEqRel.
(*From NeuralNetInterp.Util.Tactics Require Import DestructHead.*)
(*From NeuralNetInterp.Util Require Nat Wf_Uint63.*)
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances.
From NeuralNetInterp.MaxOfTwoNumbers Require Import Model.
(*
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
*)
(*Import ListNotations.
Local Open Scope raw_tensor_scope.
#[local] Generalizable All Variables.*)

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
  Notation model_Proper := logits_Proper (only parsing).

  #[export] Instance loss_fn_Proper {r batch return_per_token}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@loss_fn r batch return_per_token).
  Proof. cbv [loss_fn]; HookedTransformer.t. Qed.

  #[export] Instance acc_fn_Proper {r batch return_per_token}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@acc_fn r batch return_per_token).
  Proof. cbv [acc_fn]; HookedTransformer.t. Qed.
End Model.
Export (hints) Model.
