From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp Require Import max_parameters max max_heuristics.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.
Local Open Scope core_scope.

Definition total : int := Eval vm_compute in Shape.item (Shape.reshape (shape_of all_tokens)).
Definition expected_correct : int := Eval vm_compute in total - Uint63.of_Z (List.length heuristics.incorrect_results).
Definition totalf : float := Eval vm_compute in PrimFloat.of_uint63 total.
Definition expected_correctf : float := Eval vm_compute in PrimFloat.of_uint63 expected_correct.
Definition expected_accuracy : float := Eval vm_compute in expected_correctf / totalf.

Time Definition all_tokens_logits_concrete : PArray.concrete_tensor _ _
  := Eval vm_compute in PArray.concretize (logits all_tokens).

Definition all_tokens_logits : tensor _ _ := PArray.abstract all_tokens_logits_concrete.

Time Definition computed_accuracy : float
  := Eval vm_compute in acc_fn (return_per_token := false) all_tokens_logits all_tokens.

Theorem good_accuracy : abs (computed_accuracy - expected_accuracy) <=? 0.5 / totalf.
Proof.
Qed.
