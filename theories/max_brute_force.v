From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp Require Import max_parameters max max_heuristics.
From NeuralNetInterp.Util Require Import Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.
From NeuralNetInterp Require Import max_all_logits.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Definition total : int := Eval vm_compute in Shape.item (Shape.reshape (shape_of all_tokens)).
Definition expected_correct : int := Eval vm_compute in total - Uint63.of_Z (List.length heuristics.incorrect_results).
Definition totalf : float := Eval vm_compute in PrimFloat.of_uint63 total.
Definition expected_correctf : float := Eval vm_compute in PrimFloat.of_uint63 expected_correct.
Definition expected_accuracy : float := Eval vm_compute in expected_correctf / totalf.

Time Definition computed_accuracy : float
  := Eval vm_compute in Tensor.item (acc_fn (return_per_token := false) all_tokens_logits all_tokens).

Definition error : float := Eval cbv in (20 / totalf)%float.
Compute abs (computed_accuracy - expected_accuracy). (*      = 0.0023193359375%float *)
Compute (abs (computed_accuracy - expected_accuracy) * totalf)%float. (* = 19 *) (* probably from floating point assoc issues, etc *)
Compute Qred (PrimFloat.to_Q (computed_accuracy * totalf)) / Qred (PrimFloat.to_Q totalf). (*      = 8154 # 8192 *)
Theorem good_accuracy : (abs (computed_accuracy - expected_accuracy) <? (20 / totalf))%float = true.
Proof. vm_compute; reflexivity. Qed.
