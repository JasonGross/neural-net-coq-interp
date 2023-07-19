From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp Require Import max_parameters max max_heuristics.
From NeuralNetInterp.Util Require Import Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp Require Import max_instances.
From NeuralNetInterp Require Import max_all_logits.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Definition total : int := Eval vm_compute in Shape.item (Shape.reshape (shape_of all_tokens)).
Definition expected_correct : int := Eval vm_compute in total - Uint63.of_Z (List.length heuristics.incorrect_results).
Definition totalf : float := Eval vm_compute in PrimFloat.of_uint63 total.
Definition expected_correctf : float := Eval vm_compute in PrimFloat.of_uint63 expected_correct.
Definition expected_accuracy : float := Eval vm_compute in expected_correctf / totalf.

Definition real_accuracy : float
  := Tensor.item (acc_fn (return_per_token := false) (logits all_tokens) all_tokens).

Set NativeCompute Timing.

Derive computed_accuracy SuchThat (computed_accuracy = real_accuracy) As computed_accuracy_eq.
Proof.
  cbv [real_accuracy].
  etransitivity; revgoals.
  { apply Tensor.item_Proper; intro.
    eapply acc_fn_Proper; repeat intro; subst.
    2-3: reflexivity.
    apply all_tokens_logits_eq. }
  Time vm_compute; reflexivity.
Defined.

Definition error : float := Eval cbv in (20 / totalf)%float.
Compute abs (computed_accuracy - expected_accuracy). (*      = 0.0023193359375%float *)
Compute (abs (computed_accuracy - expected_accuracy) * totalf)%float. (* = 19 *) (* probably from floating point assoc issues, etc *)
Compute Qred (PrimFloat.to_Q (computed_accuracy * totalf)) / Qred (PrimFloat.to_Q totalf). (*      = 8154 # 8192 *)
Theorem good_accuracy : (abs (real_accuracy - expected_accuracy) <? error)%float = true.
Proof.
  rewrite <- computed_accuracy_eq.
  vm_compute; reflexivity.
Qed.
