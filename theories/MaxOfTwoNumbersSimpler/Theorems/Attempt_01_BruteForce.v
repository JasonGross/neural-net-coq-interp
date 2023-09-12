From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.Definitions.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model Heuristics TheoremStatement Model.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import AllLogits AllLogitsAccuracy AllLogitsLoss.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Compute abs (computed_accuracy - expected_accuracy). (*     = 0%float *)
Compute (abs (computed_accuracy - expected_accuracy) * totalf)%float. (* = 0 *) (* probably from floating point assoc issues, etc *)
Compute Qred (PrimFloat.to_Q (computed_accuracy * totalf)) / Qred (PrimFloat.to_Q totalf). (*      = 8192 # 8192 *)
Compute abs (computed_accuracy / expected_accuracy). (*      = 1%float *)
Compute (abs (1 - computed_accuracy / expected_accuracy) * totalf)%float. (* = 0%float *) (* probably from floating point assoc issues, etc *)

Compute (computed_loss, expected_loss). (*     = (5.4667811687832829e-07%float, 1.7639347049680509e-07%float) *)
Compute abs (computed_loss - expected_loss). (*     = 3.702846463815232e-07%float *)
Compute Qred (PrimFloat.to_Q (computed_loss * totalf)) / Qred (PrimFloat.to_Q totalf). (*      = 0x0.0000092bf6f2377%xQ *)
Compute abs (computed_loss / expected_loss). (*     = 3.0991970130109205%float *)
Compute abs (3%float - computed_loss / expected_loss). (*     = 0.099197013010920543%float *)

Theorem good_accuracy : TheoremStatement.Accuracy.best (* (abs (real_accuracy / expected_accuracy - 1) <=? error)%float = true *).
Proof.
  rewrite <- computed_accuracy_eq.
  vm_compute; reflexivity.
Qed.

Theorem good_loss : TheoremStatement.Loss.best (* (abs (real_loss / expected_loss - 3) <=? error)%float = true *).
Proof.
  rewrite <- computed_loss_eq.
  vm_compute; reflexivity.
Qed.
