From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.Definitions.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersUndertrainedSimpler Require Import Parameters Model Heuristics TheoremStatement Model.Instances.
From NeuralNetInterp.MaxOfTwoNumbersUndertrainedSimpler.Computed Require Import AllLogits AllLogitsAccuracy AllLogitsLoss.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Compute abs (computed_accuracy - expected_accuracy). (*     = 0.006103515625%float *)
Compute (abs (computed_accuracy - expected_accuracy) * totalf)%float. (*     = 50%float *) (* probably from floating point assoc issues, etc *)
Compute Qred (PrimFloat.to_Q (computed_accuracy * totalf)) / Qred (PrimFloat.to_Q totalf). (*     = 8092 # 8192 *)
Compute abs (computed_accuracy / expected_accuracy). (*     = 0.99385900270203886%float *)
Compute (abs (1 - computed_accuracy / expected_accuracy) * totalf)%float. (*     = 50.307049864897635%float *) (* probably from floating point assoc issues, etc *)

Compute (computed_loss, expected_loss). (*     = (0.12643549334277809%float, 0.12643511593341827%float) *)
Compute abs (computed_loss - expected_loss). (*     = 3.7740935981966928e-07%float *)
Compute Qred (PrimFloat.to_Q (computed_loss * totalf)) / Qred (PrimFloat.to_Q totalf). (*      =      = 2277659362819761 # 18014398509481984 *)
Compute abs (computed_loss / expected_loss). (*     = 1.0000029850042611%float *)
Compute abs (1%float - computed_loss / expected_loss). (*     = 2.9850042611023753e-06%float *)

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
