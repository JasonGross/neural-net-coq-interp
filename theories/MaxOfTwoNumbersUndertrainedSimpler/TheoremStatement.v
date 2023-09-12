From Coq Require Import Floats Sint63 Uint63 QArith List PArray.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool.
(*From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.*)
From NeuralNetInterp.MaxOfTwoNumbersUndertrainedSimpler Require Import Model Parameters Heuristics.
Import Arith.Instances.Truncating.
Local Open Scope raw_tensor_scope.

Definition total : int := Eval vm_compute in Shape.item (Shape.reshape (shape_of all_tokens)).
Definition expected_correct : int := Eval vm_compute in total - Uint63.of_Z 50 (* (List.length Heuristics.incorrect_results)*).
Definition totalf : float := Eval vm_compute in PrimFloat.of_uint63 total.
Definition expected_correctf : float := Eval vm_compute in PrimFloat.of_uint63 expected_correct.
Definition expected_accuracy : float := Eval vm_compute in expected_correctf / totalf.
Definition expected_loss : float := 0x1.02f06a0000000p-3.

Module Accuracy.
  Definition error : float := Eval cbv in (51 / totalf)%float.
  Local Notation abs := (@abs float float_has_abs) (only parsing).
  Notation best := ((abs (true_accuracy / expected_accuracy - 1) <=? error)%float = true)
                     (only parsing).
End Accuracy.

Module Loss.
  Definition error : float := 0x2p-19%float.
  Local Notation abs := (@abs float float_has_abs) (only parsing).
  Notation best := ((abs (true_loss / expected_loss - 1) <=? error)%float = true)
                     (only parsing).
End Loss.
