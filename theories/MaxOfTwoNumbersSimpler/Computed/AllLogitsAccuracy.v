From Coq Require Import Derive.
From NeuralNetInterp.Util Require Import ValueExtraction.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model Model.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import AllLogits.

Derive precomputed_accuracy SuchThat (precomputed_accuracy = true_accuracy) As precomputed_accuracy_eq.
Proof.
  cbv [true_accuracy].
  etransitivity; revgoals.
  { apply Tensor.item_Proper; intro.
    eapply Model.acc_fn_Proper; repeat intro; subst; [ | reflexivity .. ].
    apply all_tokens_logits_eq. }
  subst precomputed_accuracy; reflexivity.
Qed.

Time Definition computed_accuracy_value := vm_compute precomputed_accuracy.
Definition computed_accuracy := Eval hnf in extract computed_accuracy_value.
Definition computed_accuracy_eq : computed_accuracy = true_accuracy
  := eq_trans (extract_eq computed_accuracy_value) precomputed_accuracy_eq.
