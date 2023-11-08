From Coq Require Import Derive.
From NeuralNetInterp.Util Require Import ValueExtraction.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model Model.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import AllLogits.

(* Drop #[native_compile=no] in all definitions after this line once the attribute is automatically transitively recursive, cf https://github.com/coq/coq/pull/18033#issuecomment-1746899653 *)
#[native_compile=no] Derive precomputed_accuracy SuchThat (precomputed_accuracy = true_accuracy) As precomputed_accuracy_eq.
Proof.
  cbv [true_accuracy].
  etransitivity; revgoals.
  { apply Tensor.item_Proper; intro.
    eapply Model.acc_fn_Proper; repeat intro; subst; [ | reflexivity .. ].
    apply all_tokens_logits_eq. }
  subst precomputed_accuracy; reflexivity.
Defined.

Time #[native_compile=no] Definition computed_accuracy_value := vm_compute precomputed_accuracy.
Definition computed_accuracy := Eval hnf in extract computed_accuracy_value.
#[native_compile=no] Definition computed_accuracy_eq : computed_accuracy = true_accuracy
  := eq_trans (extract_eq computed_accuracy_value) precomputed_accuracy_eq.
