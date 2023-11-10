From Coq Require Import Derive.
From NeuralNetInterp.Util Require Import ValueExtraction.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model Model.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import AllLogits.

Derive precomputed_loss SuchThat (precomputed_loss = true_loss) As precomputed_loss_eq.
Proof.
  cbv [true_loss].
  etransitivity; revgoals.
  { apply Tensor.item_Proper; intro.
    eapply Model.loss_fn_Proper; repeat intro; subst; [ | reflexivity .. ].
    apply all_tokens_logits_eq. }
  subst precomputed_loss; reflexivity.
Qed.

Time Definition computed_loss_value := vm_compute precomputed_loss.
Definition computed_loss := Eval hnf in extract computed_loss_value.
Definition computed_loss_eq : computed_loss = true_loss
  := eq_trans (extract_eq computed_loss_value) precomputed_loss_eq.
