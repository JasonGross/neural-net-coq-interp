From Coq Require Import Derive.
From NeuralNetInterp.Util Require Import ValueExtraction.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersUndertrainedSimpler Require Import Model Model.Instances.
From NeuralNetInterp.MaxOfTwoNumbersUndertrainedSimpler.Computed Require Import AllLogits.

(* Drop #[native_compile=no] in all definitions after this line once the attribute is automatically transitively recursive, cf https://github.com/coq/coq/pull/18033#issuecomment-1746899653 *)
#[native_compile=no] Derive precomputed_loss SuchThat (precomputed_loss = true_loss) As precomputed_loss_eq.
Proof.
  cbv [true_loss].
  etransitivity; revgoals.
  { apply Tensor.item_Proper; intro.
    eapply Model.loss_fn_Proper; repeat intro; subst; [ | reflexivity .. ].
    apply all_tokens_logits_eq. }
  subst precomputed_loss; reflexivity.
Qed.

Time #[native_compile=no] Definition computed_loss_value := vm_compute precomputed_loss.
Definition computed_loss := Eval hnf in extract computed_loss_value.
#[native_compile=no] Definition computed_loss_eq : computed_loss = true_loss
  := eq_trans (extract_eq computed_loss_value) precomputed_loss_eq.
