From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model Model.ExtraComputations.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.

Set NativeCompute Timing.
(*Set NativeCompute Profiling.*)
(* expected: about 45 minutes in vm, about 19 minutes in native *)
Time Local Definition all_tokens_residual_error_concrete_value := native_compute residual_error_all_tokens_concrete_opt.

Time Definition all_tokens_residual_error_concrete : PArray.concrete_tensor _ _ := (*Eval native_compute in pre *)Eval hnf in extract all_tokens_residual_error_concrete_value.
Time Definition all_tokens_residual_error_concrete_eq : all_tokens_residual_error_concrete = residual_error_all_tokens_concrete_opt := extract_eq all_tokens_residual_error_concrete_value. (*
Proof. native_cast_no_check (eq_refl residual_error_all_tokens_concrete). Time Qed.*)

Definition all_tokens_residual_error : tensor _ _ := PArray.reabstract (fun _ => residual_error_all_tokens) all_tokens_residual_error_concrete.
Lemma all_tokens_residual_error_eq idxs : all_tokens_residual_error idxs = residual_error_all_tokens idxs.
Proof.
  cbv [all_tokens_residual_error].
  rewrite PArray.reabstract_ext_correct; [ reflexivity | ].
  rewrite all_tokens_residual_error_concrete_eq, residual_error_all_tokens_concrete_opt_eq.
  reflexivity.
Qed.