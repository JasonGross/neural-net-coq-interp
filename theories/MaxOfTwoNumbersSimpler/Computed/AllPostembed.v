From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model Model.ExtraComputations.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.

Set NativeCompute Timing.
(*Set NativeCompute Profiling.*)
(* expected: about 45 minutes in vm, about 19 minutes in native *)
Time #[native_compile=no] Local Definition centered_residual_error_m1_concrete_value := native_compute centered_residual_error_m1_concrete_opt.

Time #[native_compile=no] Definition centered_residual_error_m1_concrete : PArray.concrete_tensor _ _ := (*Eval native_compute in pre *)Eval hnf in extract centered_residual_error_m1_concrete_value.
Time Definition centered_residual_error_m1_concrete_eq : centered_residual_error_m1_concrete = centered_residual_error_m1_concrete_opt := extract_eq centered_residual_error_m1_concrete_value. (*
Proof. native_cast_no_check (eq_refl residual_error_m1_all_tokens_concrete). Time Qed.*)

Definition centered_residual_error_m1 : tensor _ _ := PArray.reabstract (fun _ => centered_residual_error_m1_float) centered_residual_error_m1_concrete.
Lemma centered_residual_error_m1_eq idxs : centered_residual_error_m1 idxs = centered_residual_error_m1_float idxs.
Proof.
  cbv [centered_residual_error_m1].
  rewrite PArray.reabstract_ext_correct; [ reflexivity | ].
  rewrite centered_residual_error_m1_concrete_eq, centered_residual_error_m1_concrete_opt_eq.
  reflexivity.
Qed.
