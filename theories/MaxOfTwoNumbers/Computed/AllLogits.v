From NeuralNetInterp.MaxOfTwoNumbers Require Import Model.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.

Set NativeCompute Timing.
(*Set NativeCompute Profiling.*)
(* expected: about 45 minutes in vm, about 19 minutes in native *)
Time Local Definition all_tokens_logits_concrete_value := native_compute logits_all_tokens_concrete.

Time Definition all_tokens_logits_concrete : PArray.concrete_tensor _ _ := (*Eval native_compute in pre *)Eval hnf in extract all_tokens_logits_concrete_value.
Time Definition all_tokens_logits_concrete_eq : all_tokens_logits_concrete = logits_all_tokens_concrete := extract_eq all_tokens_logits_concrete_value. (*
Proof. native_cast_no_check (eq_refl logits_all_tokens_concrete). Time Qed.*)

Definition all_tokens_logits : tensor _ _ := PArray.reabstract (fun _ => logits_all_tokens) all_tokens_logits_concrete.
Lemma all_tokens_logits_eq idxs : all_tokens_logits idxs = logits_all_tokens idxs.
Proof.
  cbv [all_tokens_logits].
  erewrite PArray.reabstract_ext_correct by exact all_tokens_logits_concrete_eq.
  reflexivity.
Qed.
