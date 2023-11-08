From NeuralNetInterp.MaxOfTwoNumbers Require Import Model.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.

Set NativeCompute Timing.
(*Set NativeCompute Profiling.*)
(* expected: about 45 minutes in vm, about 19 minutes in native *)
Time #[native_compile=no] Local Definition all_tokens_logits_concrete_value := native_compute logits_all_tokens_concrete_opt.

Time #[native_compile=no] Definition all_tokens_logits_concrete : PArray.concrete_tensor _ _ := (*Eval native_compute in pre *)Eval hnf in extract all_tokens_logits_concrete_value.
(* Drop #[native_compile=no] in all definitions after this line once the attribute is automatically transitively recursive, cf https://github.com/coq/coq/pull/18033#issuecomment-1746899653 *)
Time #[native_compile=no] Definition all_tokens_logits_concrete_eq : all_tokens_logits_concrete = logits_all_tokens_concrete_opt := extract_eq all_tokens_logits_concrete_value. (*
Proof. native_cast_no_check (eq_refl logits_all_tokens_concrete). Time Qed.*)

#[native_compile=no] Definition all_tokens_logits : tensor _ _ := PArray.reabstract (fun _ => logits_all_tokens) all_tokens_logits_concrete.
#[native_compile=no] Lemma all_tokens_logits_eq idxs : all_tokens_logits idxs = logits_all_tokens idxs.
Proof.
  cbv [all_tokens_logits].
  rewrite PArray.reabstract_ext_correct; [ reflexivity | ].
  rewrite all_tokens_logits_concrete_eq, logits_all_tokens_concrete_opt_eq.
  reflexivity.
Qed.
