From Coq Require Import Floats.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Default Pointed ValueExtraction Arith.Classes Arith.Instances.
Import Instances.Truncating.

Local Notation prea := (attn_pattern all_tokens : tensor _ float) (only parsing).
Local Notation prev := (PArray.concretize prea) (only parsing).
Local Definition pre := prev.

Set NativeCompute Timing.
Time Local Definition all_tokens_attn_pattern_concrete_value := native_compute pre.

Time Definition all_tokens_attn_pattern_concrete : PArray.concrete_tensor _ _ := (*Eval native_compute in pre*) Eval hnf in extract all_tokens_attn_pattern_concrete_value.
Definition all_tokens_attn_pattern_concrete_eq : all_tokens_attn_pattern_concrete = prev := extract_eq all_tokens_attn_pattern_concrete_value.
(*Proof. native_cast_no_check (eq_refl prev). Time Qed.*)

Definition all_tokens_attn_pattern : tensor _ _ := PArray.reabstract (fun _ => prea) all_tokens_attn_pattern_concrete.
Lemma all_tokens_attn_pattern_eq idxs : all_tokens_attn_pattern idxs = prea idxs.
Proof.
  cbv [all_tokens_attn_pattern].
  erewrite PArray.reabstract_ext_correct by exact all_tokens_attn_pattern_concrete_eq.
  reflexivity.
Qed.
