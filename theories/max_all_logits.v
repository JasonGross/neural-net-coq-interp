From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp Require Import max_parameters max max_heuristics.
From NeuralNetInterp.Util Require Import Pointed ValueExtraction.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Local Notation prea := (logits all_tokens) (only parsing).
Local Notation prev := (PArray.concretize prea) (only parsing).
Local Definition pre := prev.

Set NativeCompute Timing.
(*Set NativeCompute Profiling.*)
(* expected: about 45 minutes in vm, about 19 minutes in native *)
Time Local Definition all_tokens_logits_concrete_value := native_compute pre.

Time Definition all_tokens_logits_concrete : PArray.concrete_tensor _ _ := Eval hnf in extract all_tokens_logits_concrete_value.
Definition all_tokens_logits_concrete_eq : all_tokens_logits_concrete = prev := extract_eq all_tokens_logits_concrete_value.

Definition all_tokens_logits : tensor _ _ := PArray.reabstract (fun _ => prea) all_tokens_logits_concrete.
Lemma all_tokens_logits_eq idxs : all_tokens_logits idxs = prea idxs.
Proof.
  cbv [all_tokens_logits].
  erewrite PArray.reabstract_ext_correct by exact all_tokens_logits_concrete_eq.
  reflexivity.
Qed.
