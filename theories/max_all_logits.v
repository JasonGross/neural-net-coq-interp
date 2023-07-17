From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp Require Import max_parameters max max_heuristics.
From NeuralNetInterp.Util Require Import Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Set NativeCompute Timing.
(*Set NativeCompute Profiling.*)
(* expected: about 45 minutes in vm, about 19 minutes in native *)
Time Definition all_tokens_logits_concrete : PArray.concrete_tensor _ _
  := Eval native_compute in PArray.concretize (logits all_tokens(*.[:82,:]*)).
(*
native_compute: Conversion to native code done in 0.00011
native_compute: Compilation done in 0.36763
native_compute: Evaluation done in 540.73917
native_compute: Reification done in 0.29805
all_tokens_logits_concrete is defined

Finished transaction in 542.823 secs (541.953u,0.231s) (successful)
*)
(*
Time Definition all_tokens_logits_concrete_vm : PArray.concrete_tensor _ _
  := Eval vm_compute in PArray.concretize (logits all_tokens).
*)
Definition all_tokens_logits : tensor _ _ := PArray.abstract all_tokens_logits_concrete.
