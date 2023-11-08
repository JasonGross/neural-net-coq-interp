From Coq Require Import Derive.
From NeuralNetInterp.Util Require Import ValueExtraction.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model Model.ExtraComputations.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Computed Require Import AllLogits.

(* Drop #[native_compile=no] in all definitions after this line once the attribute is automatically transitively recursive, cf https://github.com/coq/coq/pull/18033#issuecomment-1746899653 *)
Time #[local,native_compile=no] Definition pre_logit_delta := logit_delta_float_gen (all_tokens (use_checkpoint:=false)) all_tokens_logits.
Time #[local,native_compile=no] Definition logit_delta_value := vm_compute pre_logit_delta.

Time Definition logit_delta := (*Eval native_compute in pre *)Eval hnf in extract logit_delta_value.
#[local] Strategy -100 [pre_logit_delta].
Time #[native_compile=no] Definition logit_delta_eq : logit_delta = ltac:(let v := eval cbv delta [pre_logit_delta] in pre_logit_delta in exact v) := extract_eq logit_delta_value.
