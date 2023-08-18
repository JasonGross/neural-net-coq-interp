From Coq Require Import Reals ZArith.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util Require Import Default Arith.Classes Arith.Instances Arith.Flocq Arith.Flocq.Instances Arith.Flocq.Definitions.
From NeuralNetInterp.Util.Tactics Require Import Head.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model Model.Instances Model.Flocqify Heuristics.
Import Dependent.ProperNotations.
Import Arith.Instances.Truncating Arith.Flocq.Instances.Truncating.
#[local] Open Scope core_scope.

Module Model.
  Export Model.Instances.Model.

  Lemma acc_fn_equiv_bounded_no_checkpoint
    (use_checkpoint1:=false)
    (use_checkpoint2:=false)
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (logits1 := logits (use_checkpoint:=use_checkpoint1) tokens1)
    (logits2 := logits (use_checkpoint:=use_checkpoint2) tokens2)
    (acc1 := acc_fn (use_checkpoint:=use_checkpoint1) logits1 tokens1)
    (acc2 := acc_fn (use_checkpoint:=use_checkpoint2) logits2 tokens2)
    : Tensor.eqfR
        (fun (x:binary_float prec emax) (y:R) => (abs ((x:R) - y) <=? (total_rounding_error:R)) = true)
        acc1 acc2.
  Proof.
    intro i.
    repeat match goal with H := _ |- _ => subst H end.
    cbv [Classes.abs Classes.sub Classes.leb R_has_abs R_has_sub R_has_leb].
  Admitted. (* XXX FIXME *)

  Lemma acc_fn_equiv_bounded
    {use_checkpoint1 use_checkpoint2}
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (logits1 := logits (use_checkpoint:=use_checkpoint1) tokens1)
    (logits2 := logits (use_checkpoint:=use_checkpoint2) tokens2)
    (acc1 := acc_fn (use_checkpoint:=use_checkpoint1) logits1 tokens1)
    (acc2 := acc_fn (use_checkpoint:=use_checkpoint2) logits2 tokens2)
    : Tensor.eqfR
        (fun (x:binary_float prec emax) (y:R) => (abs ((x:R) - y) <=? (total_rounding_error:R)) = true)
        acc1 acc2.
  Proof.
    intro i.
    rewrite <- (acc_fn_equiv_bounded_no_checkpoint i).
    do 2 f_equal.
    apply f_equal2.
    all: subst acc1 acc2 logits1 logits2 tokens1 tokens2.
    all: try apply f_equal.
    all: eapply acc_fn_Proper_dep.
    all: try solve [ repeat intro; subst; constructor ].
    all: try now apply all_tokens_Proper.
    all: eapply logits_Proper_dep.
    all: try solve [ repeat intro; subst; constructor ].
    all: try now apply all_tokens_Proper.
  Qed.
End Model.
