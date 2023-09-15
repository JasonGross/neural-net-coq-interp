From Coq Require Import Morphisms RelationClasses RelationPairs Relation_Definitions.
From Coq Require Import ZArith.
From Coq.Structures Require Import Equalities.
From Coq Require Import Floats Uint63 ZArith NArith.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util Require Import Default Arith.Classes Arith.Instances Arith.Flocq Arith.Flocq.Instances Arith.Flocq.Definitions.
From NeuralNetInterp.Util.Tactics Require Import Head BreakMatch DestructHead.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel Option Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Util.List.Instances Require Import Forall2 Forall2.Map.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
From NeuralNetInterp.TransformerLens.HookedTransformer Require Import Config Instances Module.
From NeuralNetInterp.TransformerLens.HookedTransformer.Module Require Import Instances.
From NeuralNetInterp.TrainingComputations Require Import interp_max_utils interp_max_utils.Instances.
Import Dependent.ProperNotations Dependent.RelationPairsNotations.
Import Arith.Instances.Truncating Arith.Flocq.Instances.Truncating.
#[local] Open Scope core_scope.

Module ModelComputationsFlocqify (cfg : Config) (Model : ModelSig cfg) (ModelInstances : ModelInstancesSig cfg Model) (Import ModelComputations : ModelComputationsSig cfg Model) (ModelComputationsInstances : ModelComputationsInstancesSig cfg Model ModelInstances ModelComputations).
  Export ModelInstances ModelComputationsInstances.

  Notation R x y := (Prim2B x = y).
  Notation Rf := (fun x y => R x y).

  Local Ltac t :=
    try assumption;
    try exact I;
    repeat intro; subst; try reflexivity;
    try (apply B2Prim_inj; reflexivity);
    lazymatch goal with
    | [ |- R ?x ?y ] => let x := head x in let y := head y in cbv [x y]
    | [ |- ?x = ?y :> bool ] => let x := head x in let y := head y in cbv [x y]
    | [ |- ?G ] => fail 0 "unrecognized" G
    end;
    cbv [Classes.leb Classes.ltb binary_float_has_leb];
    repeat autorewrite with prim2b;
    try reflexivity;
    break_innermost_match;
    repeat autorewrite with prim2b;
    try reflexivity.


  Lemma logit_delta_equiv
    {use_checkpoint1 use_checkpoint2}
    {tokens1 tokens2 : tensor _ IndexType}
    {logits1 logits2 : tensor _ _}
    : Tensor.eqf tokens1 tokens2
      -> Tensor.eqfR Rf logits1 logits2
      -> R (logit_delta (use_checkpoint:=use_checkpoint1) tokens1 logits1)
           (logit_delta (use_checkpoint:=use_checkpoint2) tokens2 logits2).
  Proof. intros; apply (logit_delta_Proper_dep _ _ Rf); t. Qed.
End ModelComputationsFlocqify.

Module ModelComputationsFlocqifySig (cfg : Config) (Model : ModelSig cfg) (ModelInstances : ModelInstancesSig cfg Model) (ModelComputations : ModelComputationsSig cfg Model) (ModelComputationsInstances : ModelComputationsInstancesSig cfg Model ModelInstances ModelComputations) := Nop <+ ModelComputationsFlocqify cfg Model ModelInstances ModelComputations ModelComputationsInstances.
