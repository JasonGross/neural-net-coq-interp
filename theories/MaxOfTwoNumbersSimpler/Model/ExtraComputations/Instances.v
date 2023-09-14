From Coq Require Import Morphisms RelationClasses RelationPairs.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel Option.
From NeuralNetInterp.Util.Tactics Require Import Head DestructHead.
From NeuralNetInterp.Util.List.Instances Require Import Forall2 Forall2.Map.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances HookedTransformer.Module.Instances.
From NeuralNetInterp.TrainingComputations.interp_max_utils Require Import Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Model Require Import ExtraComputations Instances.
Import Dependent.ProperNotations Dependent.RelationPairsNotations.

Module Model.
  Export ExtraComputations.Model.
  Export Instances.Model.
  Include ModelComputationsInstances cfg Model.Model Instances.Model ExtraComputations.Model.
End Model.
Export (hints) Model.
