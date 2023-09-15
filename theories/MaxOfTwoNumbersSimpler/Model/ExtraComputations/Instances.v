From NeuralNetInterp.TransformerLens.HookedTransformer Require Import Module.Instances.
From NeuralNetInterp.TrainingComputations.interp_max_utils Require Import Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Model Require Import ExtraComputations Instances.

Module Model.
  Export ExtraComputations.Model.
  Export Instances.Model.
  Include ModelComputationsInstances cfg Model.Model Instances.Model ExtraComputations.Model.
End Model.
Export (hints) Model.
