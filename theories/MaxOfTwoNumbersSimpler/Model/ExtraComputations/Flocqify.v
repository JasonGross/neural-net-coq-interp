From NeuralNetInterp.TrainingComputations.interp_max_utils Require Import Flocqify.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Model.
From NeuralNetInterp.MaxOfTwoNumbersSimpler.Model Require Import ExtraComputations Instances ExtraComputations.Instances.

Module Model.
  Export ExtraComputations.Instances.Model.
  Include ModelComputationsFlocqify cfg Model.Model Model.Instances.Model ExtraComputations.Model ExtraComputations.Instances.Model.
End Model.
Export (hints) Model.
