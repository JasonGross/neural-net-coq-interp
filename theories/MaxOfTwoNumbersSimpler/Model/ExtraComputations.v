From NeuralNetInterp.TransformerLens Require Import HookedTransformer.Module.ExtraComputations.
From NeuralNetInterp.TrainingComputations Require Import interp_max_utils.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model.

Module Export Model := ExtraComputations.ModelComputations cfg Model <+ interp_max_utils.ModelComputations cfg Model.
