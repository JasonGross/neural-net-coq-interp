From Coq Require Vector.
From Coq Require Import Derive.
From Coq.Structures Require Import Equalities.
From Coq Require Import Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Import PrimitiveProd.
From NeuralNetInterp.Util Require Export Default Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Config HookedTransformer.Module.
Import Instances.Truncating.
#[local] Open Scope core_scope.

Module ModelComputations (cfg : Config) (Import Model : ModelSig cfg).
  Import (hints) cfg.

  Notation residual_error_all_tokens
    := (@Unembed.forward 1 [of_Z (Z.of_N (@pow N N N N_has_pow cfg.d_vocab cfg.n_ctx))] (of_Z (Z.of_N cfg.n_ctx)) float (@coer_refl float) float_has_add float_has_mul (@HookedTransformer.resid_postembed 1 [of_Z (Z.of_N (@pow N N N N_has_pow cfg.d_vocab cfg.n_ctx))] (of_Z (Z.of_N cfg.n_ctx)) float (@coer_refl float) coer_Z_float float_has_add true (@all_tokens true))).

  Definition residual_error_all_tokens_concrete : PArray.concrete_tensor _ float
    := PArray.concretize residual_error_all_tokens.
End ModelComputations.
