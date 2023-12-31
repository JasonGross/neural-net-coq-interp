(** Ported from https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/HookedTransformerConfig.py *)
From Coq Require Import Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Import Default Arith.Instances.
From NeuralNetInterp.Util Require Import PrimitiveProd.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.TransformerLens.HookedTransformer.Config Require Export Common.
Import Instances.Truncating.

(*
Module Type ExtraListConfig (Import cfg : CommonConfig).
  Parameter W_E : List.concrete_tensor [d_vocab; d_model] float.
  Parameter W_pos : List.concrete_tensor [n_ctx; d_model] float.
  Parameter W_U : List.concrete_tensor [d_model; d_vocab_out] float.
  Parameter b_U : List.concrete_tensor [d_vocab_out] float.
End ExtraListConfig.
*)
Definition ln_tensor_gen d_model (nt : with_default "normalization_type" (Some LN)) A
  := (match nt with
      | Some LN => tensor [d_model] A
      | Datatypes.None => with_default "()" True I
      end).
Definition block_params_type_gen n_heads d_model d_head (nt : with_default "normalization_type" (Some LN)) A
  := ((* (W_Q W_K W_V W_O : tensor [n_heads; d_model; d_head] A)
                         (b_Q b_K b_V : tensor [n_heads; d_head] A)
                         (b_O : tensor [d_model] A)
                         (ln1_w ln1_b : tensor [d_model] A) *)
    tensor [n_heads; d_model; d_head] A * tensor [n_heads; d_model; d_head] A * tensor [n_heads; d_model; d_head] A * tensor [n_heads; d_model; d_head] A
    * tensor [n_heads; d_head] A * tensor [n_heads; d_head] A * tensor [n_heads; d_head] A
    * tensor [d_model] A
    * ln_tensor_gen d_model nt A
    * ln_tensor_gen d_model nt A)%primproj_type.
Module Type ExtraConfig (Import cfg : CommonConfig).
  Parameter W_E : tensor [d_vocab; d_model] float.
  Parameter W_pos : tensor [n_ctx; d_model] float.
  Parameter W_U : tensor [d_model; d_vocab_out] float.
  Parameter b_U : tensor [d_vocab_out] float.
  Notation ln_tensor A := (ln_tensor_gen d_model normalization_type A).
  Parameter ln_final_w : ln_tensor float.
  Parameter ln_final_b : ln_tensor float.
  Notation block_params_type A := (block_params_type_gen n_heads d_model d_head normalization_type A).
  Parameter blocks_params : list (block_params_type float).
  Notation n_layers := (List.length blocks_params).
End ExtraConfig.

(*Module Type ListConfig := CommonConfig <+ ExtraListConfig.*)
Module Type Config := CommonConfig <+ ExtraConfig.
