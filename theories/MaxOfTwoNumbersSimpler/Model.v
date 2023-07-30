From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool Option.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
From NeuralNetInterp.MaxOfTwoNumbers Require Import Parameters.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Local Open Scope raw_tensor_scope.

(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

(** Coq infra *)
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.

(** Hyperparameters *)
Definition N_LAYERS : nat := 1.
Definition N_HEADS : nat := 1.
Definition D_MODEL : nat := 32.
Definition D_HEAD : nat := 32.
(*Definition D_MLP = None*)

Definition D_VOCAB : nat := 64.

Notation tensor_of_list ls := (Tensor.PArray.abstract (Tensor.PArray.concretize (Tensor.of_list ls))) (only parsing).
Definition W_E : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.W_E.
Definition W_pos : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.W_pos.
Definition L0_attn_W_Q : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_Q.
Definition L0_attn_W_K : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_K.
Definition L0_attn_W_V : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_V.
Definition L0_attn_W_O : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_O.
Definition L0_attn_b_Q : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_Q.
Definition L0_attn_b_K : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_K.
Definition L0_attn_b_V : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_V.
Definition L0_attn_b_O : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_O.
Definition L0_ln1_b : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_ln1_b.
Definition L0_ln1_w : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_ln1_w.
Definition ln_final_b : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.ln_final_b.
Definition ln_final_w : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.ln_final_w.
Definition W_U : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.W_U.
Definition b_U : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.b_U.

#[local] Notation FLOAT := float (only parsing). (* or Q *)

Section with_batch.
  Context {r} {batch : Shape r} {pos}
    (s := (batch ::' pos)%shape)
    (resid_shape := (s ::' cfg.d_model)%shape).

  Definition embed (tokens : tensor s IndexType) : tensor resid_shape FLOAT
    := HookedTransformer.embed W_E tokens.

  Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape FLOAT
    := HookedTransformer.pos_embed (n_ctx:=cfg.n_ctx) W_pos tokens.

  Definition ln_final (resid : tensor resid_shape FLOAT) : tensor resid_shape FLOAT
    := HookedTransformer.ln_final cfg.eps ln_final_w ln_final_b resid.

  Definition unembed (resid : tensor resid_shape FLOAT) : tensor (s ::' cfg.d_vocab_out) FLOAT
    := HookedTransformer.unembed W_U b_U resid.

  Definition blocks_params : list _
    := [(L0_attn_W_Q, L0_attn_W_K, L0_attn_W_V, L0_attn_W_O,
          L0_attn_b_Q, L0_attn_b_K, L0_attn_b_V,
          L0_attn_b_O,
          L0_ln1_w, L0_ln1_b)].

  Definition logits (tokens : tensor s IndexType) : tensor (s ::' cfg.d_vocab_out) FLOAT
    := HookedTransformer.logits
         (n_ctx:=cfg.n_ctx)
         cfg.eps
         W_E
         W_pos

         blocks_params

         ln_final_w ln_final_b

         W_U b_U

         tokens.

  (** convenience *)
  Definition masked_attn_scores (tokens : tensor s IndexType)
    : tensor (batch ::' cfg.n_heads ::' pos ::' pos) FLOAT
    := Option.invert_Some
         (HookedTransformer.HookedTransformer.masked_attn_scores
            (n_ctx:=cfg.n_ctx)
            cfg.eps
            W_E
            W_pos

            blocks_params

            0

            tokens).

  Definition attn_pattern (tokens : tensor s IndexType)
    : tensor (batch ::' cfg.n_heads ::' pos ::' pos) FLOAT
    := Option.invert_Some
         (HookedTransformer.HookedTransformer.attn_pattern
            (n_ctx:=cfg.n_ctx)
            cfg.eps
            W_E
            W_pos

            blocks_params

            0

            tokens).
End with_batch.

Notation model := logits (only parsing).

Definition loss_fn {r} {batch : Shape r} {return_per_token : with_default "return_per_token" bool false}
  (logits : tensor (batch ::' cfg.n_ctx ::' cfg.d_vocab_out) FLOAT)
  (tokens : tensor (batch ::' cfg.n_ctx) IndexType)
  : tensor (if return_per_token return Shape (if return_per_token then _ else _) then Shape.squeeze batch else []) FLOAT
  := (let logits : tensor (batch ::' _) FLOAT
        := PArray.checkpoint (logits.[…, -1, :]) in
      let true_maximum : tensor (batch ::' 1) IndexType
        := reduce_axis_m1 (keepdim:=true) Reduction.max tokens in
      let log_probs
        := log_softmax_dim_m1 logits in
      let correct_log_probs
        := PArray.checkpoint (gather_dim_m1 log_probs true_maximum) in
      if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) FLOAT)
      then -Tensor.squeeze correct_log_probs
      else -Tensor.mean correct_log_probs)%core.

Definition acc_fn {r} {batch : Shape r} {return_per_token : with_default "return_per_token" bool false}
  (logits : tensor (batch ::' cfg.n_ctx ::' cfg.d_vocab_out) FLOAT)
  (tokens : tensor (batch ::' cfg.n_ctx) IndexType)
  : tensor (if return_per_token return Shape (if return_per_token then _ else _) then batch else []) FLOAT
  := (let pred_logits : tensor (batch ::' _) FLOAT
        := PArray.checkpoint (logits.[…, -1, :]) in
      let pred_tokens : tensor batch IndexType
        := reduce_axis_m1 (keepdim:=false) Reduction.argmax pred_logits in
      let true_maximum : tensor batch IndexType
        := reduce_axis_m1 (keepdim:=false) Reduction.max tokens in
      let res : tensor _ FLOAT
        := PArray.checkpoint (Tensor.of_bool (Tensor.map2 eqb pred_tokens true_maximum)) in
      if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) FLOAT)
      then res
      else Tensor.mean res)%core.

Definition all_tokens : tensor [(cfg.d_vocab ^ cfg.n_ctx)%core : N; 2] RawIndexType
  := let all_toks := Tensor.arange (start:=0) (Uint63.of_Z cfg.d_vocab) in
     PArray.checkpoint (Tensor.cartesian_prod all_toks all_toks).

Definition logits_all_tokens : tensor _ _
  := logits all_tokens.
