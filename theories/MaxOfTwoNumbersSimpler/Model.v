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
Import Arith.Instances.Uint63.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Local Open Scope raw_tensor_scope.

(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

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

Definition all_tokens : tensor [(cfg.d_vocab ^ cfg.n_ctx)%core : N; 2] RawIndexType
  := let all_toks := Tensor.arange (start:=0) (Uint63.of_Z cfg.d_vocab) in
     PArray.checkpoint (Tensor.cartesian_prod all_toks all_toks).

Section with_batch.
  Context {r} {batch : Shape r} {pos}
    (s := (batch ::' pos)%shape)
    (resid_shape := (s ::' cfg.d_model)%shape)
    {return_per_token : with_default "return_per_token" bool false}
    {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
    {zeroA : has_zero A} {oneA : has_one A}
    (defaultA : pointed A := @coer _ _ coerZ point)
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
    {ltbA : has_ltb A}
    {oppA : has_opp A} {sqrtA : has_sqrt A} {expA : has_exp A} {lnA : has_ln A}
    {use_checkpoint : with_default "use_checkpoint" bool true}.
  #[local] Existing Instance defaultA.
  #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).

  Let coer_tensor_float {r s} (x : @tensor r s float) : @tensor r s A
      := Tensor.map coer x.
  Let coerA' (x : float) : A := coer x.
  #[local] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
  #[local] Coercion coer_tensor_float : tensor >-> tensor.
  #[local] Set Warnings Append "uniform-inheritance,ambiguous-paths".
  #[local] Coercion coerA' : float >-> A.

  Definition embed (tokens : tensor s IndexType) : tensor resid_shape A
    := HookedTransformer.embed (A:=A) W_E tokens.

  Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape A
    := HookedTransformer.pos_embed (A:=A) (n_ctx:=cfg.n_ctx) W_pos tokens.

  Definition ln_final (resid : tensor resid_shape A) : tensor resid_shape A
    := HookedTransformer.ln_final (normalization_type:=None) (A:=A) cfg.eps I I resid.

  Definition unembed (resid : tensor resid_shape A) : tensor (s ::' cfg.d_vocab_out) A
    := HookedTransformer.unembed (A:=A) W_U b_U resid.

  Definition blocks_params : list _
    := [(L0_attn_W_Q:tensor _ A, L0_attn_W_K:tensor _ A, L0_attn_W_V:tensor _ A, L0_attn_W_O:tensor _ A,
          L0_attn_b_Q:tensor _ A, L0_attn_b_K:tensor _ A, L0_attn_b_V:tensor _ A,
          L0_attn_b_O:tensor _ A,
          I, I)].

  Definition logits (tokens : tensor s IndexType) : tensor (s ::' cfg.d_vocab_out) A
    := HookedTransformer.logits
         (A:=A)
         (n_ctx:=cfg.n_ctx)
         (normalization_type:=None)
         cfg.eps
         W_E
         W_pos

         blocks_params

         I I

         W_U b_U

         tokens.

  (** convenience *)
  Definition masked_attn_scores (tokens : tensor s IndexType)
    : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A
    := Option.invert_Some
         (HookedTransformer.HookedTransformer.masked_attn_scores
            (normalization_type:=None)
            (A:=A)
            (n_ctx:=cfg.n_ctx)
            cfg.eps
            W_E
            W_pos

            blocks_params

            0

            tokens).

  Definition attn_pattern (tokens : tensor s IndexType)
    : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A
    := Option.invert_Some
         (HookedTransformer.HookedTransformer.attn_pattern
            (normalization_type:=None)
            (A:=A)
            (n_ctx:=cfg.n_ctx)
            cfg.eps
            W_E
            W_pos

            blocks_params

            0

            tokens).

  Definition loss_fn
    (logits : tensor (s ::' cfg.d_vocab_out) A)
    (tokens : tensor s IndexType)
  : tensor (if return_per_token return Shape (if return_per_token then _ else _) then Shape.squeeze batch else []) A
  := (let logits : tensor (batch ::' _) A
        := checkpoint (logits.[…, -1, :]) in
      let true_maximum : tensor (batch ::' 1) IndexType
        := reduce_axis_m1 (keepdim:=true) Reduction.max tokens in
      let log_probs
        := log_softmax_dim_m1 logits in
      let correct_log_probs
        := checkpoint (gather_dim_m1 log_probs true_maximum) in
      if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) A)
      then -Tensor.squeeze correct_log_probs
      else -Tensor.mean correct_log_probs)%core.

  Definition acc_fn
    (logits : tensor (s ::' cfg.d_vocab_out) A)
    (tokens : tensor s IndexType)
    : tensor (if return_per_token return Shape (if return_per_token then _ else _) then batch else []) A
    := (let pred_logits : tensor (batch ::' _) A
          := checkpoint (logits.[…, -1, :]) in
        let pred_tokens : tensor batch IndexType
          := reduce_axis_m1 (keepdim:=false) Reduction.argmax pred_logits in
        let true_maximum : tensor batch IndexType
          := reduce_axis_m1 (keepdim:=false) Reduction.max tokens in
        let res : tensor _ A
          := checkpoint (Tensor.of_bool (Tensor.map2 eqb pred_tokens true_maximum)) in
        if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) A)
        then res
        else Tensor.mean res)%core.
End with_batch.

Notation model := logits (only parsing).

Definition logits_all_tokens : tensor _ float
  := logits all_tokens.
