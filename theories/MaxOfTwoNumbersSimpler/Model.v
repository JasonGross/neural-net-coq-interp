From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool Option PrimitiveProd.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Config HookedTransformer.Module.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters.
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

Module Import cfg <: Config.
  Include Parameters.cfg.

  Definition W_E : tensor _ _ := Eval cbv in tensor_of_list Parameters.W_E.
  Definition W_pos : tensor _ _ := Eval cbv in tensor_of_list Parameters.W_pos.
  Definition L0_attn_W_Q : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_W_Q.
  Definition L0_attn_W_K : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_W_K.
  Definition L0_attn_W_V : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_W_V.
  Definition L0_attn_W_O : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_W_O.
  Definition L0_attn_b_Q : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_b_Q.
  Definition L0_attn_b_K : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_b_K.
  Definition L0_attn_b_V : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_b_V.
  Definition L0_attn_b_O : tensor _ _ := Eval cbv in tensor_of_list Parameters.L0_attn_b_O.
  Definition ln_final_b := I.
  Definition ln_final_w := I.
  Definition W_U : tensor _ _ := Eval cbv in tensor_of_list Parameters.W_U.
  Definition b_U : tensor _ _ := Eval cbv in tensor_of_list Parameters.b_U.

  Definition blocks_params
    := [(L0_attn_W_Q, L0_attn_W_K, L0_attn_W_V, L0_attn_W_O
          , L0_attn_b_Q, L0_attn_b_K, L0_attn_b_V, L0_attn_b_O
          , I, I)%primproj].
End cfg.

Module Export Model.
  Include Model cfg.
  Export HookedTransformer.

  Section with_batch.
    Context {r} {batch : Shape r} {pos}
      (s := (batch ::' pos)%shape)
      (resid_shape := (s ::' cfg.d_model)%shape)
      {return_per_token : with_default "return_per_token" bool false}
      {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
      (defaultA : pointed A := @coer _ _ coerZ point)
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {maxA : has_max A}
      {ltbA : has_ltb A}
      {oppA : has_opp A} {sqrtA : has_sqrt A} {expA : has_exp A} {lnA : has_ln A}
      {use_checkpoint : with_default "use_checkpoint" bool true}.
    #[local] Existing Instance defaultA.

    Let coer_tensor_float {r s} (x : @tensor r s float) : @tensor r s A
        := Tensor.map coer x.
    Let coerA' (x : float) : A := coer x.
    #[local] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
    #[local] Coercion coer_tensor_float : tensor >-> tensor.
    #[local] Set Warnings Append "uniform-inheritance,ambiguous-paths".
    #[local] Coercion coerA' : float >-> A.

    Definition masked_attn_scores (tokens : tensor s IndexType)
        : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A
      := Option.invert_Some
           (HookedTransformer.masked_attn_scores (A:=A) 0 tokens).
    Definition attn_pattern (tokens : tensor s IndexType)
        : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A
      := Option.invert_Some
           (HookedTransformer.attn_pattern (A:=A) 0 tokens).

    Definition loss_fn
      (logits : tensor (s ::' cfg.d_vocab_out) A)
      (tokens : tensor s IndexType)
      : tensor (if return_per_token return Shape (if return_per_token then _ else _) then Shape.squeeze batch else []) A
      := (let logits : tensor (batch ::' _) A
            := PArray.maybe_checkpoint (logits.[…, -1, :]) in
          let true_maximum : tensor (batch ::' 1) IndexType
            := Tensor.max_dim_m1 (keepdim:=true) tokens in
          let log_probs
            := Tensor.log_softmax_dim_m1 logits in
          let correct_log_probs
            := PArray.maybe_checkpoint (gather_dim_m1 log_probs true_maximum) in
          if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) A)
          then -Tensor.squeeze correct_log_probs
          else -Tensor.mean correct_log_probs)%core.

    Definition acc_fn
      (logits : tensor (s ::' cfg.d_vocab_out) A)
      (tokens : tensor s IndexType)
      : tensor (if return_per_token return Shape (if return_per_token then _ else _) then batch else []) A
      := (let pred_logits : tensor (batch ::' _) A
            := PArray.maybe_checkpoint (logits.[…, -1, :]) in
          let pred_tokens : tensor batch IndexType
            := Tensor.argmax_dim_m1 (keepdim:=false) pred_logits in
          let true_maximum : tensor batch IndexType
            := Tensor.max_dim_m1 (keepdim:=false) tokens in
          let res : tensor _ A
            := PArray.maybe_checkpoint (Tensor.of_bool (Tensor.map2 eqb pred_tokens true_maximum)) in
          if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) A)
          then res
          else Tensor.mean res)%core.
  End with_batch.

  Definition true_accuracy : float
    := Tensor.item (acc_fn (return_per_token := false) (logits all_tokens) all_tokens).
  Definition true_loss : float
    := Tensor.item (loss_fn (return_per_token := false) (logits all_tokens) all_tokens).
End Model.
