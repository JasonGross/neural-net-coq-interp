From Coq Require Import Morphisms RelationClasses RelationPairs.
From Coq Require Vector.
From Coq.Structures Require Import Equalities.
From Coq Require Import Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel Option Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Util.List.Instances Require Import Forall2.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
From NeuralNetInterp.TransformerLens.HookedTransformer Require Import Config Instances Module.
Import Dependent.ProperNotations Dependent.RelationPairsNotations.
Import Instances.Truncating.
#[local] Open Scope core_scope.

Module ModelInstances (cfg : Config) (Model : ModelSig cfg).
  Export Model.

  Local Ltac t :=
    try eassumption;
    auto;
    repeat intro;
    try (eapply Tensor.map_Proper_dep; try eassumption; repeat intro);
    try reflexivity.

  Module Embed.
    Import Instances.HookedTransformer.Embed.
    Export Model.Embed.

    #[export] Instance forward_Proper_dep {r batch pos}
      : Dependent.Proper
          ((Dependent.const eq ==> Dependent.idR)
             ==> Dependent.const Tensor.eqf
             ==> Tensor.eqfR)
          (@forward r batch pos).
    Proof. cbv [forward]; repeat intro; apply forward_Proper_dep; t. Qed.
(*
    #[export] Instance forward_Proper {A d_vocab d_model W_E r s}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@forward A d_vocab d_model W_E r s).
    Proof. apply forward_Proper_dep; reflexivity. Qed.
  End Embed.
  Export (hints) Embed.

  Module Unembed.
    Export Unembed.
    #[export] Instance forward_Proper_dep {r batch_pos d_model d_vocab_out}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> Tensor.eqfR ==> Tensor.eqfR
             ==> Tensor.eqfR ==> Tensor.eqfR)
          (@forward r batch_pos d_model d_vocab_out).
    Proof. cbv [forward]; t. Qed.
    #[export] Instance forward_Proper {A addA mulA zeroA d_model d_vocab_out W_U b_U r batch_pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@forward A addA mulA zeroA d_model d_vocab_out W_U b_U r batch_pos).
    Proof. apply forward_Proper_dep; repeat intro; subst; reflexivity. Qed.
  End Unembed.
  Export (hints) Unembed.

  Module PosEmbed.
    Export PosEmbed.

    #[export] Instance forward_Proper_dep {r batch_pos tokens_length d_model n_ctx}
      : Dependent.Proper
          (Tensor.eqfR
             ==> Dependent.const Tensor.eqf
             ==> Tensor.eqfR)
          (@forward r batch_pos tokens_length d_model n_ctx).
    Proof. cbv [forward]; t. Qed.

    #[export] Instance forward_Proper {A d_model n_ctx W_pos r batch tokens_length}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@forward A d_model n_ctx W_pos r batch tokens_length).
    Proof. cbv [forward]; t. Qed.
  End PosEmbed.
  Export (hints) PosEmbed.

  Module LayerNorm.
    Export LayerNorm.

    #[export] Instance linpart_Proper_dep {r s d_model}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> Tensor.eqfR ==> Tensor.eqfR)
          (@linpart r s d_model).
    Proof. cbv [linpart]; t. Qed.

    #[export] Instance linpart_Proper {r A s d_model addA subA divA zeroA coerZ}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@linpart r A s d_model addA subA divA zeroA coerZ).
    Proof. apply linpart_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance scale_Proper_dep {r s d_model}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> Dependent.idR
             ==> Tensor.eqfR ==> Tensor.eqfR)
          (@scale r s d_model).
    Proof. cbv [scale]; t. Qed.

    #[export] Instance scale_Proper {r A s d_model addA mulA divA sqrtA zeroA coerZ eps}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@scale r A s d_model addA mulA divA sqrtA zeroA coerZ eps).
    Proof. apply scale_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance rescale_Proper_dep {r s d_model}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR)
          (@rescale r s d_model).
    Proof. cbv [rescale]; t. Qed.

    #[export] Instance rescale_Proper {r A s d_model divA}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@rescale r A s d_model divA).
    Proof. apply rescale_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance postrescale_Proper_dep {r s d_model}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR)
          (@postrescale r s d_model).
    Proof. cbv [postrescale]; t. Qed.

    #[export] Instance postrescale_Proper {r A s d_model addA mulA}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@postrescale r A s d_model addA mulA).
    Proof. apply postrescale_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance forward_Proper_dep {r s d_model}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR)
          (@forward r s d_model).
    Proof. cbv [forward]; t.

    Qed.
  End LayerNorm.
  Export (hints) LayerNorm.

  Module Attention.
    Export Attention.

    #[export] Instance einsum_input_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR)
          (@einsum_input r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [einsum_input]; t. Qed.

    #[export] Instance einsum_input_Proper {r A batch pos n_heads d_model d_head use_split_qkv_input addA zeroA mulA}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@einsum_input r A batch pos n_heads d_model d_head use_split_qkv_input addA zeroA mulA).
    Proof. apply einsum_input_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance q_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@q r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [q]; repeat first [ t_step | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance q_Proper {A r batch pos n_heads d_model d_head use_split_qkv_input coerZ addA zeroA mulA use_checkpoint W_Q b_Q}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@q A r batch pos n_heads d_model d_head use_split_qkv_input coerZ addA zeroA mulA use_checkpoint W_Q b_Q).
    Proof. apply q_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance k_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@k r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [k]; repeat first [ t_step | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance k_Proper {A r batch pos n_heads d_model d_head use_split_qkv_input coerZ addA zeroA mulA use_checkpoint W_K b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@k A r batch pos n_heads d_model d_head use_split_qkv_input coerZ addA zeroA mulA use_checkpoint W_K b_K).
    Proof. apply k_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance v_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@v r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [v]; repeat first [ t_step | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance v_Proper {A r batch pos n_heads d_model d_head use_split_qkv_input coerZ addA zeroA mulA use_checkpoint W_V b_V}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@v A r batch pos n_heads d_model d_head use_split_qkv_input coerZ addA zeroA mulA use_checkpoint W_V b_V).
    Proof. apply v_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance attn_scores_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR)
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@attn_scores r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [attn_scores]; repeat first [ t_step | apply q_Proper_dep | apply k_Proper_dep | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance attn_scores_Proper {r batch pos n_heads d_model d_head use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA use_checkpoint W_Q W_K b_Q b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@attn_scores r batch pos n_heads d_model d_head use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA use_checkpoint W_Q W_K b_Q b_K).
    Proof. apply attn_scores_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance apply_causal_mask_Proper_dep {r batch pos n_heads n_ctx}
      : Dependent.Proper
          ((Dependent.const eq ==> Dependent.idR)
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@apply_causal_mask r batch pos n_heads n_ctx).
    Proof. cbv [apply_causal_mask]; t. Qed.

    #[export] Instance apply_causal_mask_Proper {r batch pos n_heads n_ctx A coerZ}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@apply_causal_mask r batch pos n_heads n_ctx A coerZ).
    Proof. apply apply_causal_mask_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance masked_attn_scores_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR)
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@masked_attn_scores r batch pos n_heads d_model d_head n_ctx use_split_qkv_input).
    Proof. cbv [masked_attn_scores]; repeat first [ t_step | apply apply_causal_mask_Proper_dep | apply attn_scores_Proper_dep ]. Qed.

    #[export] Instance masked_attn_scores_Proper {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA use_checkpoint W_Q W_K b_Q b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@masked_attn_scores r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA use_checkpoint W_Q W_K b_Q b_K).
    Proof. apply masked_attn_scores_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance pattern_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR)
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@pattern r batch pos n_heads d_model d_head n_ctx use_split_qkv_input).
    Proof. cbv [pattern]; repeat first [ t_step | apply masked_attn_scores_Proper_dep ]. Qed.

    #[export] Instance pattern_Proper {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA expA use_checkpoint W_Q W_K b_Q b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@pattern r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA expA use_checkpoint W_Q W_K b_Q b_K).
    Proof. apply pattern_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance z_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR)
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@z r batch pos n_heads d_model d_head n_ctx use_split_qkv_input).
    Proof. cbv [z]; repeat first [ t_step | apply v_Proper_dep | apply pattern_Proper_dep | apply einsum_input_Proper_dep | apply masked_attn_scores_Proper_dep ]. Qed.

    #[export] Instance z_Proper {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA expA use_checkpoint W_Q W_K W_V b_Q b_K b_V}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf)
          (@z r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA expA use_checkpoint W_Q W_K W_V b_Q b_K b_V).
    Proof. apply z_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance attn_out_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR)
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@attn_out r batch pos n_heads d_model d_head n_ctx use_split_qkv_input).
    Proof. cbv [attn_out]; repeat first [ t_step | apply z_Proper_dep ]. Qed.

    #[export] Instance attn_out_Proper {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA expA use_checkpoint W_Q W_K W_V W_O b_Q b_K b_V b_O}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf)
          (@attn_out r batch pos n_heads d_model d_head n_ctx use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA expA use_checkpoint W_Q W_K W_V W_O b_Q b_K b_V b_O).
    Proof. apply attn_out_Proper_dep; repeat intro; subst; reflexivity. Qed.
  End Attention.
  Export (hints) Attention.

  Module TransformerBlock.
    Export TransformerBlock.

    #[export] Instance add_head_dimension_Proper_dep {r batch pos n_heads d_model use_split_qkv_input}
      : Dependent.Proper
          (Tensor.eqfR ==> Tensor.eqfR)
          (@add_head_dimension r batch pos n_heads d_model use_split_qkv_input).
    Proof. cbv [add_head_dimension]; t. Qed.

    #[export] Instance add_head_dimension_Proper {r batch pos n_heads d_model use_split_qkv_input A}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@add_head_dimension r batch pos n_heads d_model use_split_qkv_input A).
    Proof. apply add_head_dimension_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance query_input_Proper_dep {r batch pos n_heads d_model use_split_qkv_input}
      : Dependent.Proper (Tensor.eqfR ==> Tensor.eqfR) (@query_input r batch pos n_heads d_model use_split_qkv_input)
      := _.
    #[export] Instance query_input_Proper {r batch pos n_heads d_model use_split_qkv_input A}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@query_input r batch pos n_heads d_model use_split_qkv_input A)
      := _.
    #[export] Instance key_input_Proper_dep {r batch pos n_heads d_model use_split_qkv_input}
      : Dependent.Proper (Tensor.eqfR ==> Tensor.eqfR) (@key_input r batch pos n_heads d_model use_split_qkv_input)
      := _.
    #[export] Instance key_input_Proper {r batch pos n_heads d_model use_split_qkv_input A}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@key_input r batch pos n_heads d_model use_split_qkv_input A)
      := _.
    #[export] Instance value_input_Proper_dep {r batch pos n_heads d_model use_split_qkv_input}
      : Dependent.Proper (Tensor.eqfR ==> Tensor.eqfR) (@value_input r batch pos n_heads d_model use_split_qkv_input)
      := _.
    #[export] Instance value_input_Proper {r batch pos n_heads d_model use_split_qkv_input A}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@value_input r batch pos n_heads d_model use_split_qkv_input A)
      := _.

    #[export] Instance ln1_Proper_dep {r batch pos n_heads d_model use_split_qkv_input normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@ln1 r batch pos n_heads d_model use_split_qkv_input normalization_type).
    Proof. cbv [ln1]; repeat first [ t_step | apply LayerNorm.forward_Proper_dep ]. Qed.

    #[export] Instance ln1_Proper {r batch pos n_heads d_model use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@ln1 r batch pos n_heads d_model use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint ln1_w ln1_b).
    Proof. apply ln1_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance ln2_Proper_dep {r batch pos n_heads d_model use_split_qkv_input normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@ln2 r batch pos n_heads d_model use_split_qkv_input normalization_type).
    Proof. cbv [ln2]; repeat first [ t_step | apply LayerNorm.forward_Proper_dep ]. Qed.

    #[export] Instance ln2_Proper {r batch pos n_heads d_model use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint ln2_w ln2_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@ln2 r batch pos n_heads d_model use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint ln2_w ln2_b).
    Proof. apply ln2_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance attn_only_out_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Dependent.idR
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@attn_only_out r batch pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type).
    Proof.
      cbv [attn_only_out]; t.
      all: apply Attention.attn_out_Proper_dep; t.
      all: apply ln1_Proper_dep; t.
      all: first [ apply query_input_Proper_dep | apply key_input_Proper_dep | apply value_input_Proper_dep ]; t.
    Qed.

    #[export] Instance attn_only_out_Proper {r batch pos n_heads d_model d_head n_ctx use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA W_Q W_K W_V W_O b_Q b_K b_V b_O eps use_checkpoint ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@attn_only_out r batch pos n_heads d_model d_head n_ctx use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA W_Q W_K W_V W_O b_Q b_K b_V b_O eps use_checkpoint ln1_w ln1_b).
    Proof. apply attn_only_out_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance attn_masked_attn_scores_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Dependent.idR
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@HookedTransformer.TransformerBlock.attn_masked_attn_scores r batch pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type).
    Proof.
      cbv [HookedTransformer.TransformerBlock.attn_masked_attn_scores]; t.
      all: apply Attention.masked_attn_scores_Proper_dep; t.
      all: apply ln1_Proper_dep; t.
      all: first [ apply query_input_Proper_dep | apply key_input_Proper_dep ]; t.
    Qed.

    #[export] Instance attn_masked_attn_scores_Proper {r batch pos n_heads d_model d_head n_ctx use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA W_Q W_K b_Q b_K eps use_checkpoint ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@HookedTransformer.TransformerBlock.attn_masked_attn_scores r batch pos n_heads d_model d_head n_ctx use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA W_Q W_K b_Q b_K eps use_checkpoint ln1_w ln1_b).
    Proof. apply attn_masked_attn_scores_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance attn_pattern_Proper_dep {r batch pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Dependent.idR
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@HookedTransformer.TransformerBlock.attn_pattern r batch pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type).
    Proof.
      cbv [HookedTransformer.TransformerBlock.attn_pattern]; t.
      all: apply Attention.pattern_Proper_dep; t.
      all: apply ln1_Proper_dep; t.
      all: first [ apply query_input_Proper_dep | apply key_input_Proper_dep ]; t.
    Qed.

    #[export] Instance attn_pattern_Proper {r batch pos n_heads d_model d_head n_ctx use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA W_Q W_K b_Q b_K eps use_checkpoint ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@HookedTransformer.TransformerBlock.attn_pattern r batch pos n_heads d_model d_head n_ctx use_split_kqv_input normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA W_Q W_K b_Q b_K eps use_checkpoint ln1_w ln1_b).
    Proof. apply attn_pattern_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.
  End TransformerBlock.
  Export (hints) TransformerBlock.

  Module HookedTransformer.
    Export HookedTransformer.

    #[export] Instance embed_Proper_dep {d_vocab d_model r batch pos}
      : Dependent.Proper
          (Tensor.eqfR ==> Dependent.const Tensor.eqf ==> Tensor.eqfR)
          (@embed d_vocab d_model r batch pos)
      := _.
    #[export] Instance embed_Proper {d_vocab d_model r batch pos A W_E}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@embed d_vocab d_model r batch pos A W_E)
      := _.

    #[export] Instance pos_embed_Proper_dep {d_model n_ctx r batch pos}
      : Dependent.Proper
          (Tensor.eqfR ==> Dependent.const Tensor.eqf ==> Tensor.eqfR)
          (@pos_embed d_model n_ctx r batch pos)
      := _.
    #[export] Instance pos_embed_Proper {d_model n_ctx r batch pos A W_pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@pos_embed d_model n_ctx r batch pos A W_pos)
      := _.

    #[export] Instance resid_postembed_Proper_dep {d_vocab d_model n_ctx r batch pos}
      : Dependent.Proper
          ((Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Tensor.eqfR ==> Tensor.eqfR
             ==> Dependent.const Tensor.eqf ==> Tensor.eqfR)
          (@resid_postembed d_vocab d_model n_ctx r batch pos).
    Proof. cbv [resid_postembed]; t; first [ apply embed_Proper_dep | apply pos_embed_Proper_dep ]; t. Qed.

    #[export] Instance resid_postembed_Proper {d_vocab d_model n_ctx r batch pos A coerZ addA use_checkpoint W_E W_pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@resid_postembed d_vocab d_model n_ctx r batch pos A coerZ addA use_checkpoint W_E W_pos).
    Proof. apply resid_postembed_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Polymorphic Instance blocks_Proper_dep {n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> List.Forall2 ∘ (Tensor.eqfR ==> Tensor.eqfR))
          (@blocks n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [Dependent.Proper Dependent.respectful blocks]; clear.
      repeat first [ lazymatch goal with H : Forall2 _ _ _ |- _ => fail 1 end | intro ].
      let H := match goal with H : Forall2 _ _ _ |- _ => H end in
      induction H; cbn [List.map]; constructor; auto; [].
      destruct_head'_prod; intros.
      apply TransformerBlock.attn_only_out_Proper_dep; t.
    Qed.

    #[export] Polymorphic Instance blocks_Proper {n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint blocks_params}
      : Proper (List.Forall2 (Tensor.eqf ==> Tensor.eqf))%signature (@blocks n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint blocks_params).
    Proof. apply blocks_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance ln_final_Proper_dep {d_model r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@ln_final d_model r batch pos normalization_type).
    Proof. cbv [ln_final]; repeat first [ t_step | apply LayerNorm.forward_Proper_dep ]. Qed.

    #[export] Instance ln_final_Proper {d_model r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint ln_final_w ln_final_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@ln_final d_model r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint ln_final_w ln_final_b).
    Proof. apply ln_final_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance unembed_Proper_dep {d_vocab_out d_model r batch pos}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@unembed d_vocab_out d_model r batch pos).
    Proof. cbv [unembed]; repeat intro; apply Unembed.forward_Proper_dep; t. Qed.

    #[export] Instance unembed_Proper {d_vocab_out d_model r batch pos A zeroA addA mulA W_U b_U}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@unembed d_vocab_out d_model r batch pos A zeroA addA mulA W_U b_U)
      := _.

    #[export] Instance blocks_cps_Proper_dep {n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> Dependent.forall_relation
             (Dependent.const2 (@eq nat (* strip dependency of with_default *))
                ==> Dependent.lift2_1 Tensor.eqfR
                ==> (Dependent.lift2_1 Tensor.eqfR ==> Dependent.lift2_2 Dependent.idR)
                ==> Dependent.lift2_2 Dependent.idR))
          (@blocks_cps n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [Dependent.Proper Dependent.respectful blocks_cps with_default Dependent.forall_relation Dependent.respectful2]; clear.
      repeat first [ lazymatch goal with H : Forall2 _ _ _ |- _ => fail 1 end | intro ].
      let H := match goal with H : Forall2 _ _ _ |- _ => H end in
      induction H; cbn [fold_right List.map].
      all: intros; subst.
      all: let n := lazymatch goal with |- context[firstn ?n] => n end in
           is_var n; destruct n; cbn [fold_right firstn].
      all: eauto.
      all: [ > ].
      cbv [blocks] in *; cbn [List.map firstn fold_right] in *.
      let IH := multimatch goal with H : _ |- _ => H end in
      apply IH; clear IH; repeat intro; eauto; [].
      break_innermost_match; rewrite ?Tensor.PArray.checkpoint_correct.
      all: apply TransformerBlock.attn_only_out_Proper_dep; t.
    Qed.

    #[export] Instance blocks_cps_Proper {n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint blocks_params T R}
      : Proper (eq ==> Tensor.eqf ==> (Tensor.eqf ==> R) ==> R)
          (@blocks_cps n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint blocks_params T).
    Proof. apply blocks_cps_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance logits_Proper_dep {d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Dependent.const Tensor.eqf
             ==> Tensor.eqfR)
          (@logits d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [logits]; t.
      all: lazymatch goal with
           | [ |- ?R (?f ?i) (?g ?i) ]
             => revert i; change ((fun F G => forall i, R (F i) (G i)) f g)
           end.
      all: eapply blocks_cps_Proper_dep; [ eassumption | .. ].
      all: repeat intro; subst; t.
      all: try now eapply Forall2_length, blocks_Proper_dep; eassumption.
      all: first [ apply resid_postembed_Proper_dep | apply embed_Proper_dep | apply pos_embed_Proper_dep | apply ln_final_Proper_dep | apply unembed_Proper_dep ]; t.
    Qed.

    #[export] Instance logits_Proper {d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U}
      : Proper (Tensor.eqf ==> Tensor.eqf)
          (@logits d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U).
    Proof. apply logits_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance forward_Proper_dep {d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
             | Some LN => Tensor.eqfR (s:=[d_model])
             | None => Dependent.const eq
             end
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Dependent.const Tensor.eqf
             ==> Tensor.eqfR)
          (@forward d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type)
      := _.

    #[export] Instance forward_Proper {d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U}
      : Proper (Tensor.eqf ==> Tensor.eqf)
          (@forward d_vocab d_vocab_out n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U)
      := _.

    #[export] Instance blocks_attn_masked_attn_scores_Proper_dep {n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> List.Forall2 ∘ (Tensor.eqfR ==> Tensor.eqfR))
          (@HookedTransformer.HookedTransformer.blocks_attn_masked_attn_scores n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [Dependent.Proper HookedTransformer.HookedTransformer.blocks_attn_masked_attn_scores]; clear.
      repeat first [ lazymatch goal with H : Forall2 _ _ _ |- _ => fail 1 end | intro ].
      let H := match goal with H : Forall2 _ _ _ |- _ => H end in
      induction H; cbn [List.map]; constructor; auto; [].
      destruct_head'_prod.
      apply TransformerBlock.attn_masked_attn_scores_Proper_dep; t.
    Qed.

    #[export] Instance blocks_attn_masked_attn_scores_Proper {n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint blocks_params}
      : Proper (List.Forall2 (Tensor.eqf ==> Tensor.eqf))%signature
          (@HookedTransformer.HookedTransformer.blocks_attn_masked_attn_scores n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA eps use_checkpoint blocks_params).
    Proof. apply blocks_attn_masked_attn_scores_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance blocks_attn_pattern_Proper_dep {n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> List.Forall2 ∘ (Tensor.eqfR ==> Tensor.eqfR))
          (@HookedTransformer.HookedTransformer.blocks_attn_pattern n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [Dependent.Proper HookedTransformer.HookedTransformer.blocks_attn_pattern]; clear.
      repeat first [ lazymatch goal with H : Forall2 _ _ _ |- _ => fail 1 end | intro ].
      let H := match goal with H : Forall2 _ _ _ |- _ => H end in
      induction H; cbn [List.map]; constructor; auto; [].
      destruct_head'_prod.
      apply TransformerBlock.attn_pattern_Proper_dep; t.
    Qed.

    #[export] Instance blocks_attn_pattern_Proper {n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint blocks_params}
      : Proper (List.Forall2 (Tensor.eqf ==> Tensor.eqf))%signature
          (@HookedTransformer.HookedTransformer.blocks_attn_pattern n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint blocks_params).
    Proof. apply blocks_attn_pattern_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance masked_attn_scores_Proper_dep {d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> Dependent.const eq
             ==> Dependent.const Tensor.eqf
             ==> @option_eq ∘ Tensor.eqfR)
          (@HookedTransformer.HookedTransformer.masked_attn_scores d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [HookedTransformer.HookedTransformer.masked_attn_scores option_eq]; t.
      all: break_innermost_match_hyps; inversion_option; subst.
      all: lazymatch goal with
           | [ Hx : nth_error ?xs ?n = _, Hy : nth_error ?ys ?n = _ |- _ ]
             => let H' := fresh in
                unshelve
                  (epose proof (List.nth_error_Proper_dep_Forall2
                                  _ _ _
                                  xs ys
                                  (blocks_attn_masked_attn_scores_Proper_dep _ _ _ _ _ _ _ _ _ _ _ _ _ _)
                                  n n eq_refl) as H';
                   rewrite Hx, Hy in H'; cbv [option_eq] in H'; clear Hx Hy;
                   try now exfalso);
                shelve_unifiable; repeat intro; cbv beta; subst; try eassumption; eauto
           end.
      all: lazymatch goal with
           | [ |- ?R (?f ?i) (?g ?i) ]
             => revert i; change ((fun F G => forall i, R (F i) (G i)) f g)
           end.
      all: eapply blocks_cps_Proper_dep; [ eassumption | .. ].
      all: repeat intro; subst; t.
      all: try now eapply Forall2_length, blocks_Proper_dep; eassumption.
      all: first [ apply resid_postembed_Proper_dep | apply embed_Proper_dep | apply pos_embed_Proper_dep | apply ln_final_Proper_dep | apply unembed_Proper_dep ]; t.
    Qed.

    #[export] Instance masked_attn_scores_Proper {d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params}
      : Proper (eq ==> Tensor.eqf ==> option_eq Tensor.eqf)
          (@HookedTransformer.HookedTransformer.masked_attn_scores d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params).
    Proof. apply masked_attn_scores_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.

    #[export] Instance attn_pattern_Proper_dep {d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type}
      : Dependent.Proper
          (Dependent.idR
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR)
             ==> Dependent.const (fun _ _ => True)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> List.Forall2 ∘ (Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * Tensor.eqfR * Tensor.eqfR
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end
                                 * match normalization_type return Dependent.relation (fun A => match normalization_type with Some LN => _ | None => _ end) with
                                   | Some LN => Tensor.eqfR (s:=[d_model])
                                   | None => Dependent.const eq
                                   end)
             ==> Dependent.const eq
             ==> Dependent.const Tensor.eqf
             ==> @option_eq ∘ Tensor.eqfR)
          (@HookedTransformer.HookedTransformer.attn_pattern d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type).
    Proof.
      cbv [HookedTransformer.HookedTransformer.attn_pattern option_eq]; t.
      all: break_innermost_match_hyps; inversion_option; subst.
      all: lazymatch goal with
           | [ Hx : nth_error ?xs ?n = _, Hy : nth_error ?ys ?n = _ |- _ ]
             => let H' := fresh in
                unshelve
                  (epose proof (List.nth_error_Proper_dep_Forall2
                                  _ _ _
                                  xs ys
                                  (blocks_attn_pattern_Proper_dep _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)
                                  n n eq_refl) as H';
                   rewrite Hx, Hy in H'; cbv [option_eq] in H'; clear Hx Hy;
                   try now exfalso);
                shelve_unifiable; repeat intro; cbv beta; subst; try eassumption; eauto
           end.
      all: lazymatch goal with
           | [ |- ?R (?f ?i) (?g ?i) ]
             => revert i; change ((fun F G => forall i, R (F i) (G i)) f g)
           end.
      all: eapply blocks_cps_Proper_dep; [ eassumption | .. ].
      all: repeat intro; subst; t.
      all: first [ apply resid_postembed_Proper_dep | apply embed_Proper_dep | apply pos_embed_Proper_dep | apply ln_final_Proper_dep | apply unembed_Proper_dep ]; t.
    Qed.

    #[export] Instance attn_pattern_Proper {d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params}
      : Proper (eq ==> Tensor.eqf ==> option_eq Tensor.eqf)
          (@HookedTransformer.HookedTransformer.attn_pattern d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type A zeroA coerZ addA subA mulA divA sqrtA expA eps use_checkpoint W_E W_pos blocks_params).
    Proof. apply attn_pattern_Proper_dep; repeat intro; subst; break_innermost_match; reflexivity. Qed.
  End HookedTransformer.
  Export (hints) HookedTransformer.
End HookedTransformer.
Export (hints) HookedTransformer.
 *)
  End Embed.
End ModelInstances.
