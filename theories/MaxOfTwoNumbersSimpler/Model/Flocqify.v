From Coq Require Import ZArith.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util Require Import Default Arith.Classes Arith.Instances Arith.Flocq Arith.Flocq.Instances Arith.Flocq.Definitions.
From NeuralNetInterp.Util.Tactics Require Import Head BreakMatch.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
From NeuralNetInterp.TransformerLens.HookedTransformer Require Import Instances Module.Instances Module.Flocqify.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model Model.Instances.
Import Dependent.ProperNotations.
Import Arith.Instances.Truncating Arith.Flocq.Instances.Truncating.

Module Model.
  Export Model.Instances.Model.
  Include ModelFlocqify cfg Model.Model Instances.Model.
  Include HookedTransformer.

  Section with_batch.
    Context {r} {batch : Shape r} {pos}
      (s := (batch ::' pos)%shape)
      (resid_shape := (s ::' cfg.d_model)%shape)
      {return_per_token : with_default "return_per_token" bool false}
      {use_checkpoint1 : with_default "use_checkpoint" bool true}
      {use_checkpoint2 : bool}.
(*
    Lemma embed_equiv (tokens : tensor s IndexType)
      : Tensor.eqfR Rf (embed tokens) (embed tokens).
    Proof using Type. apply embed_Proper_dep; repeat intro; subst; try reflexivity. Qed.

    Lemma pos_embed_equiv (tokens : tensor s IndexType)
      : Tensor.eqfR Rf (pos_embed tokens) (pos_embed tokens).
    Proof using Type. apply pos_embed_Proper_dep; repeat intro; subst; try reflexivity. Qed.
*)
    (*
    Lemma ln_final_equiv (resid Bresid : tensor resid_shape _)
      : Tensor.eqfR Rf resid
          Tensor.eqfR Rf (ln_final resid) (ln_final (Tensor.map Prim2B resid)).
    Proof using Type. apply ln_final_Proper_dep; repeat intro; subst; try reflexivity. Qed.


  Definition ln_final (resid : tensor resid_shape A) : tensor resid_shape A
    := HookedTransformer.ln_final (A:=A) cfg.eps ln_final_w ln_final_b resid.

  Definition unembed (resid : tensor resid_shape A) : tensor (s ::' cfg.d_vocab_out) A
    := HookedTransformer.unembed (A:=A) W_U b_U resid.
     *)
    (*Definition blocks_params : list _
    := [(L0_attn_W_Q:tensor _ A, L0_attn_W_K:tensor _ A, L0_attn_W_V:tensor _ A, L0_attn_W_O:tensor _ A,
          L0_attn_b_Q:tensor _ A, L0_attn_b_K:tensor _ A, L0_attn_b_V:tensor _ A,
          L0_attn_b_O:tensor _ A,
          L0_ln1_w:tensor _ A, L0_ln1_b:tensor _ A)].
     *)

    Local Ltac t :=
      try assumption;
      repeat intro; subst; try reflexivity;
      try (apply B2Prim_inj; reflexivity);
      lazymatch goal with
      | [ |- R ?x ?y ] => let x := head x in let y := head y in cbv [x y]
      | [ |- ?x = ?y :> bool ] => let x := head x in let y := head y in cbv [x y]
      | [ |- ?G ] => fail 0 "unrecognized" G
      end;
      repeat autorewrite with prim2b;
      try reflexivity.
(*
    Lemma masked_attn_scores_equiv (tokens : tensor s IndexType)
      : Tensor.eqfR Rf
          (masked_attn_scores (use_checkpoint:=use_checkpoint1) tokens)
          (masked_attn_scores (use_checkpoint:=use_checkpoint2) tokens).
    Proof using Type. apply masked_attn_scores_Proper_dep; t. Qed.

    Lemma attn_pattern_equiv (tokens : tensor s IndexType)
      : Tensor.eqfR Rf
          (attn_pattern (use_checkpoint:=use_checkpoint1) tokens)
          (attn_pattern (use_checkpoint:=use_checkpoint2) tokens).
    Proof using Type. apply attn_pattern_Proper_dep; t. Qed.
*)
    Lemma loss_fn_equiv (logits logits' : tensor (s ::' cfg.d_vocab_out) _)
      (tokens tokens' : tensor s IndexType)
      : Tensor.eqfR Rf logits logits'
        -> Tensor.eqf tokens tokens'
        -> Tensor.eqfR Rf
             (loss_fn (use_checkpoint:=use_checkpoint1) logits tokens)
             (loss_fn (use_checkpoint:=use_checkpoint2) logits' tokens').
    Proof using Type.
      intros; apply loss_fn_Proper_dep; t.
      cbv [Classes.leb binary_float_has_leb]; break_innermost_match; reflexivity.
    Qed.

    Lemma acc_fn_equiv (logits logits' : tensor (s ::' cfg.d_vocab_out) _)
      (tokens tokens' : tensor s IndexType)
      : Tensor.eqfR Rf logits logits'
        -> Tensor.eqf tokens tokens'
        -> Tensor.eqfR Rf
             (acc_fn (use_checkpoint:=use_checkpoint1) logits tokens)
             (acc_fn (use_checkpoint:=use_checkpoint2) logits' tokens').
    Proof using Type. intros; apply acc_fn_Proper_dep; t. Qed.
  End with_batch.
End Model.
