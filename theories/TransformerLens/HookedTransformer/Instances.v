From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default Pointed PArray PArray.Instances Wf_Uint63.Instances List Notations Arith.Classes Arith.Instances Bool SolveProperEqRel Option List.Instances.NthError.
From NeuralNetInterp.Util.Tactics Require Import DestructHead BreakMatch.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Einsum Slicing Slicing.Instances.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Hetero Relation_Definitions.Dependent.
From NeuralNetInterp.Util.Classes Require Morphisms.Dependent.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Import Dependent.ProperNotations.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Local Open Scope raw_tensor_scope.
#[local] Generalizable All Variables.

Module HookedTransformer.
  Export HookedTransformer.
Print Reduction.sum.
  Ltac t_step :=
    first [ match goal with
            | [ |- ?x = ?x ] => reflexivity
            | [ |- ?R (PArray.checkpoint _ _) (PArray.checkpoint _ _) ]
              => apply Tensor.PArray.checkpoint_Proper_dep
            | [ |- ?R (Tensor.of_bool _ _) (Tensor.of_bool _ _) ]
              => apply (Tensor.of_bool_Proper_dep _ _ R)
            | [ |- Tensor.map2 _ _ _ _ = Tensor.map2 _ _ _ _ ]
              => apply Tensor.map2_Proper
           | [ |- ?R (Tensor.map2' _ _ _ _) (Tensor.map2' _ _ _ _) ]
             => apply (Tensor.map2'_Proper_dep _ _ R _ _ R)
            | [ |- Tensor.map _ _ _ = Tensor.map _ _ _ ]
              => apply Tensor.map_Proper
            | [ |- ?R (Tensor.map' _ _ _) (Tensor.map' _ _ _) ]
              => apply (Tensor.map'_Proper_dep _ _ R)
            | [ |- Tensor.squeeze _ _ = Tensor.squeeze _ _ ]
              => apply Tensor.squeeze_Proper
            | [ |- reduce_axis_m1 _ _ _ = reduce_axis_m1 _ _ _ ]
              => apply @Tensor.reduce_axis_m1_Proper with (RA:=eq)
            | [ |- Tensor.gather_dim_m1 _ _ _ = Tensor.gather_dim_m1 _ _ _ ]
              => apply Tensor.gather_dim_m1_Proper
            | [ |- Tensor.softmax_dim_m1 _ _ = Tensor.softmax_dim_m1 _ _ ]
              => apply Tensor.softmax_dim_m1_Proper
            | [ |- Tensor.log_softmax_dim_m1 _ _ = Tensor.log_softmax_dim_m1 _ _ ]
              => apply Tensor.log_softmax_dim_m1_Proper
            | [ |- ?R (Tensor.mean _ _) (Tensor.mean _ _) ]
              => apply (Tensor.mean_Proper_dep _ _ R)
            | [ |- ?R (Tensor.where_ _ _ _ _) (Tensor.where_ _ _ _ _) ]
              => apply (Tensor.where__Proper_dep _ _ R)
            | [ |- ?R (Tensor.repeat _ _ _) (Tensor.repeat _ _ _) ]
              => apply (Tensor.repeat_Proper_dep _ _ R)
           | [ |- ?R (Tensor.broadcast _ _) (Tensor.broadcast _ _) ]
             => apply (Tensor.broadcast_Proper_dep _ _ R)
            | [ |- ?R (@SliceIndex.slice ?A ?ri ?ro ?idxs ?s _ _) (@SliceIndex.slice ?A ?ri ?ro ?idxs ?s _ _) ]
              => eapply (@SliceIndex.slice_Proper A ri ro idxs s R)
            | [ |- ?R (@FancyIndex.slice ?rb ?sb ?ri ?ro ?s ?A ?idxs _ _) (@FancyIndex.slice ?rb ?sb ?ri ?ro ?s ?A' ?idxs' _ _) ]
              => apply (@FancyIndex.slice_Proper_dep rb sb ri ro s)
            | [ |- ?R (Reduction.argmax _ _ _ _) (Reduction.argmax _ _ _ _) ]
              => apply (Reduction.argmax_Proper_dep _ _ R)
            | [ |- ?R (Reduction.argmin _ _ _ _) (Reduction.argmin _ _ _ _) ]
              => apply (Reduction.argmin_Proper_dep _ _ R)
            | [ |- ?R (Reduction.max _ _ _ _) (Reduction.max _ _ _ _) ]
              => apply (Reduction.max_Proper_dep _ _ R)
            | [ |- ?R (Reduction.min _ _ _ _) (Reduction.min _ _ _ _) ]
              => apply (Reduction.min_Proper_dep _ _ R)
            | [ |- ?R (Reduction.mean _ _ _ _) (Reduction.mean _ _ _ _) ]
              => apply (Reduction.mean_Proper_dep _ _ R)
            | [ |- ?R (Reduction.var _ _ _ _) (Reduction.var _ _ _ _) ]
              => apply (Reduction.var_Proper_dep _ _ R)
            | [ |- ?R (Reduction.sum _ _ _ _) (Reduction.sum _ _ _ _) ]
              => apply (Reduction.sum_Proper_dep _ _ R)
            | [ |- ?R (Reduction.prod _ _ _ _) (Reduction.prod _ _ _ _) ]
              => apply (Reduction.prod_Proper_dep _ _ R)
            | [ |- FancyIndex.t_relation _ _ ] => constructor
            | [ |- FancyIndexType_relation _ _ ] => hnf
            | [ |- ?R (Classes.add _ _ ?idx) (Classes.add _ _ ?idx) ]
              => apply (Tensor.tensor_add_Proper_dep _ _ R _ _ R _ _ R)
            | [ |- ?R (Classes.sub _ _ ?idx) (Classes.sub _ _ ?idx) ]
              => apply (Tensor.tensor_sub_Proper_dep _ _ R _ _ R _ _ R)
            | [ |- ?R (Classes.mul _ _ ?idx) (Classes.mul _ _ ?idx) ]
              => apply (Tensor.tensor_mul_Proper_dep _ _ R _ _ R _ _ R)
            | [ |- ?R (Classes.div _ _ ?idx) (Classes.div _ _ ?idx) ]
              => apply (Tensor.tensor_div_by_Proper_dep _ _ R _ _ R _ _ R)
            | [ |- ?R (Classes.sqrt _ ?idx) (Classes.sqrt _ ?idx) ]
              => apply (Tensor.tensor_sqrt_Proper_dep _ _ R)
            | [ |- ?R (Classes.opp _ ?idx) (Classes.opp _ ?idx) ]
              => apply (Tensor.tensor_opp_Proper_dep _ _ R)
            | [ |- (_ + _ = _ + _)%core ] => apply f_equal2
            | [ |- (_ * _ = _ * _)%core ] => apply f_equal2
            | [ |- (_ - _ = _ - _)%core ] => apply f_equal2
            | [ |- (_ / _ = _ / _)%core ] => apply f_equal2
            | [ H : Tensor.eqfR ?R ?x ?y |- ?R (?x ?i) (?y ?i) ]
              => apply H
            | [ H : pointwise_relation _ ?R ?f ?g |- ?R (?f _) (?g _) ]
              => apply H
            | [ H : respectful _ ?R ?f ?g |- ?R (?f _) (?g _) ]
              => apply H
            | [ H : Proper _ ?f |- ?R (?f _ _) (?f _ _) ]
              => apply H
            end
          | intro
          | progress cbv [sqr]
          | progress cbv [Dependent.respectful] in *
          | progress cbv [Dependent.lift2_1 Dependent.lift2_2] in *
          | progress cbv [Dependent.lift3_1 Dependent.lift3_2 Dependent.lift3_3] in *
          | progress cbv [Dependent.lift4_1 Dependent.lift4_2 Dependent.lift4_3 Dependent.lift4_4] in *
          | solve [ auto ]
          | match goal with
            | [ |- context[match ?x with _ => _ end] ] => destruct x eqn:?; subst
            | [ H : nth_error ?ls ?n = Some ?v |- _ ]
              => lazymatch goal with
                 | [ H : Proper _ v |- _ ] => fail
                 | _ => idtac
                 end;
                 let H' := fresh in
                 let H'' := fresh in
                 let R := open_constr:(_) in
                 pose proof (_ : Proper (List.Forall2 R) ls) as H';
                 pose proof (List.nth_error_Proper_Forall2 ls ls H' n n eq_refl) as H'';
                 rewrite H in H''; cbv [option_eq] in H'';
                 change (Proper R v) in H''
            end
          | exactly_once multimatch goal with H : _ |- _ => apply H end
          | progress subst ].

  Ltac t := repeat t_step.

  Module Embed.
    Export Embed.

    #[export] Instance forward_Proper_dep {r s d_vocab d_model}
      : Dependent.Proper (Tensor.eqfR ==> Dependent.const Tensor.eqf ==> Tensor.eqfR) (@forward r s d_vocab d_model).
    Proof. cbv [forward]; t. Qed.
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

    Print forward.
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
             ==> Dependent.idR
             ==> Dependent.idR
             ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR ==> Tensor.eqfR)
          (@forward r s d_model).
    Proof. cbv [forward]; t. Qed.
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
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@q r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [q]; repeat first [ t_step | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance q_Proper {A r batch addA zeroA mulA defaultA pos n_heads d_model d_head use_split_qkv_input W_Q b_Q}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@q A r batch addA zeroA mulA defaultA pos n_heads d_model d_head use_split_qkv_input W_Q b_Q).
    Proof. apply q_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance k_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
  : Dependent.Proper
      ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
         ==> Dependent.idR
         ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
         ==> Dependent.idR
         ==> Tensor.eqfR
         ==> Tensor.eqfR
         ==> Tensor.eqfR
         ==> Tensor.eqfR)
      (@k r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [k]; repeat first [ t_step | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance k_Proper {A r batch addA zeroA mulA defaultA pos n_heads d_model d_head use_split_qkv_input W_K b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@k A r batch addA zeroA mulA defaultA pos n_heads d_model d_head use_split_qkv_input W_K b_K).
    Proof. apply k_Proper_dep; repeat intro; subst; reflexivity. Qed.

    #[export] Instance v_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@v r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [v]; repeat first [ t_step | apply einsum_input_Proper_dep ]. Qed.

    #[export] Instance v_Proper {A r batch addA zeroA mulA defaultA pos n_heads d_model d_head use_split_qkv_input W_V b_V}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@v A r batch addA zeroA mulA defaultA pos n_heads d_model d_head use_split_qkv_input W_V b_V).
    Proof. apply v_Proper_dep; repeat intro; subst; reflexivity. Qed.

    Print attn_scores.
    (* attn_scores =
fun (r : Rank) (batch : Shape r) (pos n_heads d_model d_head : ShapeType) (use_split_qkv_input : with_default "use_split_qkv_input" false) (A : Type) (sqrtA : has_sqrt A) (coerZ : has_coer Z A) (addA : has_add A) (zeroA : has_zero A) (mulA : has_mul A) (divA : has_div A) (defaultA : pointed A) (W_Q W_K : tensor [n_heads; d_model; d_head] A)
  (b_Q b_K : tensor [n_heads; d_head] A) =>
let attn_scale := √(coer φ (d_head)) in
let maybe_n_heads := fun b : bool => if b as b0 return (Shape (if b0 then 1%nat else 0%nat)) then [n_heads]%shape else []%shape in
fun query_input key_input : tensor (batch ::' pos ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A =>
let qk := map2' (fun q k : tensor [pos; n_heads; d_head] A => (fun '(tt, i1, i0, i) => ∑_(0≤d_head1<d_head)q.[[i0; i1; d_head1]] * k.[[i; i1; d_head1]]) : tensor [n_heads; pos; pos] A) (q W_Q b_Q query_input) (k W_K b_K key_input) in PArray.checkpoint (qk / broadcast' attn_scale)%core
     : forall (r : Rank) (batch : Shape r) (pos n_heads d_model d_head : ShapeType) (use_split_qkv_input : with_default "use_split_qkv_input" false) (A : Type),
       has_sqrt A ->
       has_coer Z A ->
       has_add A ->
       has_zero A ->
       has_mul A ->
       has_div A ->
       pointed A ->
       tensor [n_heads; d_model; d_head] A ->
       tensor [n_heads; d_model; d_head] A ->
       tensor [n_heads; d_head] A ->
       tensor [n_heads; d_head] A ->
       tensor (batch ::' pos ++' ((fun b : bool => if b as b0 return (Shape (if b0 then 1%nat else 0%nat)) then [n_heads]%shape else []%shape) use_split_qkv_input ::' d_model)) A ->
       tensor (batch ::' pos ++' ((fun b : bool => if b as b0 return (Shape (if b0 then 1%nat else 0%nat)) then [n_heads]%shape else []%shape) use_split_qkv_input ::' d_model)) A -> tensor (batch ::' n_heads ::' pos ::' pos) A

Arguments attn_scores {r}%nat_scope {batch}%shape_scope {pos n_heads d_model d_head}%uint63_scope {use_split_qkv_input} {A}%type_scope {sqrtA coerZ addA zeroA mulA divA defaultA} (W_Q W_K b_Q b_K query_input key_input)%tensor_scope _
*)
    #[export] Instance attn_scores_Proper_dep {r batch pos n_heads d_model d_head use_split_qkv_input}
      : Dependent.Proper
          ((Dependent.idR ==> Dependent.idR)
             ==> (Dependent.const eq ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
             ==> Dependent.idR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@attn_scores r batch pos n_heads d_model d_head use_split_qkv_input).
    Proof. cbv [attn_scores]; repeat first [ t_step | apply q_Proper_dep | apply k_Proper_dep ]. Qed.

    #[export] Instance attn_scores_Proper {r batch pos n_heads d_model d_head use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA defaultA W_Q W_K b_Q b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@attn_scores r batch pos n_heads d_model d_head use_split_qkv_input A sqrtA coerZ addA zeroA mulA divA defaultA W_Q W_K b_Q b_K).
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
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR
             ==> Tensor.eqfR)
          (@masked_attn_scores r batch pos n_heads d_model d_head n_ctx use_split_qkv_input).
    Proof. cbv [masked_attn_scores]; t. Qed.

    About masked_attn_scores.

    #[export] Instance masked_attn_scores_Proper {A r batch sqrtA coerZ addA zeroA mulA divA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K b_Q b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@masked_attn_scores A r batch sqrtA coerZ addA zeroA mulA divA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K b_Q b_K).
    Proof. cbv [masked_attn_scores]; repeat first [ t_step | apply apply_causal_mask_Proper | apply attn_scores_Proper ]. Qed.

    #[export] Instance pattern_Proper {A r batch sqrtA coerZ addA zeroA mulA divA expA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K b_Q b_K}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@pattern A r batch sqrtA coerZ addA zeroA mulA divA expA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K b_Q b_K).
    Proof. cbv [pattern]; repeat first [ t_step | apply masked_attn_scores_Proper ]. Qed.

    #[export] Instance z_Proper {A r batch sqrtA coerZ addA zeroA mulA divA expA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K W_V b_Q b_K b_V}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@z A r batch sqrtA coerZ addA zeroA mulA divA expA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K W_V b_Q b_K b_V).
    Proof.
      cbv [z]; t.
      { apply v_Proper; t. }
      { apply pattern_Proper; t. }
    Qed.

    #[export] Instance attn_out_Proper {A r batch sqrtA coerZ addA zeroA mulA divA expA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K W_V W_O b_Q b_K b_V b_O}
      : Proper (Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf ==> Tensor.eqf) (@attn_out A r batch sqrtA coerZ addA zeroA mulA divA expA defaultA pos n_heads d_model d_head n_ctx use_split_qkv_input W_Q W_K W_V W_O b_Q b_K b_V b_O).
    Proof. cbv [attn_out]; repeat first [ t_step | apply z_Proper ]. Qed.
  End Attention.
  Export (hints) Attention.

  Module TransformerBlock.
    Export TransformerBlock.

    #[export] Instance add_head_dimension_Proper {A r batch pos n_heads d_model use_split_qkv_input}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@add_head_dimension A r batch pos n_heads d_model use_split_qkv_input).
    Proof. cbv [add_head_dimension]; t. Qed.

    #[export] Instance query_input_Proper {A r batch pos n_heads d_model use_split_qkv_input}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@query_input A r batch pos n_heads d_model use_split_qkv_input)
      := _.
    #[export] Instance key_input_Proper {A r batch pos n_heads d_model use_split_qkv_input}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@key_input A r batch pos n_heads d_model use_split_qkv_input)
      := _.
    #[export] Instance value_input_Proper {A r batch pos n_heads d_model use_split_qkv_input}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@value_input A r batch pos n_heads d_model use_split_qkv_input)
      := _.

    #[export] Instance ln1_Proper {A zeroA coerZ addA subA mulA divA sqrtA default d_model normalization_type eps ln1_w ln1_b r s}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@ln1 A zeroA coerZ addA subA mulA divA sqrtA default d_model normalization_type eps ln1_w ln1_b r s).
    Proof. break_innermost_match_hyps; cbv [ln1]; try solve [ t ]; exact _. Qed.
    #[export] Instance ln2_Proper {A zeroA coerZ addA subA mulA divA sqrtA default d_model normalization_type eps ln2_w ln2_b r s}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@ln2 A zeroA coerZ addA subA mulA divA sqrtA default d_model normalization_type eps ln2_w ln2_b r s).
    Proof. break_innermost_match_hyps; cbv [ln2]; try solve [ t ]; exact _. Qed.

    #[export] Instance attn_only_out_Proper {A r batch zeroA coerZ addA subA mulA divA sqrtA expA default pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type W_Q W_K W_V W_O b_Q b_K b_V b_O eps ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@attn_only_out A r batch zeroA coerZ addA subA mulA divA sqrtA expA default pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type W_Q W_K W_V W_O b_Q b_K b_V b_O eps ln1_w ln1_b).
    Proof.
      cbv [attn_only_out]; t.
      all: apply Attention.attn_out_Proper; t.
      all: apply ln1_Proper; t.
      all: first [ apply query_input_Proper | apply key_input_Proper | apply value_input_Proper ]; t.
    Qed.

    #[export] Instance attn_masked_attn_scores_Proper {A r batch zeroA coerZ addA subA mulA divA sqrtA default pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type W_Q W_K b_Q b_K eps ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@HookedTransformer.TransformerBlock.attn_masked_attn_scores A r batch zeroA coerZ addA subA mulA divA sqrtA default pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type W_Q W_K b_Q b_K eps ln1_w ln1_b).
    Proof.
      cbv [HookedTransformer.TransformerBlock.attn_masked_attn_scores]; t.
      all: apply Attention.masked_attn_scores_Proper; t.
      all: apply ln1_Proper; t.
      all: first [ apply query_input_Proper | apply key_input_Proper | apply value_input_Proper ]; t.
    Qed.

    #[export] Instance attn_pattern_Proper {A r batch zeroA coerZ addA subA mulA divA sqrtA expA default pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type W_Q W_K b_Q b_K eps ln1_w ln1_b}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@HookedTransformer.TransformerBlock.attn_pattern A r batch zeroA coerZ addA subA mulA divA sqrtA expA default pos n_heads d_model d_head n_ctx use_split_qkv_input normalization_type W_Q W_K b_Q b_K eps ln1_w ln1_b).
    Proof.
      cbv [HookedTransformer.TransformerBlock.attn_pattern]; t.
      all: apply Attention.pattern_Proper; t.
      all: apply ln1_Proper; t.
      all: first [ apply query_input_Proper | apply key_input_Proper | apply value_input_Proper ]; t.
    Qed.
  End TransformerBlock.
  Export (hints) TransformerBlock.

  Module HookedTransformer.
    Export HookedTransformer.

    #[export] Instance embed_Proper {A d_vocab d_model W_E r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@embed A d_vocab d_model W_E r batch pos)
      := _.
    #[export] Instance pos_embed_Proper {A d_model n_ctx W_pos r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@pos_embed A d_model n_ctx W_pos r batch pos)
      := _.

    #[export] Instance resid_postembed_Proper {A addA default d_vocab d_model n_ctx W_E W_pos r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@resid_postembed A addA default d_vocab d_model n_ctx W_E W_pos r batch pos).
    Proof. cbv [resid_postembed]; t; first [ apply embed_Proper | apply pos_embed_Proper ]; t. Qed.

    #[export] Instance blocks_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos}
      : Proper (List.Forall2 (Tensor.eqf ==> Tensor.eqf))%signature (@blocks A zeroA coerZ addA subA mulA divA sqrtA expA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos).
    Proof.
      cbv [Proper blocks]; clear.
      induction blocks_params as [|?? IH]; cbn [List.map]; constructor; auto; [].
      destruct_head'_prod.
      apply TransformerBlock.attn_only_out_Proper.
    Qed.

    #[export] Instance ln_final_Proper {A zeroA coerZ addA subA mulA divA sqrtA default d_model normalization_type eps ln_final_w ln_final_b r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@ln_final A zeroA coerZ addA subA mulA divA sqrtA default d_model normalization_type eps ln_final_w ln_final_b r batch pos).
    Proof. cbv [ln_final]; break_innermost_match; try solve [ t ]; exact _. Qed.

    #[export] Instance unembed_Proper {A zeroA addA mulA d_vocab_out d_model W_U b_U r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@unembed A zeroA addA mulA d_vocab_out d_model W_U b_U r batch pos)
      := _.

    #[export] Instance blocks_cps_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos T R}
      : Proper (eq ==> Tensor.eqf ==> (Tensor.eqf ==> R) ==> R) (@blocks_cps A zeroA coerZ addA subA mulA divA sqrtA expA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos T).
    Proof.
      cbv [blocks_cps with_default]; intros n ? <-; revert n.
      let blocks := lazymatch goal with |- context[fold_right _ _ (List.firstn _ ?ls)] => ls end in
      pose proof (blocks_Proper : List.Forall2 _ blocks blocks) as H.
      induction blocks as [|?? IH], n; inversion H; clear H; subst; cbn [fold_right List.firstn]; [ now t .. | ].
      repeat intro.
      apply IH; clear IH; t; assumption.
    Qed.

    #[export] Instance logits_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab d_vocab_out n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@logits A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab d_vocab_out n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U r batch pos).
    Proof.
      cbv [logits]; t.
      refine (blocks_cps_Proper (R:=Tensor.eqf) _ _ _ _); t.
      all: first [ apply resid_postembed_Proper | apply embed_Proper | apply pos_embed_Proper | apply unembed_Proper ]; t.
      apply ln_final_Proper; t.
    Qed.

    #[export] Instance forward_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab d_vocab_out n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U r batch pos}
      : Proper (Tensor.eqf ==> Tensor.eqf) (@forward A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab d_vocab_out n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params ln_final_w ln_final_b W_U b_U r batch pos)
      := _.

    #[export] Instance blocks_attn_masked_attn_scores_Proper {A zeroA coerZ addA subA mulA divA sqrtA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos}
      : Proper (List.Forall2 (Tensor.eqf ==> Tensor.eqf))%signature (@HookedTransformer.HookedTransformer.blocks_attn_masked_attn_scores A zeroA coerZ addA subA mulA divA sqrtA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos).
    Proof.
      cbv [Proper HookedTransformer.HookedTransformer.blocks_attn_masked_attn_scores]; clear.
      induction blocks_params as [|?? IH]; cbn [List.map]; constructor; auto; [].
      destruct_head'_prod.
      apply TransformerBlock.attn_masked_attn_scores_Proper.
    Qed.

    #[export] Instance blocks_attn_pattern_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos}
      : Proper (List.Forall2 (Tensor.eqf ==> Tensor.eqf))%signature (@HookedTransformer.HookedTransformer.blocks_attn_pattern A zeroA coerZ addA subA mulA divA sqrtA expA default n_heads d_model d_head n_ctx normalization_type eps blocks_params r batch pos).
    Proof.
      cbv [Proper HookedTransformer.HookedTransformer.blocks_attn_pattern]; clear.
      induction blocks_params as [|?? IH]; cbn [List.map]; constructor; auto; [].
      destruct_head'_prod.
      apply TransformerBlock.attn_pattern_Proper.
    Qed.

    #[export] Instance masked_attn_scores_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params r batch pos}
      : Proper (eq ==> Tensor.eqf ==> option_eq Tensor.eqf) (@HookedTransformer.HookedTransformer.masked_attn_scores A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params r batch pos).
    Proof.
      intros n ? <-.
      cbv [HookedTransformer.HookedTransformer.masked_attn_scores]; t; try now constructor.
      refine (blocks_cps_Proper (R:=Tensor.eqf) _ _ _ _); t.
      apply resid_postembed_Proper; t.
    Qed.

    #[export] Instance attn_pattern_Proper {A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params r batch pos}
      : Proper (eq ==> Tensor.eqf ==> option_eq Tensor.eqf) (@HookedTransformer.HookedTransformer.attn_pattern A zeroA coerZ addA subA mulA divA sqrtA expA default d_vocab n_heads d_model d_head n_ctx normalization_type eps W_E W_pos blocks_params r batch pos).
    Proof.
      intros n ? <-.
      cbv [HookedTransformer.HookedTransformer.attn_pattern]; t; try now constructor.
      refine (blocks_cps_Proper (R:=Tensor.eqf) _ _ _ _); t.
      apply resid_postembed_Proper; t.
    Qed.
  End HookedTransformer.
  Export (hints) HookedTransformer.
End HookedTransformer.
Export (hints) HookedTransformer.
