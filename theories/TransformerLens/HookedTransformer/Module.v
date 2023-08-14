From Coq Require Vector.
From Coq.Structures Require Import Equalities.
From Coq Require Import Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Export Default Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Config.
Import Instances.Truncating.
#[local] Open Scope core_scope.

#[local] Coercion Vector.of_list : list >-> Vector.t.

Notation tensor_of_list ls := (tensor_of_list ls) (only parsing).

Module Model (cfg : Config).
  Import (hints) cfg.

  Definition all_tokens {use_checkpoint : with_default "use_checkpoint" bool true}
    : tensor [(cfg.d_vocab ^ cfg.n_ctx)%core : N; cfg.n_ctx] RawIndexType
    := let all_toks := Tensor.arange (start:=0) (Uint63.of_Z cfg.d_vocab) in
       let all_tokens := Tensor.cartesian_exp all_toks cfg.n_ctx in
       (if use_checkpoint then PArray.checkpoint else fun x => x) all_tokens.

  Module Embed.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A}.

      Definition forward (tokens : tensor s IndexType) : tensor resid_shape A
        := Embed.forward (A:=A) cfg.W_E tokens.
    End __.
  End Embed.

  Module Unembed.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A}
        {addA : has_add A} {mulA : has_mul A}.

      Definition forward (residual : tensor resid_shape A) : tensor (s ::' cfg.d_vocab_out) A
        := Unembed.forward (A:=A) cfg.W_U cfg.b_U residual.
    End __.
  End Unembed.

  Module PosEmbed.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A}.

      Definition forward (tokens : tensor s IndexType) : tensor resid_shape A
        := PosEmbed.forward (A:=A) cfg.W_pos tokens.
    End __.
  End PosEmbed.

(*  Module LayerNorm.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
        {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
        {sqrtA : has_sqrt A}
        {use_checkpoint : with_default "use_checkpoint" bool true}.
      Let coerA' (x : float) : A := coer x.
      #[local] Coercion coerA' : float >-> A.

      Definition final_linpart : tensor resid_shape A -> tensor resid_shape A
        := match cfg.normalization_type with
           | Some LN => LayerNorm.linpart (A:=A)
           | Datatypes.None => fun x => x
           end.

      Definition final_scale : tensor resid_shape A -> tensor (s ::  A
        := match cfg.normalization_type with
           | Some LN => LayerNorm.scale (A:=A) cfg.eps
           | Datatypes.None => fun x => x
           end.


    Definition scale (x : tensor (s ::' d_model) A)
      : tensor (s ::' 1) A
      := (√(reduce_axis_m1 (keepdim:=true) mean (x ²) + broadcast' eps))%core.

    Definition rescale (x : tensor (s ::' d_model) A)
      (scale : tensor (s ::' 1) A)
      : tensor (s ::' d_model) A
      := (x / scale)%core.

    Definition postrescale (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A
      := (x * broadcast w + broadcast b)%core.

    Definition forward (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A
      := let x := PArray.checkpoint (linpart x) in
         let scale := scale x in
         let x := rescale x scale in
         checkpoint (postrescale x).
  End __.
End LayerNorm.
*)
  Module Attention.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
        {addA : has_add A} {mulA : has_mul A} {divA : has_div A}
        {sqrtA : has_sqrt A} {expA : has_exp A}
        {use_checkpoint : with_default "use_checkpoint" bool true}
        (query_input key_input value_input : tensor resid_shape A).

      Definition qs : Vector.t (tensor (s ++' [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.q (A:=A) W_Q b_Q query_input)
             cfg.blocks_params.
      Definition ks : Vector.t (tensor (s ++' [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.k (A:=A) W_K b_K key_input)
             cfg.blocks_params.
      Definition vs : Vector.t (tensor (s ++' [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.v (A:=A) W_V b_V value_input)
             cfg.blocks_params.

      Definition attn_scores : Vector.t (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.attn_scores (A:=A) W_Q W_K b_Q b_K query_input key_input)
             cfg.blocks_params.

      Definition masked_attn_scores : Vector.t (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.masked_attn_scores (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K query_input key_input)
             cfg.blocks_params.

      Definition patterns : Vector.t (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.pattern (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K query_input key_input)
             cfg.blocks_params.

      Definition zs : Vector.t (tensor (s ++ [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.z (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K W_V b_Q b_K b_V query_input key_input value_input)
             cfg.blocks_params.

      Definition attn_outs : Vector.t (tensor resid_shape A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => Attention.attn_out (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K W_V W_O b_Q b_K b_V b_O query_input key_input value_input)
             cfg.blocks_params.
    End __.
  End Attention.

  Module TransformerBlock.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
        {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
        {sqrtA : has_sqrt A} {expA : has_exp A}
        {use_checkpoint : with_default "use_checkpoint" bool true}
        (resid_pre : tensor resid_shape A).
      Let coerA' (x : float) : A := coer x.
      #[local] Coercion coerA' : float >-> A.

      Let coer_ln_tensor : cfg.ln_tensor float -> cfg.ln_tensor A
          := match cfg.normalization_type as nt return cfg.ln_tensor_gen nt float -> cfg.ln_tensor_gen nt A with
             | Some LN
             | Datatypes.None
               => fun x => x
             end.

      Definition attn_only_outs : Vector.t (tensor resid_shape A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => TransformerBlock.attn_only_out (A:=A) (normalization_type:=cfg.normalization_type) (n_ctx:=cfg.n_ctx) W_Q W_K W_V W_O b_Q b_K b_V b_O cfg.eps (coer_ln_tensor ln1_w) (coer_ln_tensor ln1_b) resid_pre)
             cfg.blocks_params.

      Definition attn_masked_attn_scores : Vector.t (tensor (s ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => HookedTransformer.TransformerBlock.attn_masked_attn_scores (A:=A) (normalization_type:=cfg.normalization_type) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K  cfg.eps (coer_ln_tensor ln1_w) (coer_ln_tensor ln1_b) resid_pre)
             cfg.blocks_params.

      Definition attn_patterns : Vector.t (tensor (s ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => HookedTransformer.TransformerBlock.attn_pattern (A:=A) (normalization_type:=cfg.normalization_type) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K  cfg.eps (coer_ln_tensor ln1_w) (coer_ln_tensor ln1_b) resid_pre)
             cfg.blocks_params.
    End __.
  End TransformerBlock.

  Module HookedTransformer.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
        {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
        {sqrtA : has_sqrt A} {expA : has_exp A}
        {use_checkpoint : with_default "use_checkpoint" bool true}.
      Let coerA' (x : float) : A := coer x.
      #[local] Coercion coerA' : float >-> A.

      Let coer_ln_tensor : cfg.ln_tensor float -> cfg.ln_tensor A
          := match cfg.normalization_type as nt return cfg.ln_tensor_gen nt float -> cfg.ln_tensor_gen nt A with
             | Some LN
             | Datatypes.None
               => fun x => x
             end.

      Definition embed (tokens : tensor s IndexType) : tensor resid_shape A
        := HookedTransformer.embed (A:=A) cfg.W_E tokens.
      Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape A
        := HookedTransformer.pos_embed (A:=A) cfg.W_pos tokens.
      Definition coer_blocks_params
        := List.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => ((W_Q:tensor _ A), (W_K:tensor _ A), (W_V:tensor _ A), (W_O:tensor _ A),
                   (b_Q:tensor _ A), (b_K:tensor _ A), (b_V:tensor _ A), (b_O:tensor _ A),
                   coer_ln_tensor ln1_w, coer_ln_tensor ln1_b)).
      Definition blocks : list (tensor resid_shape A -> tensor resid_shape A)
        := HookedTransformer.blocks
             (A:=A) (normalization_type:=cfg.normalization_type) (n_ctx:=cfg.n_ctx) cfg.eps (coer_blocks_params cfg.blocks_params).

      Definition ln_final (resid : tensor resid_shape A) : tensor resid_shape A
        := HookedTransformer.ln_final (A:=A) cfg.eps (coer_ln_tensor cfg.ln_final_w) (coer_ln_tensor cfg.ln_final_b) resid.

      Definition unembed (resid : tensor resid_shape A) : tensor (s ::' cfg.d_vocab_out) A
        := HookedTransformer.unembed (A:=A) cfg.W_U cfg.b_U resid.

      Polymorphic Definition blocks_cps {T} {n} (residual : tensor resid_shape A) (K : tensor resid_shape A -> T) : T
        := HookedTransformer.blocks_cps (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type) cfg.eps (coer_blocks_params cfg.blocks_params) (n:=n) residual K.

      Definition resid_postembed (tokens : tensor s IndexType) : tensor resid_shape A
        := HookedTransformer.resid_postembed (A:=A) cfg.W_E cfg.W_pos tokens.

      Definition logits (tokens : tensor s IndexType) : tensor (s ::' cfg.d_vocab_out) A
        := HookedTransformer.logits
             (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type) cfg.eps
             cfg.W_E cfg.W_pos
             (coer_blocks_params cfg.blocks_params)
             (coer_ln_tensor cfg.ln_final_w) (coer_ln_tensor cfg.ln_final_b)
             cfg.W_U cfg.b_U
             tokens.

      Definition forward (tokens : tensor s IndexType) : tensor (s ::' cfg.d_vocab_out) A
        := HookedTransformer.forward
             (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type) cfg.eps
             cfg.W_E cfg.W_pos
             (coer_blocks_params cfg.blocks_params)
             (coer_ln_tensor cfg.ln_final_w) (coer_ln_tensor cfg.ln_final_b)
             cfg.W_U cfg.b_U
             tokens.

      Local Definition masked_attn_scores (n : nat) (tokens : tensor s IndexType)
        : option (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A)
        := HookedTransformer.HookedTransformer.masked_attn_scores
             (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type)cfg.eps
             cfg.W_E cfg.W_pos
             (coer_blocks_params cfg.blocks_params)
             n tokens.

      Local Definition attn_pattern (n : nat) (tokens : tensor s IndexType)
        : option (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A)
        := HookedTransformer.HookedTransformer.attn_pattern
             (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type)cfg.eps
             cfg.W_E cfg.W_pos
             (coer_blocks_params cfg.blocks_params)
             n tokens.
    End __.
  End HookedTransformer.

  Notation model := HookedTransformer.logits (only parsing).

  Definition logits_all_tokens : tensor _ float
    := HookedTransformer.logits all_tokens.
End Model.

Module Type ModelSig (cfg : Config) := Nop <+ Model cfg.
