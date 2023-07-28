From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
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

(** Coq infra *)
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.

Notation tensor_of_list ls := (Tensor.PArray.abstract (Tensor.PArray.concretize (Tensor.of_list ls))) (only parsing).

(*#[local] Notation FLOAT := float (only parsing). (* or Q *)*)

Module Embed.
  Section __.
    Context {A d_vocab d_model}
      (W_E : tensor A [d_vocab; d_model]).

    Definition forward {r} {s : Shape r} (tokens : tensor IndexType s) : tensor A (s ::' d_model)
      := (W_E.[tokens, :])%fancy_raw_tensor.
  End __.
End Embed.

Module Unembed.
  Section __.
    Context {A} {addA : has_add A} {mulA : has_mul A} {zeroA : has_zero A}
      {d_model d_vocab_out}
      (W_U : tensor A [d_model; d_vocab_out])
      (b_U : tensor A [d_vocab_out]).

    Definition forward {r} {batch_pos : Shape r} (residual : tensor A (batch_pos ::' d_model))
      : tensor A (batch_pos ::' d_vocab_out)
      := (Tensor.map'
            (fun residual : tensor A [d_model]
             => weaksauce_einsum {{{ {{ d_model, d_model vocab -> vocab }}
                        , residual
                        , W_U }}}
               : tensor A [d_vocab_out])
            residual
          + broadcast b_U)%core.
  End __.
End Unembed.

Module PosEmbed.
  Section __.
    Context {A d_model}
      {n_ctx:N}
      (n_ctx' : int := n_ctx)
      (W_pos : tensor A [n_ctx'; d_model]).

    Definition forward {r} {batch : Shape r} {tokens_length} (tokens : tensor IndexType (batch ::' tokens_length))
      : tensor A (batch ::' tokens_length ::' d_model)
      := repeat (W_pos.[:tokens_length, :]) batch.
  End __.
End PosEmbed.

Module LayerNorm.
  Section __.
    Context {r A} {s : Shape r} {d_model}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A} {default : pointed A}
      (eps : A) (w b : tensor A [d_model]).

    Definition linpart (x : tensor A (s ::' d_model))
      : tensor A (s ::' d_model)
      := (x - reduce_axis_m1 (keepdim:=true) mean x)%core.

    Definition scale (x : tensor A (s ::' d_model))
      : tensor A (s ::' 1)
      := (√(reduce_axis_m1 (keepdim:=true) mean (x ²) + broadcast' eps))%core.

    Definition rescale (x : tensor A (s ::' d_model))
      (scale : tensor A (s ::' 1))
      : tensor A (s ::' d_model)
      := (x / scale)%core.

    Definition postrescale (x : tensor A (s ::' d_model))
      : tensor A (s ::' d_model)
      := (x * broadcast w + broadcast b)%core.

    Definition forward (x : tensor A (s ::' d_model))
      : tensor A (s ::' d_model)
      := let x := PArray.checkpoint (linpart x) in
         let scale := scale x in
         let x := rescale x scale in
         PArray.checkpoint (postrescale x).
  End __.
End LayerNorm.

Module Attention.
  Section __.
    Context {A r} {batch : Shape r}
      {sqrtA : has_sqrt A} {coerZ : has_coer Z A} {addA : has_add A} {zeroA : has_zero A} {mulA : has_mul A} {divA : has_div A} {expA : has_exp A} {defaultA : pointed A}
      {pos n_heads d_model d_head} {n_ctx:N}
      {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
      (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
      (b_Q b_K b_V : tensor A [n_heads; d_head])
      (b_O : tensor A [d_model])
      (IGNORE : A := coerZ (-1 * 10 ^ 5)%Z)
      (attn_scale : A := √(coer (Uint63.to_Z d_head)))
      (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape)
      (query_input key_input value_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
      (mask : tensor bool [n_ctx; n_ctx] := to_bool (tril (A:=bool) (ones [n_ctx; n_ctx]))).

    (*         if self.cfg.use_split_qkv_input:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        q = self.hook_q(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]*)
    Definition einsum_input
      (input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
      (W : tensor A [n_heads; d_model; d_head])
      : tensor A ((batch ::' pos) ++' [n_heads; d_head])
      := Tensor.map'
           (if use_split_qkv_input return tensor A (maybe_n_heads use_split_qkv_input ::' d_model) -> tensor A [n_heads; d_head]
            then fun input => weaksauce_einsum {{{ {{ head_index d_model,
                                          head_index d_model d_head
                                          -> head_index d_head }}
                                      , input
                                      , W }}}
            else fun input => weaksauce_einsum {{{ {{ d_model,
                                          head_index d_model d_head
                                          -> head_index d_head }}
                                      , input
                                      , W }}})
           input.

    Definition q : tensor A (batch ++' [pos; n_heads; d_head])
      := PArray.checkpoint (einsum_input query_input W_Q + broadcast b_Q)%core.
    Definition k : tensor A (batch ++' [pos; n_heads; d_head])
      := PArray.checkpoint (einsum_input key_input W_K + broadcast b_K)%core.
    Definition v : tensor A (batch ++' [pos; n_heads; d_head])
      := PArray.checkpoint (einsum_input value_input W_V + broadcast b_V)%core.

    Definition attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
      := (let qk : tensor A (batch ++' [n_heads; pos; pos])
            := Tensor.map2'
                 (fun q k : tensor A [pos; n_heads; d_head]
                  => weaksauce_einsum
                       {{{ {{ query_pos head_index d_head ,
                                 key_pos head_index d_head
                                 -> head_index query_pos key_pos }}
                             , q
                             , k }}}
                    : tensor A [n_heads; pos; pos])
                 q
                 k in
          PArray.checkpoint (qk / broadcast' attn_scale))%core.

    Definition apply_causal_mask (attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos))
      : tensor A (batch ::' n_heads ::' pos ::' pos)
      := Tensor.map'
           (fun attn_scores : tensor A [pos; pos]
            => Tensor.where_ mask.[:pos,:pos] attn_scores (broadcast' IGNORE))
           attn_scores.

    Definition masked_attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
      := apply_causal_mask attn_scores.

    Definition pattern : tensor A (batch ::' n_heads ::' pos ::' pos)
      := PArray.checkpoint (softmax_dim_m1 masked_attn_scores).

    Definition z : tensor A (batch ::' pos ::' n_heads ::' d_head)
      := PArray.checkpoint
           (Tensor.map2'
              (fun (v : tensor A [pos; n_heads; d_head])
                   (pattern : tensor A [n_heads; pos; pos])
               => weaksauce_einsum {{{ {{  key_pos head_index d_head,
                              head_index query_pos key_pos ->
                              query_pos head_index d_head }}
                          , v
                          , pattern }}}
                 : tensor A [pos; n_heads; d_head])
              v
              pattern).

    Definition attn_out : tensor A (batch ::' pos ::' d_model)
      := (let out
            := Tensor.map'
                 (fun z : tensor A [pos; n_heads; d_head]
                  => weaksauce_einsum {{{ {{ pos head_index d_head,
                                 head_index d_head d_model ->
                                 pos d_model }}
                             , z
                             , W_O }}}
                    : tensor A [pos; d_model])
                 z in
          PArray.checkpoint (out + broadcast b_O))%core.
  End __.
End Attention.

Variant NormalizationType := LN (*| LNPre*).

Module TransformerBlock.
  Section __.
    Context {A r} {batch : Shape r}
      {zeroA : has_zero A} {coerZ : has_coer Z A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {sqrtA : has_sqrt A} {expA : has_exp A}
      {default : pointed A}
      {pos n_heads d_model d_head} {n_ctx:N}
      {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
      {normalization_type : with_default "normalization_type" (option NormalizationType) (Some LN)}
      (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
      (b_Q b_K b_V : tensor A [n_heads; d_head])
      (b_O : tensor A [d_model])
      (eps : A)
      (ln1_w ln1_b ln2_w ln2_b : match normalization_type with
                                 | Some LN => tensor A [d_model]
                                 | Datatypes.None => with_default "()" True I
                                 end)
      (resid_pre : tensor A ((batch ::' pos) ++' [d_model]))
      (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape).

    Definition add_head_dimension
      (resid_pre : tensor A ((batch ::' pos) ++' [d_model]))
      : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
      := if use_split_qkv_input return tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
         then Tensor.map'
                (fun resid_pre : tensor A [d_model]
                 => Tensor.repeat resid_pre [n_heads]
                   : tensor A [n_heads; d_model])
                resid_pre
         else resid_pre.
    Definition query_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
      := add_head_dimension resid_pre.
    Definition key_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
      := add_head_dimension resid_pre.
    Definition value_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
      := add_head_dimension resid_pre.

    #[local] Notation LayerNorm_forward
      := (match normalization_type
                return match normalization_type with Some LN => _ | _ => _ end
                       -> match normalization_type with Some LN => _ | _ => _ end
                       -> _
          with
          | Some LN => LayerNorm.forward eps
          | Datatypes.None => fun _ _ => PArray.checkpoint
          end)
           (only parsing).

    Definition ln1 {r} {s : Shape r} (t : tensor A (s ::' d_model)) : tensor A (s ::' d_model)
      := LayerNorm_forward ln1_w ln1_b t.
    Definition ln2 {r} {s : Shape r} (t : tensor A (s ::' d_model)) : tensor A (s ::' d_model)
      := LayerNorm_forward ln2_w ln2_b t.

    Definition attn_only_out : tensor A (batch ++ [pos; d_model])
      := (let attn_out : tensor A (batch ++ [pos; d_model])
            := Attention.attn_out
                 (n_ctx:=n_ctx)
                 W_Q W_K W_V W_O
                 b_Q b_K b_V b_O
                 (ln1 query_input)
                 (ln1 key_input)
                 (ln1 value_input) in
          resid_pre + attn_out)%core.

    (** convenience *)
    Local Definition attn_masked_attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
      := Attention.masked_attn_scores
           (n_ctx:=n_ctx)
           W_Q W_K
           b_Q b_K
           (ln1 query_input)
           (ln1 key_input).

    Local Definition attn_pattern : tensor A (batch ::' n_heads ::' pos ::' pos)
      := Attention.pattern
           (n_ctx:=n_ctx)
           W_Q W_K
           b_Q b_K
           (ln1 query_input)
           (ln1 key_input).
  End __.
End TransformerBlock.

Module HookedTransformer.
  Section __.
    Context {A}
      {zeroA : has_zero A} {coerZ : has_coer Z A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {sqrtA : has_sqrt A} {expA : has_exp A}
      {default : pointed A}
      {d_vocab d_vocab_out n_heads d_model d_head} {n_ctx:N}
      (*{use_split_qkv_input : with_default "use_split_qkv_input" bool false}*)
      {normalization_type : with_default "normalization_type" (option NormalizationType) (Some LN)}
      (eps : A)

      (W_E : tensor A [d_vocab; d_model])
      (W_pos : tensor A [n_ctx; d_model])

      (blocks_params
        : list ((* (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
      (b_Q b_K b_V : tensor A [n_heads; d_head])
      (b_O : tensor A [d_model])
      (ln1_w ln1_b : tensor A [d_model]) *)
              tensor A [n_heads; d_model; d_head] * tensor A [n_heads; d_model; d_head] * tensor A [n_heads; d_model; d_head] * tensor A [n_heads; d_model; d_head]
              * tensor A [n_heads; d_head] * tensor A [n_heads; d_head] * tensor A [n_heads; d_head]
              * tensor A [d_model]
              * match normalization_type with
                | Some LN => tensor A [d_model]
                | Datatypes.None => with_default "()" True I
                end
              * match normalization_type with
                | Some LN => tensor A [d_model]
                | Datatypes.None => with_default "()" True I
                end))

      (ln_final_w ln_final_b
        : match normalization_type with
          | Some LN => tensor A [d_model]
          | Datatypes.None => with_default "()" True I
          end)

      (W_U : tensor A [d_model; d_vocab_out]) (b_U : tensor A [d_vocab_out])
    .

    Section with_batch.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' d_model)%shape).

      Definition embed (tokens : tensor IndexType s) : tensor A resid_shape
        := Embed.forward W_E tokens.

      Definition pos_embed (tokens : tensor IndexType s) : tensor A resid_shape
        := PosEmbed.forward W_pos tokens.

      Definition blocks : list (tensor A resid_shape -> tensor A resid_shape)
        := List.map
             (fun '(W_Q, W_K, W_V, W_O,
                    b_Q, b_K, b_V,
                    b_O,
                    ln1_w, ln1_b)
              => TransformerBlock.attn_only_out
                   (n_ctx:=n_ctx)
                   W_Q W_K W_V W_O
                   b_Q b_K b_V
                   b_O
                   eps
                   ln1_w ln1_b)
             blocks_params.

    #[local] Notation LayerNorm_forward
      := (match normalization_type
                return match normalization_type with Some LN => _ | _ => _ end
                       -> match normalization_type with Some LN => _ | _ => _ end
                       -> _
          with
          | Some LN => LayerNorm.forward eps
          | Datatypes.None => fun _ _ t => t
          end)
           (only parsing).

      Definition ln_final (resid : tensor A resid_shape) : tensor A resid_shape
        := LayerNorm_forward ln_final_w ln_final_b resid.

      Definition unembed (resid : tensor A resid_shape) : tensor A (s ::' d_vocab_out)
        := Unembed.forward W_U b_U resid.

      Definition blocks_cps {T} {n : with_default "blocks n" nat (List.length blocks)} (residual : tensor A resid_shape) (K : tensor A resid_shape -> T) : T
        := List.fold_right
             (fun block cont residual
              => let residual := PArray.checkpoint (block residual) in
                 cont residual)
             K
             (List.firstn n blocks)
             residual.

      Definition resid_postembed (tokens : tensor IndexType s) : tensor A resid_shape
        := (let embed          := embed tokens in
            let pos_embed      := pos_embed tokens in
            PArray.checkpoint (embed + pos_embed)%core).

      Definition logits (tokens : tensor IndexType s) : tensor A (s ::' d_vocab_out)
        := (let residual       := resid_postembed tokens in
            blocks_cps
              residual
              (fun residual
               => let residual := PArray.checkpoint (ln_final residual) in
                  let logits   := PArray.checkpoint (unembed residual) in
                  logits)).

      Definition forward (tokens : tensor IndexType s) : tensor A (s ::' d_vocab_out)
        := logits tokens.

      (** convenience *)
      Local Definition blocks_attn_masked_attn_scores
        : list (tensor A resid_shape -> tensor A (batch ::' n_heads ::' pos ::' pos))
        := List.map
             (fun '(W_Q, W_K, W_V, W_O,
                    b_Q, b_K, b_V,
                    b_O,
                    ln1_w, ln1_b)
              => TransformerBlock.attn_masked_attn_scores
                   (n_ctx:=n_ctx)
                   W_Q W_K
                   b_Q b_K
                   eps
                   ln1_w ln1_b)
             blocks_params.

      Local Definition blocks_attn_pattern
        : list (tensor A resid_shape -> tensor A (batch ::' n_heads ::' pos ::' pos))
        := List.map
             (fun '(W_Q, W_K, W_V, W_O,
                    b_Q, b_K, b_V,
                    b_O,
                    ln1_w, ln1_b)
              => TransformerBlock.attn_pattern
                   (n_ctx:=n_ctx)
                   W_Q W_K
                   b_Q b_K
                   eps
                   ln1_w ln1_b)
             blocks_params.

      Local Definition masked_attn_scores (n : nat) (tokens : tensor IndexType s)
        : option (tensor A (batch ::' n_heads ::' pos ::' pos))
        := match List.nth_error blocks_attn_masked_attn_scores n with
           | Some block_n_attn_masked_attn_scores
             => Some (let residual       := resid_postembed tokens in
                      blocks_cps
                        (n:=Nat.pred n)
                        residual
                        (fun residual
                         => PArray.checkpoint (block_n_attn_masked_attn_scores residual)))
           | None => None
           end.

      Local Definition attn_pattern (n : nat) (tokens : tensor IndexType s)
        : option (tensor A (batch ::' n_heads ::' pos ::' pos))
        := match List.nth_error blocks_attn_pattern n with
           | Some block_n_attn_pattern
             => Some (let residual       := resid_postembed tokens in
                      blocks_cps
                        (n:=Nat.pred n)
                        residual
                        (fun residual
                         => PArray.checkpoint (block_n_attn_pattern residual)))
           | None => None
           end.
    End with_batch.
  End __.
End HookedTransformer.
