(** This file corresponds roughly to components.py from
    transformer_lens at
    https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/components.py

    In this file, each class in components.py is turned into a Coq
    Module purely for namespacing / code organizational purposes.

    - In this file, NN configuration parameters are taken as arguments
      to each function.

    - In [HookedTransformer/Module.v], we organize the same building
      blocks into module functors where the configuration parameters
      are taken in as module functor arguments, using the code defined
      in this file.

    - This file allows the potential for proving theorems about the
      functions that are universally quantified over all parameters,
      while the module functor organization allows the potential for
      avoiding the overhead of passing around the parameters to every
      function call (hopefully, I'm still figuring out ideal design
      choices here).

    - It might be better to combine the files and stick with just the
      module functor based organization, since that's the one we
      ultimately use.
 *)
(** Design principles:

    Two goals for code in this file:

    1. Be able to run the code (efficiently, on primitive floats and
       arrays)

    2. Be able to reuse the same code to prove theorems about the
       computation (using, e.g., reals or rationals instead of
       primitive floats, so we get more nice mathematical properties).

    As a result, we want to:

    - parameterize each function over the type of data (and the
      associated operations on that datatype that we need)

    - order the arguments so that we can define [Proper] instances
      relating instantiations on different datatypes; arguments that
      are the same across instantiations (such as vocab size) come
      first, while arguments that vary (such as how to do addition on
      the datatype) come later.

    Additionally, we parameterize functions over the batch size and
    the shape of the tensor.

    Ordering of the particular datatype operations is currently a bit
    ad-hoc and disorganized, though some effort is made to maintain
    consistency across components.

    In each component, arguments are specified with a [Context]
    directive in an unnamed [Section] to allow sharing of argument
    declarations between different functions within that component.
    Coq automatically determines the subset of arguments used for each
    function.

    - Most arguments are implicit and maximally inserted (using curly
      braces [{}]) so that they are picked up by type inference or
      typeclass resolution.  Exceptions are the weights and biases and
      the tensors that are passed into the python code.


    - We pass a [use_checkpoint : with_default "use_checkpoint" bool
      true] argument which specifies whether or not we materialize
      concrete arrays at various points in the code.

      - Because our tensors are represented as functions from indices
        to data, by default computation is lazy and deferred until
        concrete indices are given.

      - This is useful to avoid duplicating computation when
        broadcasting scalars, but would involve massive duplication in
        other cases such as recomputing the same mean repeatedly for
        every index separately.

      - Hence we use [PArray.maybe_checkpoint] to materialize
        computations before any step that might duplicate them.

      - Materializing arrays gets in the way of proofs, so we use this
        parameter to ease infrastructure for removing all array
        materialization simultaneously in a single proof step.  *)

From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool PrimitiveProd.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.TransformerLens Require Export HookedTransformer.Config.
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
Local Open Scope core_scope.

Notation tensor_of_list ls := (Tensor.PArray.abstract (Tensor.PArray.concretize (Tensor.of_list ls))) (only parsing).

(*#[local] Notation FLOAT := float (only parsing). (* or Q *)*)

Module Embed.
  Section __.
    Context {r} {s : Shape r} {d_vocab d_model A}
      (W_E : tensor [d_vocab; d_model] A).

    Definition forward (tokens : tensor s IndexType) : tensor (s ::' d_model) A
      := (W_E.[tokens, :])%fancy_raw_tensor.
  End __.
End Embed.

Module Unembed.
  Section __.
    Context
      {r} {batch_pos : Shape r}
      {d_model d_vocab_out}
      {A} {addA : has_add A} {mulA : has_mul A} {zeroA : has_zero A}
      (W_U : tensor [d_model; d_vocab_out] A)
      (b_U : tensor [d_vocab_out] A).

    Definition forward (residual : tensor (batch_pos ::' d_model) A)
      : tensor (batch_pos ::' d_vocab_out) A
      := (Tensor.map'
            (fun residual : tensor [d_model] A
             => weaksauce_einsum {{{ {{ d_model, d_model vocab -> vocab }}
                        , residual
                        , W_U }}}
               : tensor [d_vocab_out] A)
            residual
          + broadcast b_U)%core.
  End __.
End Unembed.

Module PosEmbed.
  Section __.
    Context {r} {batch : Shape r} {tokens_length : ShapeType}
      {d_model} {n_ctx:N}
      (n_ctx' : int := n_ctx)
      {A}
      (W_pos : tensor [n_ctx'; d_model] A).

    Definition forward (tokens : tensor (batch ::' tokens_length) IndexType)
      : tensor (batch ::' tokens_length ::' d_model) A
      := repeat batch (W_pos.[:tokens_length, :]).
  End __.
End PosEmbed.

Module LayerNorm.
  Section __.
    Context {r} {s : Shape r} {d_model}
      {A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A}
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (defaultA : pointed A := @coer _ _ coerZ point)
      (eps : A) (w b : tensor [d_model] A).
    #[local] Existing Instance defaultA.

    Definition linpart (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A
      := let xbar := reduce_axis_m1 (keepdim:=true) mean x in
         PArray.maybe_checkpoint (x - xbar)%core.

    Definition scale (x : tensor (s ::' d_model) A)
      : tensor (s ::' 1) A
      := PArray.maybe_checkpoint (√(reduce_axis_m1 (keepdim:=true) mean (x ²) + broadcast' eps))%core.

    Definition rescale (x : tensor (s ::' d_model) A)
      (scale : tensor (s ::' 1) A)
      : tensor (s ::' d_model) A
      := (x / scale)%core.

    Definition postrescale (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A
      := (x * broadcast w + broadcast b)%core.

    Definition forward (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A
      := let x := linpart x in
         let scale := scale x in
         let x := rescale x scale in
         PArray.maybe_checkpoint (postrescale x).
  End __.
End LayerNorm.

Module Attention.
  Section __.
    Context {r} {batch : Shape r}
      {pos n_heads d_model d_head} {n_ctx:N}
      {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
      {A}
      {zeroA : has_zero A} {coerZ : has_coer Z A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {maxA : has_max A}
      {sqrtA : has_sqrt A} {expA : has_exp A}
      (defaultA : pointed A := @coer _ _ coerZ point)
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (W_Q W_K W_V W_O : tensor [n_heads; d_model; d_head] A)
      (b_Q b_K b_V : tensor [n_heads; d_head] A)
      (b_O : tensor [d_model] A)
      (IGNORE : A := coerZ (-1 * 10 ^ 5)%Z)
      (attn_scale : A := √(coer (Uint63.to_Z d_head)))
      (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape)
      (query_input key_input value_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A)
      (mask : tensor [n_ctx; n_ctx] bool := to_bool (tril (A:=bool) (ones [n_ctx; n_ctx]))).

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
      (input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A)
      (W : tensor [n_heads; d_model; d_head] A)
      : tensor ((batch ::' pos) ++' [n_heads; d_head]) A
      := Tensor.map'
           (if use_split_qkv_input return tensor (maybe_n_heads use_split_qkv_input ::' d_model) A -> tensor [n_heads; d_head] A
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
    #[local] Existing Instance defaultA.

    Definition q : tensor (batch ++' [pos; n_heads; d_head]) A
      := PArray.maybe_checkpoint (einsum_input query_input W_Q + broadcast b_Q)%core.
    Definition k : tensor (batch ++' [pos; n_heads; d_head]) A
      := PArray.maybe_checkpoint (einsum_input key_input W_K + broadcast b_K)%core.
    Definition v : tensor (batch ++' [pos; n_heads; d_head]) A
      := PArray.maybe_checkpoint (einsum_input value_input W_V + broadcast b_V)%core.

    Definition attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A
      := (let qk : tensor (batch ++' [n_heads; pos; pos]) A
            := Tensor.map2'
                 (fun q k : tensor [pos; n_heads; d_head] A
                  => weaksauce_einsum
                       {{{ {{ query_pos head_index d_head ,
                                 key_pos head_index d_head
                                 -> head_index query_pos key_pos }}
                             , q
                             , k }}}
                    : tensor [n_heads; pos; pos] A)
                 q
                 k in
          PArray.maybe_checkpoint (qk / broadcast' attn_scale))%core.

    Definition apply_causal_mask (attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A)
      : tensor (batch ::' n_heads ::' pos ::' pos) A
      := Tensor.map'
           (fun attn_scores : tensor [pos; pos] A
            => Tensor.where_ mask.[:pos,:pos] attn_scores (broadcast' IGNORE))
           attn_scores.

    Definition masked_attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A
      := apply_causal_mask attn_scores.

    Definition pattern : tensor (batch ::' n_heads ::' pos ::' pos) A
      := PArray.maybe_checkpoint (softmax_dim_m1 (PArray.maybe_checkpoint masked_attn_scores)).

    Definition z : tensor (batch ::' pos ::' n_heads ::' d_head) A
      := PArray.maybe_checkpoint
           (Tensor.map2'
              (fun (v : tensor [pos; n_heads; d_head] A)
                   (pattern : tensor [n_heads; pos; pos] A)
               => weaksauce_einsum {{{ {{  key_pos head_index d_head,
                              head_index query_pos key_pos ->
                              query_pos head_index d_head }}
                          , v
                          , pattern }}}
                 : tensor [pos; n_heads; d_head] A)
              v
              pattern).

    Definition attn_out : tensor (batch ::' pos ::' d_model) A
      := (let out
            := Tensor.map'
                 (fun z : tensor [pos; n_heads; d_head] A
                  => weaksauce_einsum {{{ {{ pos head_index d_head,
                                 head_index d_head d_model ->
                                 pos d_model }}
                             , z
                             , W_O }}}
                    : tensor [pos; d_model] A)
                 z in
          PArray.maybe_checkpoint (out + broadcast b_O))%core.
  End __.
End Attention.

Module MLP.
  Section __.
    Context {r} {batch : Shape r}
      {pos d_model d_mlp}
      (act_fn_kind : ActivationKind)
      {A}
      {addA : has_add A} {mulA : has_mul A}
      {maxA : has_max A}
      {zeroA : has_zero A}
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (W_in : tensor [d_model; d_mlp] A) (b_in : tensor [d_mlp] A)
      (W_out : tensor [d_mlp; d_model] A) (b_out : tensor [d_model] A)
      (x : tensor (batch ::' pos ::' d_model) A)
    .

    Definition pre_act : tensor (batch ::' pos ::' d_mlp) A
      := let x' : tensor (batch ::' pos ::' d_mlp) A
           := Tensor.map'
                (fun x : tensor [pos; d_mlp] A
                 => weaksauce_einsum {{{ {{ pos d_model , d_model d_mlp -> pos d_mlp }}
                            , x
                            , W_in }}}
                   : tensor [pos; d_mlp] A)
                x in
         x' + broadcast b_in.

    Definition act_fn : tensor (batch ::' pos ::' d_mlp) A -> tensor (batch ::' pos ::' d_mlp) A
      := match act_fn_kind with
         | relu => Tensor.relu
         end.

    (* TODO: if act_fn is *_ln, then handle ln *)
    Definition post_act : tensor (batch ::' pos ::' d_mlp) A
      := act_fn pre_act.

    Definition forward : tensor (batch ::' pos ::' d_model) A
      := let fx' : tensor (batch ::' pos ::' d_model) A
           := Tensor.map'
                (fun fx : tensor [pos; d_mlp] A
                 => weaksauce_einsum {{{ {{ pos d_mlp , d_mlp d_model -> pos d_model }}
                            , fx
                            , W_out }}}
                   : tensor [pos; d_mlp] A)
                post_act in
         fx' + broadcast b_out.
  End __.
End MLP.

Module TransformerBlock.
  Section __.
    Context {r} {batch : Shape r}
      {pos n_heads d_model d_head} {n_ctx:N}
      {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
      {normalization_type : with_default "normalization_type" (option NormalizationType) (Some LN)}
      (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape)
      (ln_s := (batch ::' pos ++ maybe_n_heads use_split_qkv_input)%shape)
      {A}
      {zeroA : has_zero A} {coerZ : has_coer Z A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {maxA : has_max A}
      {sqrtA : has_sqrt A} {expA : has_exp A}
      (defaultA : pointed A := @coer _ _ coerZ point)
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (W_Q W_K W_V W_O : tensor [n_heads; d_model; d_head] A)
      (b_Q b_K b_V : tensor [n_heads; d_head] A)
      (b_O : tensor [d_model] A)
      (eps : A)
      (ln1_w ln1_b ln2_w ln2_b : ln_tensor_gen d_model normalization_type A)
      (resid_pre : tensor ((batch ::' pos) ++' [d_model]) A).
    #[local] Existing Instance defaultA.


    Definition add_head_dimension
      (resid_pre : tensor ((batch ::' pos) ++' [d_model]) A)
      : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A
      := if use_split_qkv_input return tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A
         then Tensor.map'
                (fun resid_pre : tensor [d_model] A
                 => Tensor.repeat [n_heads] resid_pre
                   : tensor [n_heads; d_model] A)
                resid_pre
         else resid_pre.
    Definition query_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A
      := add_head_dimension resid_pre.
    Definition key_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A
      := add_head_dimension resid_pre.
    Definition value_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A
      := add_head_dimension resid_pre.

    #[local] Notation LayerNorm_forward
      := (match normalization_type
                return ln_tensor_gen _ normalization_type _ -> ln_tensor_gen _ normalization_type _ -> _
          with
          | Some LN => LayerNorm.forward eps
          | Datatypes.None => fun _ _ x => PArray.maybe_checkpoint x
          end)
           (only parsing).

    Definition ln1 (*{r} {s : Shape r}*) (t : tensor (ln_s ::' d_model) A) : tensor (ln_s ::' d_model) A
      := LayerNorm_forward ln1_w ln1_b t.
    Definition ln2 (*{r} {s : Shape r}*) (t : tensor (ln_s ::' d_model) A) : tensor (ln_s ::' d_model) A
      := LayerNorm_forward ln2_w ln2_b t.

    Definition attn_only_out : tensor (batch ++ [pos; d_model]) A
      := (let attn_out : tensor (batch ++ [pos; d_model]) A
            := Attention.attn_out
                 (n_ctx:=n_ctx)
                 W_Q W_K W_V W_O
                 b_Q b_K b_V b_O
                 (ln1 query_input)
                 (ln1 key_input)
                 (ln1 value_input) in
          resid_pre + attn_out)%core.

    (** convenience *)
    Local Definition attn_masked_attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A
      := Attention.masked_attn_scores
           (n_ctx:=n_ctx)
           W_Q W_K
           b_Q b_K
           (ln1 query_input)
           (ln1 key_input).

    Local Definition attn_pattern : tensor (batch ::' n_heads ::' pos ::' pos) A
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
    Context {d_vocab d_vocab_out n_heads d_model d_head} {n_ctx:N}
      {r} {batch : Shape r} {pos}
      (s := (batch ::' pos)%shape)
      (resid_shape := (s ::' d_model)%shape)
      {normalization_type : with_default "normalization_type" (option NormalizationType) (Some LN)}
      {A}
      {zeroA : has_zero A} {coerZ : has_coer Z A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {maxA : has_max A}
      {sqrtA : has_sqrt A} {expA : has_exp A}
      (defaultA : pointed A := @coer _ _ coerZ point)
      (*{use_split_qkv_input : with_default "use_split_qkv_input" bool false}*)
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (eps : A)

      (W_E : tensor [d_vocab; d_model] A)
      (W_pos : tensor [n_ctx; d_model] A)

      (blocks_params
        : list (block_params_type_gen n_heads d_model d_head normalization_type A)
            (* (W_Q W_K W_V W_O : tensor [n_heads; d_model; d_head] A)
      (b_Q b_K b_V : tensor [n_heads; d_head] A)
      (b_O : tensor [d_model] A)
      (ln1_w ln1_b : tensor [d_model] A) *))

      (ln_final_w ln_final_b : ln_tensor_gen d_model normalization_type A)

      (W_U : tensor [d_model; d_vocab_out] A) (b_U : tensor [d_vocab_out] A)
    .
    #[local] Existing Instance defaultA.

    Definition embed (tokens : tensor s IndexType) : tensor resid_shape A
      := Embed.forward W_E tokens.

    Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape A
      := PosEmbed.forward W_pos tokens.

    Definition blocks : list (tensor resid_shape A -> tensor resid_shape A)
      := List.map
           (fun '((W_Q, W_K, W_V, W_O,
                    b_Q, b_K, b_V,
                    b_O,
                    ln1_w, ln1_b)%primproj)
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
                return ln_tensor_gen _ normalization_type _ -> ln_tensor_gen _ normalization_type _ -> _
          with
          | Some LN => LayerNorm.forward eps
          | Datatypes.None => fun _ _ t => t
          end)
           (only parsing).

    Definition ln_final (resid : tensor resid_shape A) : tensor resid_shape A
      := LayerNorm_forward ln_final_w ln_final_b resid.

    Definition unembed (resid : tensor resid_shape A) : tensor (s ::' d_vocab_out) A
      := Unembed.forward W_U b_U resid.

    Polymorphic Definition blocks_cps {T} {n : with_default "blocks n" nat (List.length blocks)} (residual : tensor resid_shape A) (K : tensor resid_shape A -> T) : T
      := List.fold_right
           (fun block cont residual
            => let residual := PArray.maybe_checkpoint (block residual) in
               cont residual)
           K
           (List.firstn n blocks)
           residual.

    Definition resid_postembed (tokens : tensor s IndexType) : tensor resid_shape A
      := (let embed          := embed tokens in
          let pos_embed      := pos_embed tokens in
          PArray.maybe_checkpoint (embed + pos_embed)%core).

    Definition logits (tokens : tensor s IndexType) : tensor (s ::' d_vocab_out) A
      := (let residual       := resid_postembed tokens in
          blocks_cps
            residual
            (fun residual
             => let residual := PArray.maybe_checkpoint (ln_final residual) in
                let logits   := PArray.maybe_checkpoint (unembed residual) in
                logits)).

    Definition forward (tokens : tensor s IndexType) : tensor (s ::' d_vocab_out) A
      := logits tokens.

    (** convenience *)
    Local Definition blocks_attn_masked_attn_scores
      : list (tensor resid_shape A -> tensor (batch ::' n_heads ::' pos ::' pos) A)
      := List.map
           (fun '((W_Q, W_K, W_V, W_O,
                    b_Q, b_K, b_V,
                    b_O,
                    ln1_w, ln1_b)%primproj)
            => TransformerBlock.attn_masked_attn_scores
                 (n_ctx:=n_ctx)
                 W_Q W_K
                 b_Q b_K
                 eps
                 ln1_w ln1_b)
           blocks_params.

    Local Definition blocks_attn_pattern
      : list (tensor resid_shape A -> tensor (batch ::' n_heads ::' pos ::' pos) A)
      := List.map
           (fun '((W_Q, W_K, W_V, W_O,
                    b_Q, b_K, b_V,
                    b_O,
                    ln1_w, ln1_b)%primproj)
            => TransformerBlock.attn_pattern
                 (n_ctx:=n_ctx)
                 W_Q W_K
                 b_Q b_K
                 eps
                 ln1_w ln1_b)
           blocks_params.

    Local Definition masked_attn_scores (n : nat) (tokens : tensor s IndexType)
      : option (tensor (batch ::' n_heads ::' pos ::' pos) A)
      := match List.nth_error blocks_attn_masked_attn_scores n with
         | Some block_n_attn_masked_attn_scores
           => Some (let residual       := resid_postembed tokens in
                    blocks_cps
                      (n:=Nat.pred n)
                      residual
                      (fun residual
                       => PArray.maybe_checkpoint (block_n_attn_masked_attn_scores residual)))
         | None => None
         end.

    Local Definition attn_pattern (n : nat) (tokens : tensor s IndexType)
      : option (tensor (batch ::' n_heads ::' pos ::' pos) A)
      := match List.nth_error blocks_attn_pattern n with
         | Some block_n_attn_pattern
           => Some (let residual       := resid_postembed tokens in
                    blocks_cps
                      (n:=Nat.pred n)
                      residual
                      (fun residual
                       => PArray.maybe_checkpoint (block_n_attn_pattern residual)))
         | None => None
         end.
  End __.
End HookedTransformer.
