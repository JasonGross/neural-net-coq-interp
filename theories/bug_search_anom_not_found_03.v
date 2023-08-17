(* -*- mode: coq; coq-prog-args: ("-emacs" "-q" "-w" "+implicit-core-hint-db,+implicits-in-term,+non-reversible-notation,+deprecated-intros-until-0,+deprecated-focus,+unused-intro-pattern,+variable-collision,+unexpected-implicit-declaration,+omega-is-deprecated,+deprecated-instantiate-syntax,+non-recursive,+undeclared-scope,+deprecated-hint-rewrite-without-locality,+deprecated-hint-without-locality,+deprecated-instance-without-locality,+deprecated-typeclasses-transparency-without-locality,-ltac2-missing-notation-var,unsupported-attributes" "-w" "-deprecated-native-compiler-option" "-native-compiler" "no" "-R" "/github/workspace/neural-net-coq-interp/theories" "NeuralNetInterp" "-Q" "/github/workspace/cwd" "Top" "-Q" "/home/coq/.opam/4.13.1+flambda/lib/coq/user-contrib/Bignums" "Bignums" "-Q" "/home/coq/.opam/4.13.1+flambda/lib/coq/user-contrib/Ltac2" "Ltac2" "-top" "NeuralNetInterp.bug_search_anom_not_found_03") -*- *)
(* File reduced by coq-bug-minimizer from original input, then from 220 lines to 39 lines, then from 52 lines to 522 lines, then from 524 lines to 229 lines, then from 699 lines to 216 lines, then from 228 lines to 118 lines, then from 131 lines to 665 lines, then from 670 lines to 135 lines, then from 148 lines to 1326 lines, then from 1331 lines to 173 lines, then from 186 lines to 747 lines *)
(* coqc version 8.19+alpha compiled with OCaml 4.13.1
   coqtop version buildkitsandbox:/home/coq/.opam/4.13.1+flambda/.opam-switch/build/coq-core.dev/_build/default,master (61ee398ed32f9334dd664ea8ed2697178e6e3844)
   Expected coqc runtime on this file: 0.000 sec *)
Require Coq.Init.Ltac.
Module Export AdmitTactic.
Module Import LocalFalse.
Inductive False : Prop := .
End LocalFalse.
Axiom proof_admitted : False.
Import Coq.Init.Ltac.
Tactic Notation "admit" := abstract case proof_admitted.
End AdmitTactic.
Require NeuralNetInterp.Util.Classes.RelationPairs.Dependent.
Require NeuralNetInterp.Util.Option.
Require NeuralNetInterp.Torch.Tensor.Instances.
Require Coq.Arith.Arith.
Require Coq.Arith.Wf_nat.
Require Coq.Array.PArray.
Require Coq.Bool.Bool.
Require Coq.Classes.Morphisms.
Require Coq.Classes.RelationClasses.
Require Coq.Floats.Floats.
Require Coq.Lists.List.
Require Coq.NArith.NArith.
Require Coq.Numbers.Cyclic.Int63.Sint63.
Require Coq.Numbers.Cyclic.Int63.Uint63.
Require Coq.PArith.PArith.
Require Coq.QArith.QArith.
Require Coq.QArith.Qabs.
Require Coq.QArith.Qround.
Require Coq.Reals.Reals.
Require Coq.Relations.Relation_Definitions.
Require Coq.Setoids.Setoid.
Require Coq.Strings.String.
Require Coq.Structures.Equalities.
Require Coq.Unicode.Utf8.
Require Coq.Wellfounded.Wellfounded.
Require Coq.ZArith.Wf_Z.
Require Coq.ZArith.ZArith.
Require Coq.micromega.Lia.
Require Coq.micromega.Lqa.
Require Ltac2.Init.
Require NeuralNetInterp.Util.Arrow.
Require NeuralNetInterp.Util.Classes.
Require NeuralNetInterp.Util.Notations.
Require NeuralNetInterp.Util.Tactics.DestructHyps.
Require NeuralNetInterp.Util.Tactics.Head.
Require Ltac2.Bool.
Require Ltac2.Constant.
Require Ltac2.Constr.
Require Ltac2.Constructor.
Require Ltac2.Evar.
Require Ltac2.FSet.
Require Ltac2.Float.
Require Ltac2.Ident.
Require Ltac2.Ind.
Require Ltac2.Int.
Require Ltac2.Ltac1.
Require Ltac2.Message.
Require Ltac2.Meta.
Require Ltac2.Proj.
Require Ltac2.Std.
Require Ltac2.String.
Require Ltac2.Uint63.
Require NeuralNetInterp.Util.Arith.Reals.Definitions.
Require NeuralNetInterp.Util.Arith.ZArith.
Require NeuralNetInterp.Util.Bool.
Require NeuralNetInterp.Util.Default.
Require NeuralNetInterp.Util.List.
Require NeuralNetInterp.Util.Tactics.BreakMatch.
Require Ltac2.Char.
Require Ltac2.Control.
Require Ltac2.Env.
Require Ltac2.FMap.
Require Ltac2.Printf.
Require NeuralNetInterp.Util.Arith.Classes.
Require NeuralNetInterp.Util.Monad.
Require NeuralNetInterp.Util.Nat.
Require NeuralNetInterp.Util.Tactics.DestructHead.
Require Ltac2.Option.
Require Ltac2.Pattern.
Require NeuralNetInterp.Util.ErrorT.
Require NeuralNetInterp.Util.List.Proofs.
Require Ltac2.Array.
Require Ltac2.List.
Require Ltac2.Fresh.
Require NeuralNetInterp.TransformerLens.HookedTransformer.Config.Common.
Require Ltac2.Notations.
Require NeuralNetInterp.Util.Pointed.
Require NeuralNetInterp.Util.PolymorphicOption.
Require NeuralNetInterp.Util.Arith.QArith.
Require NeuralNetInterp.Util.Slice.
Require NeuralNetInterp.Util.Arith.FloatArith.Definitions.
Require Ltac2.Ltac2.
Require NeuralNetInterp.Util.Arith.Instances.
Require NeuralNetInterp.Util.Tactics2.Constr.
Require NeuralNetInterp.Util.Tactics2.Constr.Unsafe.MakeAbbreviations.
Require NeuralNetInterp.Util.Tactics2.FixNotationsForPerformance.
Require NeuralNetInterp.Util.Tactics2.Ident.
Require NeuralNetInterp.Util.Tactics2.List.
Require NeuralNetInterp.Util.Wf_Uint63.
Require NeuralNetInterp.Util.PArray.
Require NeuralNetInterp.Util.PArray.Proofs.
Require NeuralNetInterp.Torch.Tensor.
Require NeuralNetInterp.Torch.Slicing.
Require NeuralNetInterp.TransformerLens.HookedTransformer.Config.
Require NeuralNetInterp.Torch.Einsum.

Module NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer_WRAPPED.
Module HookedTransformer.
Import Coq.Floats.Floats Coq.Numbers.Cyclic.Int63.Sint63 Coq.Numbers.Cyclic.Int63.Uint63 Coq.QArith.QArith Coq.micromega.Lia Coq.Lists.List Coq.Array.PArray Coq.Classes.Morphisms Coq.Classes.RelationClasses.
Import NeuralNetInterp.Util.Default NeuralNetInterp.Util.Pointed NeuralNetInterp.Util.PArray NeuralNetInterp.Util.List NeuralNetInterp.Util.Notations NeuralNetInterp.Util.Arith.Classes NeuralNetInterp.Util.Arith.Instances NeuralNetInterp.Util.Bool.
Import NeuralNetInterp.Torch.Tensor NeuralNetInterp.Torch.Einsum NeuralNetInterp.Torch.Slicing.
Export NeuralNetInterp.TransformerLens.HookedTransformer.Config.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import NeuralNetInterp.Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Local Open Scope raw_tensor_scope.

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
    #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).

    Definition linpart (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A
      := (x - reduce_axis_m1 (keepdim:=true) mean x)%core.

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

Module Attention.
  Section __.
    Context {r} {batch : Shape r}
      {pos n_heads d_model d_head} {n_ctx:N}
      {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
      {A}
      {sqrtA : has_sqrt A} {coerZ : has_coer Z A} {addA : has_add A} {zeroA : has_zero A} {mulA : has_mul A} {divA : has_div A} {expA : has_exp A}
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
    #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).

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
      := checkpoint (einsum_input query_input W_Q + broadcast b_Q)%core.
    Definition k : tensor (batch ++' [pos; n_heads; d_head]) A
      := checkpoint (einsum_input key_input W_K + broadcast b_K)%core.
    Definition v : tensor (batch ++' [pos; n_heads; d_head]) A
      := checkpoint (einsum_input value_input W_V + broadcast b_V)%core.

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
          checkpoint (qk / broadcast' attn_scale))%core.

    Definition apply_causal_mask (attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A)
      : tensor (batch ::' n_heads ::' pos ::' pos) A
      := Tensor.map'
           (fun attn_scores : tensor [pos; pos] A
            => Tensor.where_ mask.[:pos,:pos] attn_scores (broadcast' IGNORE))
           attn_scores.

    Definition masked_attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A
      := apply_causal_mask attn_scores.

    Definition pattern : tensor (batch ::' n_heads ::' pos ::' pos) A
      := checkpoint (softmax_dim_m1 masked_attn_scores).

    Definition z : tensor (batch ::' pos ::' n_heads ::' d_head) A
      := checkpoint
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
          checkpoint (out + broadcast b_O))%core.
  End __.
End Attention.

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
    #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).


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
          | Datatypes.None => fun _ _ x => checkpoint x
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
    Definition attn_masked_attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A
      := Attention.masked_attn_scores
           (n_ctx:=n_ctx)
           W_Q W_K
           b_Q b_K
           (ln1 query_input)
           (ln1 key_input).

    Definition attn_pattern : tensor (batch ::' n_heads ::' pos ::' pos) A
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
    #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).

    Definition embed (tokens : tensor s IndexType) : tensor resid_shape A
      := Embed.forward W_E tokens.

    Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape A
      := PosEmbed.forward W_pos tokens.

    Definition blocks : list (tensor resid_shape A -> tensor resid_shape A)
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
            => let residual := checkpoint (block residual) in
               cont residual)
           K
           (List.firstn n blocks)
           residual.

    Definition resid_postembed (tokens : tensor s IndexType) : tensor resid_shape A
      := (let embed          := embed tokens in
          let pos_embed      := pos_embed tokens in
          checkpoint (embed + pos_embed)%core).

    Definition logits (tokens : tensor s IndexType) : tensor (s ::' d_vocab_out) A
      := (let residual       := resid_postembed tokens in
          blocks_cps
            residual
            (fun residual
             => let residual := checkpoint (ln_final residual) in
                let logits   := checkpoint (unembed residual) in
                logits)).

    Definition forward (tokens : tensor s IndexType) : tensor (s ::' d_vocab_out) A
      := logits tokens.

    (** convenience *)
    Definition blocks_attn_masked_attn_scores
      : list (tensor resid_shape A -> tensor (batch ::' n_heads ::' pos ::' pos) A)
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

    Definition blocks_attn_pattern
      : list (tensor resid_shape A -> tensor (batch ::' n_heads ::' pos ::' pos) A)
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

    Definition masked_attn_scores (n : nat) (tokens : tensor s IndexType)
      : option (tensor (batch ::' n_heads ::' pos ::' pos) A)
      := match List.nth_error blocks_attn_masked_attn_scores n with
         | Some block_n_attn_masked_attn_scores
           => Some (let residual       := resid_postembed tokens in
                    blocks_cps
                      (n:=Nat.pred n)
                      residual
                      (fun residual
                       => checkpoint (block_n_attn_masked_attn_scores residual)))
         | None => None
         end.

    Definition attn_pattern (n : nat) (tokens : tensor s IndexType)
      : option (tensor (batch ::' n_heads ::' pos ::' pos) A)
      := match List.nth_error blocks_attn_pattern n with
         | Some block_n_attn_pattern
           => Some (let residual       := resid_postembed tokens in
                    blocks_cps
                      (n:=Nat.pred n)
                      residual
                      (fun residual
                       => checkpoint (block_n_attn_pattern residual)))
         | None => None
         end.
  End __.
End HookedTransformer.

End HookedTransformer.

End NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer_WRAPPED.
Module Export NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer.
Module Export NeuralNetInterp.
Module Export TransformerLens.
Module HookedTransformer.
Include NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer_WRAPPED.HookedTransformer.
End HookedTransformer.

End TransformerLens.

End NeuralNetInterp.

End NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer.
Module Export Instances.
Import NeuralNetInterp.Util.Option.
Import NeuralNetInterp.Torch.Tensor.Instances.
Import NeuralNetInterp.TransformerLens.HookedTransformer.
Import Dependent.RelationPairsNotations.

Module Export HookedTransformer.
Definition block_params_type_genR n_heads d_model d_head nt : Dependent.relation (block_params_type_gen n_heads d_model d_head nt).
Admitted.

  Module Export HookedTransformer.

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
             ==> List.Forall2 ∘ @block_params_type_genR n_heads d_model d_head normalization_type
             ==> Dependent.const eq
             ==> Dependent.const Tensor.eqf
             ==> @option_eq ∘ Tensor.eqfR)
          (@HookedTransformer.HookedTransformer.masked_attn_scores d_vocab n_heads d_model d_head n_ctx r batch pos normalization_type).
Admitted.
  End HookedTransformer.
End HookedTransformer.

End Instances.
Module Export Parameters.
Import Coq.Floats.Floats.
Import Coq.NArith.NArith.
Import NeuralNetInterp.TransformerLens.HookedTransformer.Config.Common.
Local Open Scope float_scope.

Module cfg <: CommonConfig.
  Definition d_model := 32%N.
  Definition n_ctx := 2%N.
  Definition d_head := 32%N.
  Definition n_heads := 1%N.
  Definition d_vocab := 64%N.
  Definition eps := 1e-05.
  Definition normalization_type := Some LN.
  Definition d_vocab_out := 64%N.
End cfg.

End Parameters.
Import Coq.Floats.Floats.
Import Coq.QArith.QArith.
Import Coq.Lists.List.
Import NeuralNetInterp.Util.Default.
Import NeuralNetInterp.Util.Pointed.
Import NeuralNetInterp.Util.Arith.Classes.
Import NeuralNetInterp.Util.Arith.Instances.
Import NeuralNetInterp.Util.Option.
Import NeuralNetInterp.Torch.Tensor.
Import NeuralNetInterp.TransformerLens.HookedTransformer.
Import Instances.Truncating.

Module Model (cfg : Config).

  Module Export HookedTransformer.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
        {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
        {sqrtA : has_sqrt A} {expA : has_exp A}
        {use_checkpoint : with_default "use_checkpoint" bool true}.
Let coerA' (x : float) : A.
Admitted.
      #[local] Coercion coerA' : float >-> A.
Let coer_ln_tensor : cfg.ln_tensor float -> cfg.ln_tensor A.
Admitted.
      Definition coer_blocks_params
        := List.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => ((W_Q:tensor _ A), (W_K:tensor _ A), (W_V:tensor _ A), (W_O:tensor _ A),
                   (b_Q:tensor _ A), (b_K:tensor _ A), (b_V:tensor _ A), (b_O:tensor _ A),
                   coer_ln_tensor ln1_w, coer_ln_tensor ln1_b)).
Definition masked_attn_scores (n : nat) (tokens : tensor s IndexType)
        : option (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A).
exact (HookedTransformer.HookedTransformer.masked_attn_scores
             (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type)cfg.eps
             cfg.W_E cfg.W_pos
             (coer_blocks_params cfg.blocks_params)
             n tokens).
Defined.
    End __.
  End HookedTransformer.
End Model.
Import ListNotations.

Module cfg <: Config.
  Include Parameters.cfg.
  Parameter W_E : tensor [d_vocab; d_model] float.
  Parameter W_pos : tensor [n_ctx; d_model] float.
  Parameter W_U : tensor [d_model; d_vocab_out] float.
  Parameter b_U : tensor [d_vocab_out] float.
  Notation ln_tensor A := (ln_tensor_gen d_model normalization_type A).
  Parameter ln_final_w : ln_tensor float.
  Parameter ln_final_b : ln_tensor float.
  Notation block_params_type A := (block_params_type_gen n_heads d_model d_head normalization_type A).
  Parameter block_params : block_params_type float.
  Definition blocks_params := [block_params].
End cfg.
  Include Model cfg.

  Section with_batch.
    Context {r} {batch : Shape r} {pos}
      (s := (batch ::' pos)%shape)
      (resid_shape := (s ::' cfg.d_model)%shape)
      {return_per_token : with_default "return_per_token" bool false}
      {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
      (defaultA : pointed A := @coer _ _ coerZ point)
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {ltbA : has_ltb A}
      {oppA : has_opp A} {sqrtA : has_sqrt A} {expA : has_exp A} {lnA : has_ln A}
      {use_checkpoint : with_default "use_checkpoint" bool true}.
Definition masked_attn_scores (tokens : tensor s IndexType)
        : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A.
exact (Option.invert_Some
           (HookedTransformer.masked_attn_scores (A:=A) 0 tokens)).
Defined.
  End with_batch.
Import NeuralNetInterp.Torch.Tensor.Instances.
Import Dependent.ProperNotations.

  #[export] Instance masked_attn_scores_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR)
        (@masked_attn_scores r batch pos).
  Proof.
    cbv [masked_attn_scores].
    pose proof (@HookedTransformer.HookedTransformer.masked_attn_scores_Proper_dep) as H.
    repeat intro.
    repeat (let v := open_constr:(_) in specialize (H v)).
    move H at bottom.
    revert H.
    lazymatch goal with
    | [ |- ?R _ _ ?R'' ?x ?y -> ?R' (invert_Some ?x' ?i) (invert_Some ?y' ?i) ]
      => unify x x'; unify y y'; unify R'' R'; set (x'':=x); set (y'':=y);
         intro H;
         refine (@invert_Some_Proper_dep _ _ (Tensor.eqfR R') x y H i)
    end.
    Unshelve.
    Search HookedTransformer.block_params_type_genR.
