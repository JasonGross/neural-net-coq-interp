From Coq Require Vector.
From Coq Require Import Derive.
From Coq.Structures Require Import Equalities.
From Coq Require Import Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util Require Import PrimitiveProd.
From NeuralNetInterp.Util Require Export Default Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Config.
Import Instances.Truncating.
#[local] Open Scope core_scope.

#[local] Coercion Vector.of_list : list >-> Vector.t.
#[local] Open Scope primproj_scope.

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
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.q (A:=A) W_Q b_Q query_input)
             cfg.blocks_params.
      Definition ks : Vector.t (tensor (s ++' [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.k (A:=A) W_K b_K key_input)
             cfg.blocks_params.
      Definition vs : Vector.t (tensor (s ++' [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.v (A:=A) W_V b_V value_input)
             cfg.blocks_params.

      Definition attn_scores : Vector.t (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.attn_scores (A:=A) W_Q W_K b_Q b_K query_input key_input)
             cfg.blocks_params.

      Definition masked_attn_scores : Vector.t (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.masked_attn_scores (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K query_input key_input)
             cfg.blocks_params.

      Definition patterns : Vector.t (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.pattern (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K query_input key_input)
             cfg.blocks_params.

      Definition zs : Vector.t (tensor (s ++ [cfg.n_heads; cfg.d_head]) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => Attention.z (A:=A) (n_ctx:=cfg.n_ctx) W_Q W_K W_V b_Q b_K b_V query_input key_input value_input)
             cfg.blocks_params.

      Definition attn_outs : Vector.t (tensor resid_shape A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
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
          := match cfg.normalization_type as nt return Config.ln_tensor_gen _ nt float -> Config.ln_tensor_gen _ nt A with
             | Some LN
             | Datatypes.None
               => fun x => x
             end.

      Definition attn_only_outs : Vector.t (tensor resid_shape A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => TransformerBlock.attn_only_out (A:=A) (normalization_type:=cfg.normalization_type) (n_ctx:=cfg.n_ctx) W_Q W_K W_V W_O b_Q b_K b_V b_O cfg.eps (coer_ln_tensor ln1_w) (coer_ln_tensor ln1_b) resid_pre)
             cfg.blocks_params.

      Definition attn_masked_attn_scores : Vector.t (tensor (s ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
              => HookedTransformer.TransformerBlock.attn_masked_attn_scores (A:=A) (normalization_type:=cfg.normalization_type) (n_ctx:=cfg.n_ctx) W_Q W_K b_Q b_K  cfg.eps (coer_ln_tensor ln1_w) (coer_ln_tensor ln1_b) resid_pre)
             cfg.blocks_params.

      Definition attn_patterns : Vector.t (tensor (s ::' pos ::' pos) A) cfg.n_layers
        := Vector.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b)%primproj : cfg.block_params_type float)
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
          := match cfg.normalization_type as nt return Config.ln_tensor_gen _ nt float -> Config.ln_tensor_gen _ nt A with
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
(*
  Local Ltac set_step _ :=
    match goal with
    | [ |- context G[let s : ?T := ?v in @?f s] ]
      => lazymatch T with
         | Shape _ => idtac
         | float -> _ => idtac
         | int => idtac
         | N => idtac
         end;
         let G := context G [f v] in
         change G; cbv beta iota
    | [ |- context G[let s : ?T := ?v in @?f s] ]
      => lazymatch goal with
         | [ s' := v |- _ ]
           => let G := context G [f s'] in
              change G; cbv beta iota
         end
    | [ |- context G[let s := Definitions.PrimFloat.of_Z 0 in @?f s] ]
      => let G := context G[f 0%float] in
         change G; cbv beta iota
    | [ |- context G[let s : ?T := ?v in @?f s] ]
      => lazymatch T with
         | match cfg.normalization_type with _ => _ end -> match cfg.normalization_type with _ => _ end => idtac
         | tensor _ _ => idtac
         end;
         lazymatch v with
         | context[let _ := _ in _] => fail
         | _ => idtac
         end;
         let s := fresh s in
         pose v as s;
         let G := context G [f s] in
         change G; cbv beta iota
    end.
  From Ltac2 Require Import Ltac2 Printf.
  Derive logits_all_tokens SuchThat (logits_all_tokens = HookedTransformer.logits all_tokens :> tensor _ float) As eq_logits_all_tokens.
  Proof.
    Ltac2 rec step_cbv (term : constr) (k : constr -> constr) :=
      lazy_match! term with
      | ?f (let x : ?t := ?v in @?body x)
        => let x' := Fresh.in_goal @x' in
           let new_body := Constr.in_context x' t (fun () => Control.refine (fun () => k constr:($f ($body &x')))) in
           let res := constr:(let x := $v in $new_body) in
           let res := (eval cbv beta in $res) in
           res
      | (let x : ?t := ?v in @?body x) ?y
        => let x' := Fresh.in_goal @x' in
           let new_body := Constr.in_context x' t (fun () => Control.refine (fun () => k constr:($body &x' $y))) in
           let res := constr:(let x := $v in $new_body) in
           let res := (eval cbv beta in $res) in
           printf "%t" res;
           k res
      | let y := (let x := ?v in @?body x) in @?body' y
        => let res := constr:(let x := $v in let y := $body x in $body' y) in
           let res := (eval cbv beta in $res) in
           k res
      | (let x : ?t := ?v in @?body x)
        => let x' := Fresh.in_goal @x' in
           let new_body := Constr.in_context x' t (fun () => Control.refine (fun () => step_cbv constr:($body &x') k)) in
           let res := constr:(let x := $v in $new_body) in
           let res := (eval cbv beta in $res) in
           res
      | ?f ?x
        => step_cbv
             f
             (fun f'
              => let res := constr:($f' $x) in
                 let res := (eval cbv beta in $res) in
                 k res)
      | _
        => match Constr.Unsafe.kind term with
           | Constr.Unsafe.App f args
             => step_cbv
                  f
                  (fun f =>
                     let res := Constr.Unsafe.make (Constr.Unsafe.App f args) in
                     let res := (eval cbv beta in $res) in
                     k res)
           | Constr.Unsafe.Constant c _
             => let c := Std.ConstRef c in
                let res := (eval cbv delta [$c] in $term) in
                k res
           | _ => k term
           end
      end.
    Ltac2 rec stepn_cbv (n : int) (term : constr)
      := if Int.le n 0
         then term
         else stepn_cbv (Int.sub n 1) (step_cbv term (fun x => x)).
    lazy_match! goal with
    | [ |- _ = ?rhs ]
      => let rhs' := stepn_cbv 14 rhs in
         printf "%t" rhs'
    end.
      lazymatch term with
      | ?f ?x => let f' := step_cbv term in
                 constr:(f' x)
      | let x :=
    cbv beta iota delta [HookedTransformer.logits].
    Set Printing Coercions.
    cbv beta iota delta [HookedTransformer.logits
                           HookedTransformer.HookedTransformer.logits HookedTransformer.HookedTransformer.resid_postembed
                           HookedTransformer.HookedTransformer.embed
                           HookedTransformer.HookedTransformer.pos_embed
                           HookedTransformer.HookedTransformer.ln_final
                           HookedTransformer.HookedTransformer.unembed
                           HookedTransformer.Embed.forward
                           HookedTransformer.PosEmbed.forward
                           Classes.add Classes.pow
                           tensor_add
                           map2
                           float_has_add N_has_pow
                           coer_refl
                           HookedTransformer.coer_blocks_params
                           Slicing.FancyIndex.slice Slicing.FancyIndex.slice_ Slicing.SliceIndex.slice Slicing.FancyIndex.broadcast
                           RawIndex.split_radd
                           reshape_app_combine' map_dep
                           RawIndex.uncurry_radd
                           repeat repeat' reshape_app_combine
                           Nat.radd
                           coer_tensor Tensor.map
                           coer coer_Z_float point default_Z];
      repeat first [ set_step ()
                   | progress cbv beta iota delta [Slicing.SliceIndex.SliceIndexType.slice Slice.invert_index Slice.concretize Slice.norm_concretize PolymorphicOption.Option.sequence_return Slice.step Shape.tl Shape.hd Shape.nil Shape.snoc Slice.start int_has_one int_has_zero Slice.Concrete.normalize Slice.stop Slice.Concrete.start Slicing.FancyIndex.FancyIndexType.broadcast repeat' map adjust_index_for] in *; cbn beta iota delta [fst snd] in * ];
      cbv beta iota zeta in embed, pos_embed;
      match goal with
      | [ |- ?x = ?f ?y ]
        => let y' := open_constr:(_) in
           let H := fresh in
           assert (H : forall a, y a = y' a);
           [ intro | clear H; transitivity (f y') ]
      end;
      repeat match goal with H : _ |- _ => clear H end;
      cbv beta iota delta [
                           HookedTransformer.HookedTransformer.blocks_cps
                           HookedTransformer.HookedTransformer.blocks
                           HookedTransformer.Unembed.forward
                           map'];
      repeat set_step ().
    clear.
    cbv beta iota delta [HookedTransformer.HookedTransformer.blocks_cps].
    Print HookedTransformer.Hook
      | [ |- context G[let s : ?T := ?v in @?f s] ]
             => lazymatch T with
                | match cfg.normalization_type with _ => _ end -> match cfg.normalization_type with _ => _ end => idtac
                | tensor _ _ => idtac
                end;
                let s := fresh s in
                pose v as s;
                let G := context G [f s] in
                change G; cbv beta iota
           end.

    | [ |-
        =>
 *)
  (*
  Set Printing Implicit. Set Printing All.
  Definition logits_all_tokens : tensor _ float
    := HookedTransformer.logits all_tokens.
  Print logits_all_tokens.
   *)
  Notation logits_all_tokens
    := (@HookedTransformer.logits 1 [Uint63.of_Z (Z.of_N (@pow N N N N_has_pow cfg.d_vocab cfg.n_ctx))] (of_Z (Z.of_N cfg.n_ctx)) float (@coer_refl float) coer_Z_float float_has_add float_has_sub float_has_mul float_has_div float_has_sqrt float_has_exp true (@all_tokens true)).

  Definition logits_all_tokens_concrete : PArray.concrete_tensor _ float
    := PArray.concretize logits_all_tokens.

(*
  Local Ltac set_step _ :=
    match goal with
    | [ H := context G[let s : ?T := ?v in @?f s] |- _ ]
      => lazymatch goal with
         | [ s' := v |- _ ]
           => let G' := context G[f s'] in
              change G' in (value of H)
         | _
           => let s' := fresh s in
              pose v as s';
              let G' := context G[f s'] in
              change G' in (value of H)
         end;
         cbv beta iota in H
    | [ H := context G[let s : ?T := ?v in _] |- _ ]
      => assert_fails is_var v;
         lazymatch goal with
         | [ s' := v |- _ ]
           => change v with s' in (value of H)
         | _
           => let s' := fresh s in
              pose v as s';
              change v with s' in (value of H)
         end;
         cbv beta iota in H
    | [ |- context G[let s : ?T := ?v in @?f s] ]
      => lazymatch goal with
         | [ s' := v |- _ ]
           => let G' := context G[f s'] in
              change G'
         | _
           => let s' := fresh s in
              pose v as s';
              let G' := context G[f s'] in
              change G'
         end;
         cbv beta iota
    | [ |- context G[let s : ?T := ?v in _] ]
      => assert_fails is_var v;
         lazymatch goal with
         | [ s' := v |- _ ]
           => change v with s'
         | _
           => let s' := fresh s in
              pose v as s';
              change v with s'
         end;
         cbv beta iota
    end.
  Local Ltac subst_cleanup _ :=
    repeat match goal with
      | [ H := ?v |- _ ] => is_var v; subst H
      | [ H := ?x, H' := ?y |- _ ] => constr_eq x y; change H' with H in *; clear H'
      end.
  Local Ltac lift_lets _ := repeat set_step (); subst_cleanup ().
  #[local] Ltac set_checkpoint _ :=
    repeat match goal with
      | [ H := context G[?x] |- _ ]
        => lazymatch x with PArray.checkpoint _ => idtac end;
           lazymatch (eval cbv delta [H] in H) with
           | x => fail
           | _ => idtac
           end;
           let x' := fresh "t" in
           pose x as x';
           let G' := context G[x'] in
           change G' in (value of H)
      | [ |- context G[?x] ]
        => lazymatch x with PArray.checkpoint _ => idtac end;
           let x' := fresh "t" in
           pose x as x';
           let G' := context G[x'] in
           change G'
      end.

  #[local] Ltac subst_local_cleanup _ :=
    repeat match goal with
      | [ H := [ _ ] : ?T |- _ ]
        => lazymatch T with
           | Shape _ => idtac
           | forall b, Shape _ => idtac
           | Slice.Concrete.Slice IndexType => idtac
           | IndexType => idtac
           | Slice.Slice ShapeType => idtac
           end;
           subst H
      | [ H := [ fun x => coer x ] : float -> float |- _ ] => cbv in H; subst H
      | [ H := [ coer point ] : float |- _ ] => cbv in H; subst H
      | [ H := [ coer_Z_float _ ] : float |- _ ] => cbv in H; subst H
      end;
    cbv beta iota in *.

  #[local] Ltac do_red _ :=
    cbv beta iota delta [repeat repeat' reduce_axis_m1 map map' reduce_axis_m1' reshape_app_combine broadcast broadcast' reshape_app_combine' RawIndex.uncurry_radd RawIndex.split_radd reshape_snoc_split reshape_app_split reshape_app_split' RawIndex.curry_radd RawIndex.combine_radd RawIndex.hd RawIndex.tl
                           Nat.radd
                           Classes.sqrt Classes.add Classes.sub Classes.opp Classes.mul Classes.div Classes.sqr Classes.one Classes.zero Classes.exp
                           Tensor.get Tensor.raw_get Slicing.SliceIndex.SliceIndexType.slice Slice.invert_index Slice.concretize PolymorphicOption.Option.sequence_return Slice.step Slice.start Slice.stop Slice.Concrete.length Slicing.SliceIndex.slice Slicing.FancyIndex.slice Slicing.FancyIndex.slice_ Slicing.FancyIndex.broadcast Slicing.FancyIndex.FancyIndexType.broadcast Slice.Concrete.normalize Slice.Concrete.step Slice.Concrete.stop Slice.Concrete.start
                           RawIndex.snoc RawIndex.nil
                           map_dep map2 map2' map3
                           Shape.tl Shape.hd Shape.snoc Shape.nil
      ] in *;
    cbn beta iota delta [fst snd Primitive.fst Primitive.snd] in *;
    lift_lets (); set_checkpoint (); subst_local_cleanup ().

  Derive logits_all_tokens_concrete_opt
    SuchThat (logits_all_tokens_concrete_opt = logits_all_tokens_concrete)
    As logits_all_tokens_concrete_opt_eq.
  Proof.
    Unshelve.
    2:{ pose proof cfg.blocks_params as blocks_params.
        destruct cfg.normalization_type as [nt|]; [ destruct nt | ].
        all: shelve. }
    cbv beta iota delta [logits_all_tokens_concrete logits_all_tokens
                           HookedTransformer.coer_blocks_params] in *;
      lift_lets (); set_checkpoint ().
    set (blocks_params' := cfg.blocks_params) in *.
    set (ln_final_w := cfg.ln_final_w) in *.
    set (ln_final_b := cfg.ln_final_b) in *.
    clearbody blocks_params' ln_final_w ln_final_b.
    assert_succeeds destruct cfg.normalization_type.
    cbv beta iota delta [HookedTransformer.HookedTransformer.logits HookedTransformer.HookedTransformer.ln_final coer_ln_tensor HookedTransformer.HookedTransformer.unembed HookedTransformer.Unembed.forward HookedTransformer.HookedTransformer.resid_postembed HookedTransformer.HookedTransformer.pos_embed HookedTransformer.HookedTransformer.embed HookedTransformer.Embed.forward HookedTransformer.PosEmbed.forward] in *;
      lift_lets (); set_checkpoint ().
    cbv beta iota delta [HookedTransformer.HookedTransformer.blocks_cps] in *;
      lift_lets (); set_checkpoint ().
    subst_local_cleanup ().
    lazymatch goal with
    | [ |- _ = ?concretize (List.fold_right ?k ?f ?ls ?resid) ]
      => let f' := open_constr:(_) in
         let ls' := open_constr:(_) in
         replace f with f'; [ | assert (forall x, f' x = f x) ];
         [ replace ls with ls'
         | .. ]
    end.
    2: { rewrite List.firstn_all.
         repeat match goal with H : _ |- _ => clear H end.
         cbv beta iota delta [HookedTransformer.HookedTransformer.blocks]; lift_lets (); set_checkpoint ().
         rewrite List.map_map.
         instantiate (1:=ltac:(destruct cfg.normalization_type as [nt|]; [ destruct nt | ])).
         destruct cfg.normalization_type as [nt|]; [ destruct nt | ].
         all: cbv beta iota zeta.
         all: cbv beta iota delta [TransformerBlock.attn_only_out]; lift_lets (); set_checkpoint ().
         all: match goal with
              | [ |- _ = List.map ?f _ ]
                => let f' := open_constr:(_) in
                   replace f with f';
                   [ shelve | assert (forall x y, f' x y = f x y); [ intros ?? | shelve ] ]
              end.
         all: lift_lets (); set_checkpoint ().
         all: cbv beta iota delta [TransformerBlock.ln1 LayerNorm.forward TransformerBlock.query_input TransformerBlock.key_input TransformerBlock.value_input TransformerBlock.add_head_dimension LayerNorm.scale LayerNorm.rescale LayerNorm.linpart LayerNorm.postrescale] in *; lift_lets (); set_checkpoint (); do_red ().
         all: cbv beta iota delta [Attention.attn_out Attention.z Attention.v Attention.pattern] in *; lift_lets (); set_checkpoint (); do_red ().
         all: cbv beta iota delta [HookedTransformer.Attention.masked_attn_scores HookedTransformer.Attention.attn_scores Attention.einsum_input Attention.q Attention.k] in *; lift_lets (); set_checkpoint (); do_red ().
         all: cbv [Attention.apply_causal_mask] in *; do_red (); cbv beta iota zeta in *; do_red ().
         all: cbv beta iota delta [ones bool_has_one tril bool_has_zero to_bool Classes.eqb Classes.neqb bool_has_eqb] in *; lift_lets (); set_checkpoint (); do_red ().
         Import Primitive (fst, snd).
         all: cbv beta iota delta [softmax_dim_m1] in *; lift_lets (); do_red ().
         all: cbv beta iota delta [Bool.where_ where_ float_has_mul tensor_add float_has_add tensor_mul tensor_div_by float_has_div float_has_exp float_has_sqrt tensor_sqrt float_has_sub] in *; do_red ().
         all: vm_compute coer_Z_float in *.
         all: repeat match goal with H : _ |- _ => clear H end.
         all: cbv beta iota delta [coer coer_Z_float] in *; do_red ().
          coer

         Set Printing All.
         cbv [Sint6
         SEt Printing
         vm_compute in mask.

         match goal with
         | [ H := context G[?x] |- _ ]
           => lazymatch x with PArray.checkpoint _ => idtac end;
              lazymatch (eval cbv delta [H] in H) with
              | x => fail
              | _ => idtac
              end;
              idtac H x
         end.
              pose x as x';
           let G' := context G[x'] in
           change G' in (value of H)
         end.
      | [ |- context G[?x] ]
        => lazymatch x with PArray.checkpoint _ => idtac end;
           let x' := fresh x in
           pose x as x';
           let G' := context G[x'] in
           change G'
      end.
         all: cbv beta iota delta [Attention.attn_out] in *; lift_lets ();
           do_red ().

i         all: cbv
           reshape_app_combine broadcast' reshape_app_combine' repeat' RawIndex.uncurry_radd RawIndex.split_radd Nat.radd] in *; lift_lets ().
         all: cbv
         2: {
         2: {
         destruct cfg.normalization_type as [nt|]; [ destruct nt | ].
    cbv beta iota delta [LayerNorm.forward LayerNorm.scale LayerNorm.rescale LayerNorm.postrescale LayerNorm.linpart] in *; lift_lets ().
    s := [cfg.d_vocab ^ cfg.n_ctx; cfg.n_ctx]%shape : Shape 2
         cbv [
    match goal with
    | [ |-
    repeat match goal with
           | [ H := coer point |- _ ] => vm_compute in H; subst H
           end.

    cbn beta iota delta [fst snd] in *.
    repeat cbv beta iota delta [Slice.Concrete.step Slice.Concrete.stop Slice.Concrete.start Slice.Concrete.base_len Slice.Concrete.raw_length PolymorphicOption.option_map Slice.norm_concretize PolymorphicOption.Option.sequence_return Slice.Concrete.normalize Slice.concretize Slice.Concrete.base_len Slice.start Slice.stop Slice.step Slice.Concrete.start Slice.Concrete.stop Slice.Concrete.step Slice.Concrete.base_len] in *; lift_lets ().
    cbv beta iota delta [Slice.Concrete.base_len Slice.Concrete.step Slice.Concrete.start] in *; lift_lets ().
    exfalso.
    clear -x0.
    lift_lets ().
    Set Printing Coercions.
    cbv beta iota delta [int_has_one] in *.

    Set Printing Coercions.
                           LayerNorm.forward LayerNorm.scale LayerNorm.rescale LayerNorm.postrescale LayerNorm.linpart] in *; lift_lets ().
    destruct cfg.normalization_type.
    set_step ().
  Definition logi
*)
End Model.

Module Type ModelSig (cfg : Config) := Nop <+ Model cfg.
