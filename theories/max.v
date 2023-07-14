From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp Require Import max_parameters.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Import ListNotations.
Local Open Scope raw_tensor_scope.

(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

(** Coq infra *)
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.

(** Hyperparameters *)
Definition N_LAYERS : nat := 1.
Definition N_HEADS : nat := 1.
Definition D_MODEL : nat := 32.
Definition D_HEAD : nat := 32.
(*Definition D_MLP = None*)

Definition D_VOCAB : nat := 64.

Notation tensor_of_list ls := (Tensor.PArray.abstract (Tensor.PArray.concretize (Tensor.of_list ls))) (only parsing).
Definition W_E : tensor _ _ := Eval cbv in tensor_of_list max_parameters.W_E.
Definition W_pos : tensor _ _ := Eval cbv in tensor_of_list max_parameters.W_pos.
Definition L0_attn_W_Q : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_Q.
Definition L0_attn_W_K : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_K.
Definition L0_attn_W_V : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_V.
Definition L0_attn_W_O : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_O.
Definition L0_attn_b_Q : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_Q.
Definition L0_attn_b_K : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_K.
Definition L0_attn_b_V : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_V.
Definition L0_attn_b_O : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_O.
Definition L0_ln1_b : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_ln1_b.
Definition L0_ln1_w : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_ln1_w.
Definition ln_final_b : tensor _ _ := Eval cbv in tensor_of_list max_parameters.ln_final_b.
Definition ln_final_w : tensor _ _ := Eval cbv in tensor_of_list max_parameters.ln_final_w.
Definition W_U : tensor _ _ := Eval cbv in tensor_of_list max_parameters.W_U.
Definition b_U : tensor _ _ := Eval cbv in tensor_of_list max_parameters.b_U.

#[local] Notation FLOAT := float (only parsing). (* or Q *)

Definition embed {r} {s : Shape r} (tokens : tensor IndexType s) : tensor FLOAT (s ::' Shape.tl (shape_of W_E))
  := (W_E.[tokens, :])%fancy_raw_tensor.

Definition pos_embed {r} {s : Shape (S r)} (tokens : tensor int s)
  (tokens_length := Shape.tl s) (* s[-1] *)
  (batch := Shape.droplastn 1 s) (* s[:-1] *)
  (d_model := Shape.tl (shape_of W_pos)) (* s[-1] *)
  : tensor FLOAT (batch ++' [tokens_length] ::' d_model)
  := repeat (W_pos.[:tokens_length, :]) batch.

Section layernorm.
  Context {r A} {s : Shape r} {d_model}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A} {default : pointed A}
    (eps : A) (w b : tensor A [d_model]).

  Definition layernorm_linpart (x : tensor A (s ::' d_model))
    : tensor A (s ::' d_model)
    := (x - reduce_axis_m1 (keepdim:=true) mean x)%core.

  Definition layernorm_scale (x : tensor A (s ::' d_model))
    : tensor A (s ::' 1)
    := (√(reduce_axis_m1 (keepdim:=true) mean (x ²) + broadcast' eps))%core.

  Definition layernorm_rescale (x : tensor A (s ::' d_model))
                               (scale : tensor A (s ::' 1))
    : tensor A (s ::' d_model)
    := (x / scale)%core.

  Definition layernorm_postrescale (x : tensor A (s ::' d_model))
    : tensor A (s ::' d_model)
    := (x * broadcast w + broadcast b)%core.

  Definition layernorm (x : tensor A (s ::' d_model))
    : tensor A (s ::' d_model)
    := let x := layernorm_linpart x in
       let scale := layernorm_scale x in
       let x := layernorm_rescale x scale in
       PArray.checkpoint (layernorm_postrescale x).
End layernorm.

Section ln.
  Context {r} {s : Shape r}
    (d_model := Uint63.of_Z cfg.d_model)
    (eps := cfg.eps).

  Section ln1.
    Context (w := L0_ln1_w) (b := L0_ln1_b).

    Definition ln1_linpart (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_linpart x.
    Definition ln1_scale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln1_rescale (x : tensor FLOAT (s ::' d_model)) (scale : tensor FLOAT (s ::' 1)) : tensor FLOAT (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln1_postrescale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln1 (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm eps w b x.
  End ln1.

  Section ln_final.
    Context (w := ln_final_w) (b := ln_final_b).

    Definition ln_final_linpart (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_linpart x.
    Definition ln_final_scale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln_final_rescale (x : tensor FLOAT (s ::' d_model)) (scale : tensor FLOAT (s ::' 1)) : tensor FLOAT (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln_final_postrescale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln_final (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm eps w b x.
  End ln_final.
End ln.

Section Attention.
  Context {A r} {batch : Shape r}
    {sqrtA : has_sqrt A} {coerZ : has_coer Z A} {addA : has_add A} {zeroA : has_zero A} {mulA : has_mul A} {divA : has_div A} {expA : has_exp A}
    {pos n_heads d_model d_head} {n_ctx:N}
    {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
    (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
    (b_Q b_K b_V : tensor A [n_heads; d_head])
    (b_O : tensor A [d_model])
    (IGNORE : A)
    (n_ctx' : int := Uint63.of_Z n_ctx)
    (attn_scale : A := √(coer (Uint63.to_Z d_head)))
    (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape)
    (query_input key_input value_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
    (mask : tensor bool [n_ctx'; n_ctx'] := to_bool (tril (A:=bool) (ones [n_ctx'; n_ctx']))).

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
    := (einsum_input query_input W_Q + broadcast b_Q)%core.
  Definition k : tensor A (batch ++' [pos; n_heads; d_head])
    := (einsum_input key_input W_K + broadcast b_K)%core.
  Definition v : tensor A (batch ++' [pos; n_heads; d_head])
    := (einsum_input value_input W_V + broadcast b_V)%core.

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
        qk / broadcast' attn_scale)%core.

  Definition apply_causal_mask (attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos))
    : tensor A (batch ::' n_heads ::' pos ::' pos)
    := Tensor.map'
         (fun attn_scores : tensor A [pos; pos]
          => Tensor.where_ mask.[:pos,:pos] attn_scores (broadcast' IGNORE))
         attn_scores.

  Definition masked_attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
    := apply_causal_mask attn_scores.

  Definition pattern : tensor A (batch ::' n_heads ::' pos ::' pos)
    := softmax_dim_m1 masked_attn_scores.

  Definition z : tensor A (batch ::' pos ::' n_heads ::' d_head)
    := Tensor.map2'
         (fun (v : tensor A [pos; n_heads; d_head])
              (pattern : tensor A [n_heads; pos; pos])
          => weaksauce_einsum {{{ {{  key_pos head_index d_head,
                         head_index query_pos key_pos ->
                         query_pos head_index d_head }}
                     , v
                     , pattern }}}
            : tensor A [pos; n_heads; d_head])
         v
         pattern.

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
        out + broadcast b_O)%core.
End Attention.

Section Attention0.
  Context {r} {batch : Shape r}
    (IGNORE := -1e5)
    (query_input key_input value_input : tensor FLOAT (batch ++' [cfg.n_ctx; cfg.d_model])).

  Definition L0_attn_out : tensor FLOAT (batch ::' cfg.n_ctx ::' cfg.d_model)
    := attn_out
         (n_ctx:=cfg.n_ctx)
         L0_attn_W_Q L0_attn_W_K L0_attn_W_V L0_attn_W_O
         L0_attn_b_Q L0_attn_b_K L0_attn_b_V L0_attn_b_O
         IGNORE
         query_input key_input value_input.
End Attention0.

Section TransformerBlock.
  Context {A r} {batch : Shape r}
    {zeroA : has_zero A} {coerZ : has_coer Z A}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
    {sqrtA : has_sqrt A} {expA : has_exp A}
    {default : pointed A}
    {pos n_heads d_model d_head} {n_ctx:N}
    {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
    (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
    (b_Q b_K b_V : tensor A [n_heads; d_head])
    (b_O : tensor A [d_model])
    (IGNORE : A)
    (eps : A)
    (ln1_w ln1_b ln2_w ln2_b : tensor A [d_model])
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

  Definition transformer_block_ln1 {r} {s : Shape r} (t : tensor A (s ::' d_model)) : tensor A (s ::' d_model)
    := layernorm eps ln1_w ln1_b t.
  Definition transformer_block_ln2 {r} {s : Shape r} (t : tensor A (s ::' d_model)) : tensor A (s ::' d_model)
    := layernorm eps ln2_w ln2_b t.

  Definition transformer_block_attn_only_out : tensor A (batch ++ [pos; d_model])
    := (let attn_out : tensor A (batch ++ [pos; d_model])
          := attn_out
               (n_ctx:=n_ctx)
               W_Q W_K W_V W_O
               b_Q b_K b_V b_O
               IGNORE
               (transformer_block_ln1 query_input)
               (transformer_block_ln1 key_input)
               (transformer_block_ln1 value_input) in
        resid_pre + attn_out)%core.
End TransformerBlock.

Section L0.
  Context {r} {batch : Shape r}
    (IGNORE := -1e5)
    (residual : tensor FLOAT (batch ++' [cfg.n_ctx; cfg.d_model])).

  Definition block0 : tensor FLOAT (batch ::' cfg.n_ctx ::' cfg.d_model)
    := transformer_block_attn_only_out
         (n_ctx:=cfg.n_ctx)
         L0_attn_W_Q L0_attn_W_K L0_attn_W_V L0_attn_W_O
         L0_attn_b_Q L0_attn_b_K L0_attn_b_V L0_attn_b_O
         IGNORE cfg.eps
         L0_ln1_w L0_ln1_b
         residual.
End L0.

Definition unembed {r} {s : Shape r} (residual : tensor FLOAT (s ::' cfg.n_ctx ::' cfg.d_model)) : tensor FLOAT (s ::' cfg.n_ctx ::' cfg.d_vocab_out)
  := (Tensor.map'
        (fun residual : tensor FLOAT [cfg.n_ctx; cfg.d_model]
         => weaksauce_einsum {{{ {{ pos d_model, d_model vocab -> pos vocab }}
                    , residual
                    , W_U }}}
           : tensor FLOAT [cfg.n_ctx; cfg.d_vocab_out])
        residual
      + broadcast b_U)%core.

Definition logits {r} {batch : Shape r} (tokens : tensor IndexType (batch ::' cfg.n_ctx))
  : tensor FLOAT (batch ::' cfg.n_ctx ::' cfg.d_vocab_out)
  := (let embed := embed tokens in
      let pos_embed := pos_embed tokens in
      let resid_shape := (batch ::' cfg.n_ctx ::' cfg.d_model)%shape in
      let residual : tensor FLOAT resid_shape := PArray.checkpoint (embed + pos_embed) in
      let residual : tensor FLOAT resid_shape := PArray.checkpoint (block0 residual) in
      let residual : tensor FLOAT resid_shape := PArray.checkpoint (ln_final residual) in
      let logits                          := PArray.checkpoint (unembed residual) in
      logits)%core.

Goal True.
  pose (PArray.concretize (logits (tensor_of_list [0; 1]%uint63))) as v.
  cbv beta delta [logits PArray.checkpoint] in v.
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.
  repeat match (eval cbv delta [v] in v) with
         | context V[let x := ?val in @?k x]
           => lazymatch val with
              | context[PArray.concretize ?val] => fail 1 val
              | _ => idtac
              end;
              let V := context V[match val with x => k x end] in
              change V in (value of v); cbv beta in v
         end.
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.
  repeat match (eval cbv delta [v] in v) with
         | context V[let x := ?val in @?k x]
           => lazymatch val with
              | context[PArray.concretize ?val] => fail 1 val
              | _ => idtac
              end;
              let V := context V[match val with x => k x end] in
              change V in (value of v); cbv beta in v
         end.
  set (k := PArray.concretize _) in (value of v) at 2.
  lazymatch (eval cbv delta [k] in k) with
  | PArray.concretize (block0 ?x)
    => set (k0 := x) in (value of k); set (k1 := block0 k0) in (value of k)
  end.
  cbv beta delta [block0] in k1.
  cbv beta zeta in k1.
  cbv beta delta [transformer_block_attn_only_out] in k1.
  cbv beta zeta in k1.
  lazymatch (eval cbv delta [k1] in k1) with
  | (_ + ?x)%core
    => set (k2 := x) in (value of k1)
  end.
  clear -k2.
  cbv beta delta [attn_out] in k2.
  cbv beta zeta in k2.
  lazymatch (eval cbv delta [k2] in k2) with
  | (map' ?f ?x + ?y)%core
    => set (k3 := x) in (value of k2);
       set (k4 := y) in (value of k2)
  end.
  clear -k3.
  cbv beta delta [z] in k3.
  cbv beta zeta in k3.
  let k := k3 in
  lazymatch (eval cbv delta [k] in k) with
  | (map2' ?f ?x ?y)%core
    => let k1 := fresh "k" in
       let k2 := fresh "k" in
       set (k1 := x) in (value of k);
       set (k2 := y) in (value of k)
  end.
  cbv beta delta [v] in *.
  cbv beta zeta in k1.
  cbv beta delta [value_input key_input query_input add_head_dimension] in *.
  cbv beta zeta iota in k1, k2.
  cbv beta delta [transformer_block_ln1] in *.
  cbv beta zeta iota in k1, k2.
  set (lnv := layernorm _ _ _ _) in *.
  cbv beta delta [layernorm] in *.
  cbv beta iota zeta in lnv.
  cbv beta delta [PArray.checkpoint] in lnv.
  set (k_tmp := PArray.concretize _) in (value of lnv).
  Time vm_compute in k_tmp.
  subst k_tmp.
  cbv [einsum_input] in k1.
  FIXME EINSUM WRONG
               (*
  cbv beta delta [Shape.hd Shape.nil Shape.tl Shape.snoc fst snd Shape.cons Shape.app Nat.radd Tensor.raw_get] in k1.

  cbv beta iota zeta in k1.
  Set Printing All.

  cbn [fst snd] in k1.


  pose
  Timeout 5 Compute PArray.concretize lnv.
  pose (lnvc := PArray.concretize lnv).
  vm_compute in lnvc.
  vm_compute in lnv.
  clear -lnv.
  set (lnlv := layernorm_linpart _) in *.
  cbv beta delta [layernorm_linpart] in *.
  set (mv := reduce_axis_m1 _ _) in *.
  cbv beta delta [reduce_axis_m1] in *.
  clear -mv.
  cbv beta iota in mv.
  set (mv' := reduce_axis_m1' _ _) in *.
  clear -mv'.
  cbv beta delta [reduce_axis_m1'] in *.
  vm_compute Shape.tl in mv'.
  set (k_tmp := reshape_snoc_split) in *.
  cbv in k_tmp; subst k_tmp.
  cbv beta iota in mv'.
  cbv beta delta [map] in mv'.
  pose (PArray.concretize mv') as mv'c.
  vm_compute in mv'c.
  cbv [PArray.concretize Shape.snoc init_default] in mv'c.
  set (k_tmp := _ <=? max_length) in *.
  vm_compute in k_tmp; subst k_tmp; cbv beta iota in mv'c.
  vm_compute make in mv'c.
  cbv -[mv'] in mv'c.
  subst mv'.
  cbv beta iota zeta in mv'c.
  cbv [mean] in mv'c.
  vm_compute inject_Z_coer in mv'c.
  cbv [raw_get] in mv'c.
  cbv [RawIndex.snoc RawIndex.nil] in mv'c.
  cbv [sum] in mv'c.
  cbv [map_reduce] in mv'c.
  Timeout 5 cbv -[add Q_has_add k0 Qdiv] in mv'c.
  Time repeat (time (set (k0v := k0 _) in (value of mv'c) at 1;
                     timeout 5 vm_compute in k0v; subst k0v)).
  Time vm_compute in mv'c.
.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  do 10 (set (k0v := k0 _) in (value of mv'c) at 1;
          timeout 5 vm_compute in k0v; subst k0v).
  do 10 (set (k0v := k0 _) in (value of mv'c) at 1;
          timeout 5 vm_compute in k0v; subst k0v).
  do 10 (set (k0v := k0 _) in (value of mv'c) at 1;
          timeout 5 vm_compute in k0v; subst k0v).
  clear -k0v.
  set (val := (_ + _)%core) in (value of k0) at 1.
  cbv [ltb] in *.
  vm_compute RawIndex.repeat in k0.
  vm_compute Shape.snoc in k0.
  vm_compute PArray.abstract in k0.
  clearbody val.
  vm_compute in val.
  vm_compute RawIndex in *.

  vm_compute in k0v.
(*
  generalize dependent (_ + _)
  cbv in k0.
  cbv [RawIndex
  vm_compute in k0v.
  cbv in k0v.
  Timeout 5 vm_compute in k0v.
  vm_compute in k0v.
  subst k0.
  Set Printing All.
  cbv in mv'c.
  vm_compute

  cbv -
  cbv [reshape_snoc_split] in mv'.
  cbv [reshape_snoc_split] in mv'.

  Timeout 5 Compute PArray.concretize mv'.
  cbv beta delta [reduce_axis_m1'] in *.

  Timeout 5 Compute PArray.concretize mv.
  let k := k1 in
  let k1 := fresh "k" in
  let k2 := fresh "k" in
  lazymatch (eval cbv delta [k] in k) with
  | (map2' ?f ?x ?y)%core
    => set (k1 := x) in (value of k);
       set (k2 := y) in (value of k)
  end.



  set (k2 := attn_out _ _ _
  Time native_compute in k.
  Time vm_compute in k.
  subst k.

  cbv
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.

Time Timeout 5 Compute PArray.concretize (logits (tensor_of_list [0; 1]%uint63)).
Compute PArray.concrete_tensor

Compute PArray.concretize (embed (tensor_of_list [0; 1]%uint63)).
Compute PArray.concretize (pos_embed (tensor_of_list [[0; 1]]%uint63) : tensor FLOAT [1; cfg.n_ctx; cfg.d_model]).
*)
