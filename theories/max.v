From Coq Require Import Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp Require Import max_parameters Tensor.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Local Open Scope Q_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Import ListNotations.
(* Should use IEEE 754 floats from flocq, but let's use rationals for now for ease of linearity, proving, etc *)
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

Definition embed {r} {s : Shape r} (tokens : tensor int s) : tensor Q (s ::' Shape.tl (shape_of W_E))
  := Tensor.map (fun i => W_E.[i]) tokens.

Definition pos_embed {r} {s : Shape (S r)} (tokens : tensor int s)
  (tokens_length := stl s) (* s[-1] *)
  (batch := sdroplastn 1 s) (* s[:-1] *)
  (d_model := stl (shape_of W_pos)) (* s[-1] *)
  : tensor Q (batch ++' [tokens_length] ::' d_model)
  := repeat (W_pos.[[:tokens_length]]) batch.

Section layernorm.
  Context {r A} {s : Shape r} {d_model}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A}
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
       layernorm_postrescale x.
End layernorm.

Section ln.
  Context {r} {s : Shape r}
    (d_model := Uint63.of_Z cfg.d_model)
    (eps := cfg.eps).

  Section ln1.
    Context (w := L0_ln1_w) (b := L0_ln1_b).

    Definition ln1_linpart (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm_linpart x.
    Definition ln1_scale (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln1_rescale (x : tensor Q (s ::' d_model)) (scale : tensor Q (s ::' 1)) : tensor Q (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln1_postrescale (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln1 (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm eps w b x.
  End ln1.

  Section ln_final.
    Context (w := ln_final_w) (b := ln_final_b).

    Definition ln_final_linpart (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm_linpart x.
    Definition ln_final_scale (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln_final_rescale (x : tensor Q (s ::' d_model)) (scale : tensor Q (s ::' 1)) : tensor Q (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln_final_postrescale (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln_final (x : tensor Q (s ::' d_model)) : tensor Q (s ::' d_model)
      := layernorm eps w b x.
  End ln_final.
End ln.

From Ltac2 Require Ltac2 Constr List Ident Fresh Printf.
From NeuralNetInterp.Util.Tactics2 Require Constr FixNotationsForPerformance Constr.Unsafe.MakeAbbreviations.
Module Einsum.
  Import Ltac2.
  Import FixNotationsForPerformance MakeAbbreviations Printf.

  Module Import Internals.
    Ltac2 rec decompose_lam_idents (c : constr) : ident list
      := match Constr.Unsafe.kind_nocast c with
         | Constr.Unsafe.Lambda b body
           => let rest := decompose_lam_idents body in
              match Constr.Binder.name b with
              | Some n => n :: rest
              | None => rest
              end
         | _ => []
         end.

    Ltac2 rec dedup (ids : ident list) : ident list
      := match ids with
         | [] => []
         | id :: ids
           => let ids := dedup ids in
              if List.mem Ident.equal id ids
              then ids
              else id :: ids
         end.

    Ltac2 set_diff (ids1 : ident list) (ids2 : ident list) : ident list
      := let (overlap, diff) := List.partition (fun a => List.mem Ident.equal a ids2) ids1 in
         diff.

    Ltac2 rec get_body (at_head : bool) (c : constr) :=
      match Constr.Unsafe.kind_nocast c with
      | Constr.Unsafe.Var v
        => let r := Std.VarRef v in
           eval cbv delta [$r] in c
      | Constr.Unsafe.Constant n _
        => let r := Std.ConstRef n in
           eval cbv delta [$r] in c
      | Constr.Unsafe.App f args
        => if at_head
           then let f := get_body at_head f in
                Constr.Unsafe.make (Constr.Unsafe.App f args)
           else c
      | _ => c
      end.

    Ltac2 rec make_einsum_noclose_nolift (sum_to : constr (* max *) -> constr (* body *) -> constr) (intT : constr) (ids : ident list) (body : constr) : constr
      := match ids with
         | [] => body
         | id :: ids
           => let body := make_einsum_noclose_nolift sum_to intT ids body in
              let body := mkLambda (Constr.Binder.make (Some id) intT) body in
              sum_to (Constr.Unsafe.make (Constr.Unsafe.Var id)) body
         end.

    Local Notation try_tc := (ltac:(try typeclasses eauto)) (only parsing).
    Ltac2 make_einsum (src_ids : ident list list) (dest_ids : ident list) (body : constr) : constr
      := let ty := Constr.type body in
         let src_ids := List.flat_map (fun x => x) src_ids in
         let einsum_ids := dedup (set_diff src_ids dest_ids) in
         let einsum_ids_rev := List.rev einsum_ids in
         let nbinders := List.length einsum_ids in
         let body := Constr.Unsafe.liftn nbinders 1 body in
         let body := Constr.Unsafe.closenl einsum_ids_rev 1 body in
         let start := '(0%uint63) in
         let step := '(1%uint63) in
         let sum := '(@Wf_Uint63.sum $ty try_tc try_tc) in
         let sum_to stop body := mkApp sum [start; stop; step; body] in
         make_einsum_noclose_nolift sum_to 'int einsum_ids body.

    Ltac subst_type_lets_in_goal _ :=
      repeat match goal with
        | [ H := [ _ ] : Type |- _ ] => match goal with |- context[H] => idtac end; subst H
        end.
  End Internals.

  Local Notation try_tc := (ltac:(try typeclasses eauto)) (only parsing).

  (* Kludge around COQBUG(https://github.com/coq/coq/issues/17833#issuecomment-1627483008) *)
  Local Notation indirect_einsum tensor_value kidxs iidxs jidxs
    := ltac2:(let get_body v := get_body false (Constr.pretype v) in
              let get_idents v := decompose_lam_idents (get_body v) in
              let t := get_body tensor_value in
              let src_ids := List.map get_idents [iidxs; jidxs] in
              let dest_ids := get_idents kidxs in
              let t := make_einsum src_ids dest_ids t in
              exact $t)
               (only parsing).

  #[export] Hint Extern 1 => progress subst_type_lets_in_goal () : typeclass_instances.
  Declare Custom Entry einsum_args.
  Notation "{{{ {{ i1 .. i_ , j1 .. j_ -> k1 .. k_ }} , t1 , t2 }}}"
      := (match t1%tensor, t2%tensor, _ as A, _ as B, _ as C, _ as r1, _ as r2, _ as r3, _ as s1, _ as s2, _ as s3 return @tensor _ C s3 with
          | t1', t2', A, B, C, r1, r2, r3, s1, s2, s3
            => match t1' : @tensor r1 A s1, t2' : @tensor r2 B s2 return @tensor r3 C s3 with
               | t1', t2'
                 => match ((fun i1 => .. ((fun i_ => I) : True -> _) ..) : True -> _),
                      ((fun j1 => .. ((fun j_ => I) : True -> _) ..) : True -> _),
                      ((fun k1 => .. ((fun k_ => I) : True -> _) ..) : True -> _),
                      (* for typing *)
                      ((@reshape_S_fun_combine
                          True True _
                          (fun i1 => .. (@reshape_S_fun_combine
                                           True True _
                                           (fun i_ => (I:tensor_fun_of_rank True True 0))) ..)) : tensor_fun_of_rank True True r1),
                      ((@reshape_S_fun_combine
                          True True _
                          (fun j1 => .. (@reshape_S_fun_combine
                                           True True _
                                           (fun j_ => (I:tensor_fun_of_rank True True 0))) ..)) : tensor_fun_of_rank True True r2),
                      ((@reshape_S_fun_combine
                          True True _
                          (fun k1 => .. (@reshape_S_fun_combine
                                           True True _
                                           (fun k_ => (I:tensor_fun_of_rank True True 0))) ..)) : tensor_fun_of_rank True True r3)
                          return _
                    with
                    | __EINSUM_IIDXS, __EINSUM_JIDXS, __EINSUM_KIDXS
                      , _, _, _
                      => @with_shape
                           r1 (tensor C s3) s1
                           (fun i1
                            => .. (fun i_
                                   => @with_shape
                                        r2 (tensor C s3) s2
                                        (fun j1
                                         => .. (fun j_
                                                => @init
                                                     r3 C s3
                                                     (reshape_S_fun_combine
                                                        (I:=int)
                                                        (fun k1
                                                         => .. (reshape_S_fun_combine
                                                                  (I:=int)
                                                                  (fun k_
                                                                   => match @Arith.Classes.mul
                                                                              A B C try_tc
                                                                              (tensor_get t1' (pair .. (pair tt i1) .. i_))
                                                                              (tensor_get t2' (pair .. (pair tt j1) .. j_))
                                                                            return @tensor_fun_of_rank int C 0
                                                                      with
                                                                      | __EINSUM_TENSOR_VALUE
                                                                        => indirect_einsum
                                                                             __EINSUM_TENSOR_VALUE __EINSUM_KIDXS __EINSUM_IIDXS __EINSUM_JIDXS
                                                                      end
                                                              )) ..
                                                     ))
                                              ) ..
                                 )) ..
                           )
                    end
               end
          end)
           (only parsing, in custom einsum_args at level 0, i1 binder, i_ binder, j1 binder, j_ binder, k1 binder, k_ binder, t1 constr at level 10, t2 constr at level 10).

  Notation "'weaksauce_einsum' x"
    := (match x return _ with
        | y => ltac2:(let y := get_body false &y in
                      let z := (eval cbv beta iota delta [reshape_S_fun_combine reshape_app_combine_gen with_shape] in y) in
                      let z := (eval cbn beta iota delta [Nat.radd] in z) in
                     exact $z)
        end)
         (x custom einsum_args at level 10, at level 10, only parsing).
  (*
  Set Printing Implicit.
  Check (weaksauce_einsum {{{ {{ query_pos head_index d_head,
          key_pos head_index d_head
          -> head_index query_pos key_pos }}, (_:tensor _ [2;1;5]), (_:tensor _ [2;1;5]) }}} : tensor _ [1; 2; 2]).
   *)
End Einsum.
Import Einsum.
Section Attention.
  Context {A r} {batch : Shape r}
    {sqrtA : has_sqrt A} {coerZ : has_coer Z A} {addA : has_add A} {zeroA : has_zero A} {mulA : has_mul A} {divA : has_div A}
    {pos n_heads d_model d_head} {n_ctx:N}
    {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
    (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
    (b_Q b_K b_V : tensor A [n_heads; d_head])
    (b_O : tensor A [d_model])
    (IGNORE : A)
    (n_ctx' : int := Uint63.of_Z n_ctx)
    (attn_scale : A := √(coer (Uint63.to_Z d_head)))
    (maybe_n_heads := fun b : bool => if b return Shape (if b then _ else _) then [n_heads] else [])
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
    := (let input : tensor (tensor A _) (batch ::' pos) := reshape_app_split input in
        let W : tensor (tensor (tensor A [d_head]) [d_model]) [n_heads] := W in
        let f (input : tensor A [d_model]) (W : tensor (tensor A [d_head]) [d_model]) : tensor A [d_head]
          := let input_W : tensor (tensor A [d_head]) [d_model] := tensor_map2 (fun i w => broadcast' i * w)%core input W in
             reduce_axis_m1 (keepdim:=false) PArray.sum (input_W:tensor A [d_model; d_head]) in
        tensor_map
          (if use_split_qkv_input return tensor (tensor A [d_model]) (maybe_n_heads use_split_qkv_input) -> tensor (tensor A [d_head]) [n_heads]
           then fun input
                => tensor_map2 f input W
           else fun input : tensor A [d_model]
                => tensor_map (f input) W)
          input).

  Definition q : tensor A ((batch ::' pos) ++' [n_heads; d_head])
    := (einsum_input query_input W_Q + broadcast b_Q)%core.
  Definition k : tensor A ((batch ::' pos) ++' [n_heads; d_head])
    := (einsum_input query_input W_K + broadcast b_K)%core.
  Definition v : tensor A ((batch ::' pos) ++' [n_heads; d_head])
    := (einsum_input query_input W_V + broadcast b_V)%core.

  Definition attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
    := (let q : tensor (tensor A [pos; n_heads; d_head]) batch := q in
        let k : tensor (tensor A [pos; n_heads; d_head]) batch := k in
        let qk : tensor (tensor A [n_heads; pos; pos]) batch
          := tensor_map2
               (fun q k : tensor A [pos; n_heads; d_head]
                => weaksauce_einsum
                     {{{ {{ query_pos head_index d_head ,
                               key_pos head_index d_head
                               -> head_index query_pos key_pos }}
                           , q
                           , k }}})
               q
               k in
        let qk : tensor A (batch ::' n_heads ::' pos ::' pos) := qk in
        qk / broadcast' attn_scale)%core.


  (*HERE*)
End Attention.
Eval cbv in embed (tensor_of_list [0; 1]%uint63).
Eval cbv in pos_embed (tensor_of_list [[0; 1]]%uint63).
