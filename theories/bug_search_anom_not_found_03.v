(* -*- mode: coq; coq-prog-args: ("-emacs" "-q" "-w" "+implicit-core-hint-db,+implicits-in-term,+non-reversible-notation,+deprecated-intros-until-0,+deprecated-focus,+unused-intro-pattern,+variable-collision,+unexpected-implicit-declaration,+omega-is-deprecated,+deprecated-instantiate-syntax,+non-recursive,+undeclared-scope,+deprecated-hint-rewrite-without-locality,+deprecated-hint-without-locality,+deprecated-instance-without-locality,+deprecated-typeclasses-transparency-without-locality,-ltac2-missing-notation-var,unsupported-attributes" "-w" "-deprecated-native-compiler-option" "-native-compiler" "no" "-R" "/github/workspace/neural-net-coq-interp/theories" "NeuralNetInterp" "-Q" "/github/workspace/cwd" "Top" "-Q" "/home/coq/.opam/4.13.1+flambda/lib/coq/user-contrib/Bignums" "Bignums" "-Q" "/home/coq/.opam/4.13.1+flambda/lib/coq/user-contrib/Ltac2" "Ltac2" "-top" "NeuralNetInterp.bug_search_anom_not_found_03") -*- *)
(* File reduced by coq-bug-minimizer from original input, then from 220 lines to 39 lines, then from 52 lines to 522 lines, then from 524 lines to 229 lines, then from 699 lines to 216 lines, then from 228 lines to 118 lines, then from 131 lines to 665 lines, then from 670 lines to 135 lines, then from 148 lines to 1326 lines, then from 1331 lines to 173 lines, then from 186 lines to 747 lines, then from 752 lines to 498 lines, then from 511 lines to 883 lines, then from 887 lines to 744 lines, then from 757 lines to 844 lines, then from 849 lines to 749 lines, then from 762 lines to 829 lines, then from 834 lines to 761 lines, then from 774 lines to 850 lines, then from 855 lines to 774 lines, then from 787 lines to 837 lines, then from 842 lines to 790 lines, then from 803 lines to 1422 lines, then from 1427 lines to 811 lines, then from 824 lines to 1082 lines *)
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
Require NeuralNetInterp.Torch.Slicing.
Require NeuralNetInterp.TransformerLens.HookedTransformer.Config.
Require Ltac2.Constr.
Require Ltac2.Printf.
Require Ltac2.List.
Require Ltac2.Std.
Require Ltac2.Pattern.
Require Ltac2.Init.
Require Ltac2.Bool.
Require Ltac2.Int.
Require Ltac2.Message.
Require Ltac2.Control.

Require Ltac2.Array.
Axiom proof_admitted : False.
Tactic Notation "admit" := abstract case proof_admitted.
Module Export Notations.

Import Ltac2.Init.

Ltac2 Notation "lazy_match!" t(tactic(6)) "with" m(constr_matching) "end" :=
  Pattern.lazy_match0 t m.

Ltac2 exact0 ev c :=
  Control.enter (fun _ =>
    match ev with
    | true =>
      let c := c () in
      Control.refine (fun _ => c)
    | false =>
      Control.with_holes c (fun c => Control.refine (fun _ => c))
    end
  ).

Ltac2 Notation "exact" c(thunk(open_constr)) := exact0 false c.

End Notations.
Module Ltac2_DOT_Ltac2_WRAPPED.
Module Export Ltac2.

Export Ltac2.Init.

End Ltac2.

End Ltac2_DOT_Ltac2_WRAPPED.
Module Export Ltac2.
Module Ltac2.
Include Ltac2_DOT_Ltac2_WRAPPED.Ltac2.
End Ltac2.
Module Export Constr.
Import Ltac2.Ltac2.

Module Export Unsafe.
  Export Ltac2.Constr.Unsafe.
  Ltac2 rec kind_nocast (c : constr)
    := let k := kind c in
       match k with
       | Cast c _ _ => kind_nocast c
       | _ => k
       end.
End Unsafe.

End Constr.
Module Export MakeAbbreviations.
Import Ltac2.Ltac2.

Ltac2 mkApp (f : constr) (args : constr list) :=
  make (App f (Array.of_list args)).
Ltac2 mkLambda b (body : constr) :=
  make (Lambda b body).
Ltac2 mkRel (i : int) :=
  make (Rel i).
Ltac2 mkVar (i : ident) :=
  make (Var i).

End MakeAbbreviations.

Ltac2 Notation "eval" "cbv" s(strategy) "in" c(tactic(6)) :=
  Std.eval_cbv s c.

Ltac2 Notation "eval" "cbn" s(strategy) "in" c(tactic(6)) :=
  Std.eval_cbn s c.
Module Export Einsum.
Import Coq.Numbers.Cyclic.Int63.Sint63.
Import Coq.Unicode.Utf8.
Import NeuralNetInterp.Torch.Tensor.
Import NeuralNetInterp.Util.Wf_Uint63.
Export NeuralNetInterp.Util.Arith.Classes.

Import Ltac2.Ltac2.

Ltac2 mutable debug () := false.

Module Import Internals.
  Ltac2 debug_printf fmt := if debug () then Printf.printf fmt else Message.Format.kfprintf (fun x => ()) fmt.

  Ltac2 Notation "debug_printf" fmt(format) := debug_printf fmt.

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

  Ltac2 shape_to_list (c : constr) : constr list
    := let rec aux c acc
         := lazy_match! c with
            | Shape.nil => acc
            | Shape.snoc ?cs ?c
              => aux cs (c :: acc)
            end in
       aux c [].

  Ltac2 ident_of_constr (c : constr) : ident option
    := match Constr.Unsafe.kind_nocast c with
       | Constr.Unsafe.Var v => Some v
       | _ => None
       end.

  Ltac2 toplevel_rels (c : constr) : int list
    := let rec aux (acc : int list) (c : constr)
         := match Constr.Unsafe.kind_nocast c with
            | Constr.Unsafe.Rel i => i :: acc
            | Constr.Unsafe.App f args
              => let acc := aux acc f in
                 Array.fold_right aux acc args
            | _ => acc
            end in
       List.sort_uniq Int.compare (aux [] c).

  Local Notation try_tc := (ltac:(try typeclasses eauto)) (only parsing).

  Ltac2 insert_all_einsums (sum_to : constr -> constr -> constr) (names : ident option list) (body : constr) : constr
    := let rawindexT := 'RawIndexType in
       let nbinders := List.length names in
       let body := Constr.Unsafe.liftn nbinders (Int.add 1 nbinders) body in
       let rec aux (names : ident option list) (rel_above : int) (body : constr) :   int * constr
         := match names with
            | [] => (1, body)
            | name :: names
              => let (cur_rel, body) := aux names (Int.add 1 rel_above) body in
                 (Int.add 1 cur_rel,
                   sum_to (mkRel (Int.add cur_rel rel_above)) (mkLambda (Constr.Binder.make name rawindexT) body))
            end in
       let (_, body) := aux names 0 body in
       body.

  Ltac2 constr_dropn (n : int) (k : int) (c : constr) : constr
    := let k := Int.sub k 1 in
       let invalid := mkVar (ident:(__CONSTR_DROPN_INVALID)) in
       debug_printf "dropping %i %i %t" n k c;
       let res := Constr.Unsafe.substnl (List.repeat invalid n) k c in
       debug_printf "dropped %i %i %t" n k res;
       res.

  Ltac2 Type exn ::= [ InternalEinsumNotEqual (constr, constr) ].
  Ltac2 Type exn ::= [ InternalEinsumNotEnoughArgs (int, constr, constr array) ].
  Ltac2 Type exn ::= [ InternalEinsumBadKind (Constr.Unsafe.kind) ].

  Ltac2 rec remove_dead_einsum_helper (hd_c : constr) (nargs : int) (names : 'a list) (body : constr) :   int *   int list * (  int *   constr)
    := match names with
       | [] => (1, toplevel_rels body, (0, body))
       | _ :: names
         => match Constr.Unsafe.kind_nocast body with
            | Constr.Unsafe.App f args
              => (if Int.ge nargs (Array.length args)
                  then Control.throw (InternalEinsumNotEnoughArgs nargs f args)
                  else ());
                 (let first_args := List.firstn nargs (Array.to_list args) in
                  let fargs := mkApp f first_args in
                  if Bool.neg (Constr.equal fargs hd_c)
                  then Control.throw (InternalEinsumNotEqual fargs hd_c)
                  else ());
                 let lam_body_pos := Int.sub (Array.length args) 1 in
                 let lam_body := Array.get args lam_body_pos in
                 match Constr.Unsafe.kind_nocast lam_body with
                 | Constr.Unsafe.Lambda b body
                   => let (cur_rel, used_rels, (accumulated_shift, body)) := remove_dead_einsum_helper hd_c nargs names body in
                      let (cur_rel_used, used_rels)
                        := match used_rels with
                           | r :: rs => if Int.equal cur_rel r
                                        then (true, rs)
                                        else (false, used_rels)
                           | [] => (false, used_rels)
                           end in
                      (Int.add cur_rel 1,
                        used_rels,
                        if cur_rel_used
                        then

                          let body := constr_dropn (Int.neg accumulated_shift) 1 body in
                          let lam_body := mkLambda b body in
                          Array.set args lam_body_pos lam_body;
                          (0, Constr.Unsafe.make (Constr.Unsafe.App f args))
                        else
                          (Int.sub accumulated_shift 1,
                            body))
                 | k => Control.throw (InternalEinsumBadKind k)
                 end
            | k => Control.throw (InternalEinsumBadKind k)
            end
       end.

  Ltac2 remove_dead_einsum (hd_c : constr) (nargs : int) (names : 'a list) (body : constr) : constr
    := debug_printf "remove dead from %t" body;
       let (_cur_rel, _used_rels, (accumulated_shift, body)) := remove_dead_einsum_helper hd_c nargs names body in

       constr_dropn (Int.neg accumulated_shift) 1 body.

  Ltac2 insert_einsums (ty : constr) (names : ident option list) (body : constr) : constr
    := let start := '(0%uint63) in
       let step := '(1%uint63) in
       let sum := '(@Reduction.sum $ty try_tc try_tc) in
       let sum_to stop body := mkApp sum [start; stop; step; body] in
       let body := insert_all_einsums sum_to names body in
       let n_sum_args := match Constr.Unsafe.kind_nocast sum with
                         | Constr.Unsafe.App _ args => Array.length args
                         | k => Control.throw (InternalEinsumBadKind k)
                         end in
       let body := remove_dead_einsum sum n_sum_args names body in
       body.

  Ltac2 Type exn ::= [ EinsumExtraNames (Constr.Unsafe.kind, ident option list) ].

  Ltac2 insert_all_einsums_below (ty : constr) (names : ident option list) (body : constr) : constr
    := let rec aux (cur_names : ident option list) (body : constr)
         := match cur_names with
            | [] => insert_einsums ty names body
            | n :: ns
              => match Constr.Unsafe.kind_nocast body with
                 | Constr.Unsafe.Lambda b body
                   => mkLambda b (aux ns body)
                 | k => Control.throw (EinsumExtraNames k cur_names)
                 end
            end in
       aux names body.

  Ltac2 make_einsum (shapes : constr list) (body : constr) : constr
    := let ty := Constr.type body in
       let c := Std.eval_pattern (List.map (fun s => (s, Std.AllOccurrences)) shapes) body in
       let names := List.map ident_of_constr shapes in
       match Constr.Unsafe.kind_nocast c with
       | Constr.Unsafe.App f shape_args
         => let f := insert_all_einsums_below ty names f in
            let f := Constr.Unsafe.make (Constr.Unsafe.App f shape_args) in

            let f := (eval cbv beta in f) in

            f
       | k
         => Control.throw (InternalEinsumBadKind k)
       end.

  Ltac subst_type_lets_in_goal _ :=
    repeat match goal with
      | [ H := [ _ ] : Type |- _ ] => match goal with |- context[H] => idtac end; subst H
      end.
End Internals.

Local Notation try_tc := (ltac2:(ltac1:(try typeclasses eauto))) (only parsing).

Local Notation indirect_einsum tensor_value ishape jshape
  := ltac2:(let get_body v := get_body false (Constr.pretype v) in
            let get_shape v := shape_to_list (get_body v) in
            let t := get_body tensor_value in
            let shapes := List.append (get_shape ishape) (get_shape jshape) in
            let t := make_einsum shapes t in
            exact $t)
             (only parsing).
Import RawIndex.UncurryNotation.
#[local] Notation "'unify_rank_from_idxs' r @ i1 .. i_"
  := ((uncurry_fun i1 .. i_ => I) : RawIndex r -> True)
       (only parsing, i1 binder, i_ binder, at level 10).

#[export] Hint Extern 1 => progress subst_type_lets_in_goal () : typeclass_instances.
Declare Custom Entry einsum_args.

Notation "{{{ {{ i1 .. i_ , j1 .. j_ -> k1 .. k_ }} , t1 , t2 }}}"
  := (match t1%tensor, t2%tensor, _ as A, _ as B, _ as C, _ as r1, _ as r2, _ as r3, _ as s1, _ as s2, _ as s3 return @tensor _ s3 C with
      | t1', t2', A, B, C, r1, r2, r3, s1, s2, s3
        => match t1' : @tensor r1 s1 A, t2' : @tensor r2 s2 B return @tensor r3 s3 C with
           | t1', t2'
             => match
                 unify_rank_from_idxs r1 @ i1 .. i_,
                 unify_rank_from_idxs r2 @ j1 .. j_,
                 unify_rank_from_idxs r3 @ k1 .. k_
                   return @tensor r3 s3 C
                 with
               | _, _, _
                 => @with_shape
                      r1 s1 (tensor s3 C)
                      (λ i1 .. i_ ,
                        @with_shape
                          r2 s2 (tensor s3 C)
                          (λ j1 .. j_ ,
                            (match
                                (Shape.snoc .. (Shape.snoc Shape.nil i1) .. i_),
                                (Shape.snoc .. (Shape.snoc Shape.nil j1) .. j_)
                                return @tensor r3 s3 C
                              with
                              | __I_SHAPE, __J_SHAPE
                                => @Tensor.uncurry
                                     r3 s3 C
                                     (λ k1 .. k_ ,
                                       match @Arith.Classes.mul
                                               A B C try_tc
                                               (raw_get t1' (RawIndex.snoc .. (RawIndex.snoc RawIndex.nil i1) .. i_))
                                               (raw_get t2' (RawIndex.snoc .. (RawIndex.snoc RawIndex.nil j1) .. j_))
                                             return C
                                       with
                                       | __EINSUM_TENSOR_VALUE
                                         => indirect_einsum
                                              __EINSUM_TENSOR_VALUE __I_SHAPE __J_SHAPE
                                        end)
                               end)))
                end
           end
      end)
       (only parsing, in custom einsum_args at level 0, i1 binder, i_ binder, j1 binder, j_ binder, k1 binder, k_ binder, t1 constr at level 10, t2 constr at level 10).

Notation "'weaksauce_einsum' x"
  := (match x return _ with
      | y => ltac2:(let y := get_body false &y in
                    let z := (eval cbv beta iota delta [Tensor.with_shape Shape.uncurry Tensor.uncurry RawIndex.uncurry Shape.uncurry_dep RawIndex.uncurry_dep Shape.uncurry_map_dep RawIndex.uncurry_map_dep] in y) in
                    let z := (eval cbn beta iota delta [Nat.radd] in z) in
                    let z := (eval cbn beta iota zeta delta [Shape.snoc Shape.nil] in z) in

                    exact $z)
      end)
       (x custom einsum_args at level 10, at level 10, only parsing).

End Einsum.

Module Export NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer_WRAPPED.
Module Export HookedTransformer.
Import Coq.Numbers.Cyclic.Int63.Sint63.
Import Coq.QArith.QArith.
Import NeuralNetInterp.Util.Default.
Import NeuralNetInterp.Util.Pointed.
Import NeuralNetInterp.Util.Arith.Instances.
Import NeuralNetInterp.Torch.Tensor.
Import NeuralNetInterp.Torch.Slicing.
Export NeuralNetInterp.TransformerLens.HookedTransformer.Config.
Import Arith.Instances.Truncating.

Module Export Embed.
  Section __.
    Context {r} {s : Shape r} {d_vocab d_model A}
      (W_E : tensor [d_vocab; d_model] A).
Definition forward (tokens : tensor s IndexType) : tensor (s ::' d_model) A.
Admitted.
  End __.
End Embed.

Module Export PosEmbed.
  Section __.
    Context {r} {batch : Shape r} {tokens_length : ShapeType}
      {d_model} {n_ctx:N}
      (n_ctx' : int := n_ctx)
      {A}
      (W_pos : tensor [n_ctx'; d_model] A).
Definition forward (tokens : tensor (batch ::' tokens_length) IndexType)
      : tensor (batch ::' tokens_length ::' d_model) A.
Admitted.
  End __.
End PosEmbed.

Module Export LayerNorm.
  Section __.
    Context {r} {s : Shape r} {d_model}
      {A}
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A}
      {use_checkpoint : with_default "use_checkpoint" bool true}
      (defaultA : pointed A := @coer _ _ coerZ point)
      (eps : A) (w b : tensor [d_model] A).
Definition forward (x : tensor (s ::' d_model) A)
      : tensor (s ::' d_model) A.
Admitted.
  End __.
End LayerNorm.

Module Export Attention.
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
Definition einsum_input
      (input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A)
      (W : tensor [n_heads; d_model; d_head] A)
      : tensor ((batch ::' pos) ++' [n_heads; d_head]) A.
admit.
Defined.
    #[local] Existing Instance defaultA.
Definition v : tensor (batch ++' [pos; n_heads; d_head]) A.
Admitted.
Definition pattern : tensor (batch ::' n_heads ::' pos ::' pos) A.
Admitted.


  End __.
End Attention.

Module Export TransformerBlock.
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
Definition query_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A.
exact (add_head_dimension resid_pre).
Defined.
Definition key_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A.
admit.
Defined.
Definition value_input : tensor ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)) A.
admit.
Defined.

    #[local] Notation LayerNorm_forward
      := (match normalization_type
                return ln_tensor_gen _ normalization_type _ -> ln_tensor_gen _ normalization_type _ -> _
          with
          | Some LN => LayerNorm.forward eps
          | Datatypes.None => fun _ _ x => checkpoint x
          end)
           (only parsing).
Definition ln1   (t : tensor (ln_s ::' d_model) A) : tensor (ln_s ::' d_model) A.
exact (LayerNorm_forward ln1_w ln1_b t).
Defined.
Definition attn_only_out : tensor (batch ++ [pos; d_model]) A.
Proof using A W_K W_O W_Q W_V addA b_K b_O b_Q b_V batch coerZ d_head d_model defaultA divA
eps expA ln1_b ln1_w ln_s maybe_n_heads mulA n_ctx n_heads normalization_type pos r
resid_pre sqrtA subA use_checkpoint use_split_qkv_input zeroA
.
Admitted.
Definition attn_masked_attn_scores : tensor (batch ::' n_heads ::' pos ::' pos) A.
Proof using A W_K W_Q addA b_K b_Q batch coerZ d_head d_model defaultA divA eps ln1_b ln1_w
ln_s maybe_n_heads mulA n_ctx n_heads normalization_type pos r resid_pre sqrtA subA
use_checkpoint use_split_qkv_input zeroA
.
Admitted.
  End __.
End TransformerBlock.

Module Export HookedTransformer.
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

      {use_checkpoint : with_default "use_checkpoint" bool true}
      (eps : A)

      (W_E : tensor [d_vocab; d_model] A)
      (W_pos : tensor [n_ctx; d_model] A)

      (blocks_params
        : list (block_params_type_gen n_heads d_model d_head normalization_type A)
             )

      (ln_final_w ln_final_b : ln_tensor_gen d_model normalization_type A)

      (W_U : tensor [d_model; d_vocab_out] A) (b_U : tensor [d_vocab_out] A)
    .
    #[local] Existing Instance defaultA.
    #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).
    Definition embed (tokens : tensor s IndexType) : tensor resid_shape A.
      Proof using A W_E batch d_model d_vocab pos r resid_shape s.
Admitted.
      Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape A.
      Proof using A W_pos batch d_model n_ctx pos r resid_shape s.
        Admitted.
Definition blocks : list (tensor resid_shape A -> tensor resid_shape A).
exact (List.map
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
           blocks_params).
Defined.
Polymorphic Definition blocks_cps {T} {n : with_default "blocks n" nat (List.length blocks)} (residual : tensor resid_shape A) (K : tensor resid_shape A -> T) : T.
admit.
Defined.
    Definition resid_postembed (tokens : tensor s IndexType) : tensor resid_shape A.
      Proof using A W_E W_pos addA batch coerZ d_model d_vocab defaultA n_ctx pos r resid_shape s
        use_checkpoint.
      Admitted.
Definition blocks_attn_masked_attn_scores
      : list (tensor resid_shape A -> tensor (batch ::' n_heads ::' pos ::' pos) A).
exact (List.map
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
           blocks_params).
Defined.
Definition masked_attn_scores (n : nat) (tokens : tensor s IndexType)
      : option (tensor (batch ::' n_heads ::' pos ::' pos) A).
exact (match List.nth_error blocks_attn_masked_attn_scores n with
         | Some block_n_attn_masked_attn_scores
           => Some (let residual       := resid_postembed tokens in
                    blocks_cps
                      (n:=Nat.pred n)
                      residual
                      (fun residual
                       => checkpoint (block_n_attn_masked_attn_scores residual)))
         | None => None
         end).
Defined.
  End __.
End HookedTransformer.

End HookedTransformer.

End NeuralNetInterp_DOT_TransformerLens_DOT_HookedTransformer_WRAPPED.
Module Export Instances.
Import NeuralNetInterp.Util.Option.
Import NeuralNetInterp.Torch.Tensor.Instances.
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
Import NeuralNetInterp.Util.Arith.Instances.
Import NeuralNetInterp.Util.Option.
Import NeuralNetInterp.Torch.Tensor.
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
