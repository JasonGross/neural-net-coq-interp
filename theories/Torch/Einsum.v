From Coq Require Import Sint63 Uint63 Utf8.
From Ltac2 Require Ltac2 Constr List Ident String Fresh Printf.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Arrow.
From NeuralNetInterp.Util.Tactics2 Require Constr FixNotationsForPerformance Constr.Unsafe.MakeAbbreviations List Ident.
From NeuralNetInterp.Util Require Export Arith.Classes.

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
       let einsum_ids := List.uniq Ident.equal (List.diff Ident.equal src_ids dest_ids) in
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

Local Notation try_tc := (ltac2:(ltac1:(try typeclasses eauto))) (only parsing).

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

#[local] Notation "'make_idxs_for_einsum' i1 .. i_"
  := ((fun i1 => .. ((fun i_ => I) : True -> _) ..) : True -> _)
       (only parsing, i1 binder, i_ binder, at level 10).
Import RawIndex.UncurryNotation.
#[local] Notation "'unify_rank_from_idxs' r @ i1 .. i_"
  := ((uncurry_fun i1 .. i_ => I) : RawIndex r -> True)
       (only parsing, i1 binder, i_ binder, at level 10).
(* TODO: fix naming *)
#[export] Hint Extern 1 => progress subst_type_lets_in_goal () : typeclass_instances.
Declare Custom Entry einsum_args.
Notation "{{{ {{ i1 .. i_ , j1 .. j_ -> k1 .. k_ }} , t1 , t2 }}}"
  := (match t1%tensor, t2%tensor, _ as A, _ as B, _ as C, _ as r1, _ as r2, _ as r3, _ as s1, _ as s2, _ as s3 return @tensor _ C s3 with
      | t1', t2', A, B, C, r1, r2, r3, s1, s2, s3
        => match t1' : @tensor r1 A s1, t2' : @tensor r2 B s2 return @tensor r3 C s3 with
           | t1', t2'
             => match make_idxs_for_einsum i1 .. i_,
                  make_idxs_for_einsum j1 .. j_,
                  make_idxs_for_einsum k1 .. k_,
                  (* for typing *)
                  unify_rank_from_idxs r1 @ i1 .. i_,
                  unify_rank_from_idxs r2 @ j1 .. j_,
                  unify_rank_from_idxs r3 @ k1 .. k_
                    return @tensor r3 C s3
                with
                | __EINSUM_IIDXS, __EINSUM_JIDXS, __EINSUM_KIDXS
                  , _, _, _
                  => @with_shape
                       r1 (tensor C s3) s1
                       (λ i1 .. i_ ,
                         @with_shape
                           r2 (tensor C s3) s2
                           (λ j1 .. j_ ,
                             (@Tensor.uncurry
                                r3 C s3
                                (λ k1 .. k_ ,
                                  match @Arith.Classes.mul
                                          A B C try_tc
                                          (t1' (RawIndex.snoc .. (RawIndex.snoc RawIndex.nil i1) .. i_))
                                          (t2' (RawIndex.snoc .. (RawIndex.snoc RawIndex.nil j1) .. j_))
                                        return C
                                  with
                                  | __EINSUM_TENSOR_VALUE
                                    => indirect_einsum
                                         __EINSUM_TENSOR_VALUE __EINSUM_KIDXS __EINSUM_IIDXS __EINSUM_JIDXS
                                  end))))
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
                    (*let z := (eval cbn beta iota zeta delta [PrimitiveProd.Primitive.fst PrimitiveProd.Primitive.snd Shape.snoc Shape.nil] in z) in*)
                    exact $z)
      end)
       (x custom einsum_args at level 10, at level 10, only parsing).
(*
Set Printing Implicit.
Check (weaksauce_einsum {{{ {{ query_pos head_index d_head,
                   key_pos head_index d_head
                   -> head_index query_pos key_pos }}, (_:tensor _ [2;1;5]), (_:tensor _ [2;1;5]) }}} : tensor _ [1; 2; 2]).
*)
