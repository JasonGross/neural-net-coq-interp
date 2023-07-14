From Coq Require Import Sint63 Uint63 Utf8.
From Ltac2 Require Ltac2 Constr List Ident String Fresh Printf.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Wf_Uint63 Arrow.
From NeuralNetInterp.Util.Tactics2 Require Constr FixNotationsForPerformance Constr.Unsafe.MakeAbbreviations List Ident.
From NeuralNetInterp.Util Require Export Arith.Classes.

Import Ltac2.
Import FixNotationsForPerformance MakeAbbreviations Printf.

Module Import Internals.
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
FIXME REWORK
  Ltac2 rec insert_einsums_helper (rawindexT : constr) (sum_to : (* stop : *) constr -> (* body : *) constr -> constr) (shapes : constr list) (lift_rel_by : int) (c : constr) : (* used_rels : *) int list * (* cur_rel : *) int * (* new_body : *) constr
    := match Constr.Unsafe.kind_nocast c with
       | Constr.Unsafe.Lambda b body
         => let (name, shape, shapes)
              := match shapes with
                 | shape :: shapes => (ident_of_constr shape, shapes, names)
                 | [] => (Constr.Binder.name b, names)
                 end in
            let (used_rels, cur_rel, body) := insert_einsums_helper rawindexT sum_to names body in
            let (cur_rel_used, used_rels)
              := match used_rels with
                 | r :: rs
                   => if Int.equal cur_rel r
                      then (true, rs)
                      else (false, used_rels)
                 | [] => (false, used_rels)
                 end in
            let b := Constr.Binder.make name rawindexT in
            (used_rels, Int.add cur_rel 1,
              if cur_rel_used
              then mkLambda b (sum_to (Constr.Unsafe.liftn lift_rel_by 1 shape) (Constr.Unsafe.liftn 1 1 (mkLambda b body)))
              else mkLambda b body)
       | _ => (toplevel_rels c, 1, c)
       end.

  Ltac2 Type exn ::= [ InternalEinsumBadKind (Constr.Unsafe.kind) ].

  Local Notation try_tc := (ltac:(try typeclasses eauto)) (only parsing).
  Ltac2 make_einsum (shapes : constr list) (body : constr) : constr
    := let ty := Constr.type body in
       let c := Std.eval_pattern (List.map (fun s => (s, Std.AllOccurrences)) shapes) body in
       let rawindexT := 'RawIndexType in
       let start := '(0%uint63) in
       let step := '(1%uint63) in
       let sum := '(@Reduction.sum $ty try_tc try_tc) in
       let sum_to stop body := mkApp sum [start; stop; step; body] in
       let names := List.map ident_of_constr shapes in
       match Constr.Unsafe.kind_nocast c with
       | Constr.Unsafe.App f shape_args
         => let (_, _, f) := insert_einsums_helper rawindexT sum_to names f in
            let f := Constr.Unsafe.make (Constr.Unsafe.App f shape_args) in
            printf "before: %t" f;
            let f := (eval cbv beta in f) in
            printf "after: %t" f;
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

(* Kludge around COQBUG(https://github.com/coq/coq/issues/17833#issuecomment-1627483008) *)
Local Notation indirect_einsum tensor_value ishape jshape
  := ltac2:(let get_body v := get_body false (Constr.pretype v) in
            let get_shape v := shape_to_list (get_body v) in
            let t := get_body tensor_value in
            let shapes := List.append (get_shape ishape) (get_shape jshape) in
            let t := make_einsum shapes t in
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
(*TODO: use Ltac/Ltac2 just to strip off the unused einsums, and to push the einsums under the Tensor.uncurry.  Use Gallina to introduce einsums for all shapes.*)
Notation "{{{ {{ i1 .. i_ , j1 .. j_ -> k1 .. k_ }} , t1 , t2 }}}"
  := (match t1%tensor, t2%tensor, _ as A, _ as B, _ as C, _ as r1, _ as r2, _ as r3, _ as s1, _ as s2, _ as s3 return @tensor _ C s3 with
      | t1', t2', A, B, C, r1, r2, r3, s1, s2, s3
        => match t1' : @tensor r1 A s1, t2' : @tensor r2 B s2 return @tensor r3 C s3 with
           | t1', t2'
             => match (* for typing *)
                 unify_rank_from_idxs r1 @ i1 .. i_,
                 unify_rank_from_idxs r2 @ j1 .. j_,
                 unify_rank_from_idxs r3 @ k1 .. k_
                   return @tensor r3 C s3
                 with
               | _, _, _
                 => @with_shape
                      r1 (tensor C s3) s1
                      (λ i1 .. i_ ,
                        @with_shape
                          r2 (tensor C s3) s2
                          (λ j1 .. j_ ,
                            (match
                                (Shape.snoc .. (Shape.snoc Shape.nil i1) .. i_),
                                (Shape.snoc .. (Shape.snoc Shape.nil j1) .. j_)
                                return @tensor r3 C s3
                              with
                              | __I_SHAPE, __J_SHAPE
                                => @Tensor.uncurry
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
