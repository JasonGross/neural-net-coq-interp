From Coq Require Import Sint63 Uint63 Utf8.
From Ltac2 Require Ltac2 Constr List Ident String Fresh Printf.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Wf_Uint63 Arrow.
From NeuralNetInterp.Util.Tactics2 Require Constr FixNotationsForPerformance Constr.Unsafe.MakeAbbreviations List Ident.
From NeuralNetInterp.Util Require Export Arith.Classes.

Import Ltac2.
Import FixNotationsForPerformance MakeAbbreviations Printf.

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
  (* inserts einsum for all binders *)
  Ltac2 insert_all_einsums (sum_to : constr -> constr -> constr) (names : ident option list) (body : constr) : constr
    := let rawindexT := 'RawIndexType in
       let nbinders := List.length names in
       let body := Constr.Unsafe.liftn nbinders (Int.add 1 nbinders) body in
       let rec aux (names : ident option list) (rel_above : int) (body : constr) : (* cur_rel : *) int * constr
         := match names with
            | [] => (1, body)
            | name :: names
              => let (cur_rel, body) := aux names (Int.add 1 rel_above) body in
                 (Int.add 1 cur_rel,
                   sum_to (mkRel (Int.add cur_rel rel_above)) (mkLambda (Constr.Binder.make name rawindexT) body))
            end in
       let (_, body) := aux names 0 body in
       body.

  (* like Constr.Unsafe.liftn (-n) k c, except without anomalies *)
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
  (* removes unused einsum *)
  Ltac2 rec remove_dead_einsum_helper (hd_c : constr) (nargs : int) (names : 'a list) (body : constr) : (* cur_rel : *) int * (* used_rels : *) int list * ((* accumulated shift : *) int * (* body : *) constr)
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
                          (* drop down rels that we dropped *)
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
       (* drop down rels that we dropped *)
       constr_dropn (Int.neg accumulated_shift) 1 body.

  (* inserts einsum for used binders *)
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

  (* inserts Reduction.sum for all binders *)
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
            (*printf "before: %t" f;*)
            let f := (eval cbv beta in f) in
            (*printf "after: %t" f;*)
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
