From Ltac2 Require Import Array Init.

(** Definitions to be dropped when <8.19 compat is dropped *)

Ltac2 rec fold_right_aux (f : 'a -> 'b -> 'b) (a : 'a array) (x : 'b) (pos : int) (len : int) :=
  (* Note: one could compare pos<0.
     We keep an extra len parameter so that the function can be used for any sub array *)
  match Int.equal len 0 with
  | true => x
  | false => fold_right_aux f a (f (get a pos) x) (Int.sub pos 1) (Int.sub len 1)
  end.

Ltac2 fold_right (f : 'a -> 'b -> 'b) (a : 'a array) (x : 'b) : 'b :=
  fold_right_aux f a x (Int.sub (length a) 1) (length a).
