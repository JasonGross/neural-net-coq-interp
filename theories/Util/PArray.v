From Coq Require Import Bool ZArith NArith Uint63 List PArray Wellfounded Lia.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63 Slice Arith.Instances.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Open Scope uint63_scope.

#[local] Coercion is_true : bool >-> Sortclass.
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.
#[local] Coercion N.of_nat : nat >-> N.
Fixpoint fill_array_of_list {A} (ls : list A) (start : int) (arr : array A) {struct ls} : array A
  := match ls with
     | [] => arr
     | x :: xs
       => fill_array_of_list xs (start+1) arr.[ start <- x ]
     end.
Definition array_of_list {A} (ls : list A) {default : pointed A} : array A
  := fill_array_of_list ls 0 (PArray.make (List.length ls) default).

Import LoopNotation.
Definition map_default {A B} {default : pointed B} (f : A -> B) (xs : array A) : array B
  := let len := PArray.length xs in
     with_state (PArray.make len default)
       for (i := 0;; i <? len;; i++) {{
           res <-- get;;
           set (res.[i <- f xs.[i]])
       }}.

Definition map {A B} (f : A -> B) (xs : array A) : array B
  := map_default (default:=f (PArray.default xs)) f xs.

Import Slice.ConcreteProjections.

Definition slice {A} (xs : array A) (s : Slice int) : array A
  := let len := PArray.length xs in
     let s := Slice.norm_concretize s len in
     if (s.(start) =? 0) && (s.(step) =? 1) && (s.(stop) =? len)
     then
       xs
     else
       let new_len := Slice.Concrete.length s in
       let res := PArray.make new_len (PArray.default xs) in
       with_state res
         for (i := 0;; i <? new_len;; i++) {{
             res <-- get;;
             set (res.[i <- xs.[i * s.(step) + s.(start)]])
         }}.

Export SliceNotations.
Notation "x .[ [ s ] ]" := (slice x s) (at level 2, s custom slice at level 60, format "x .[ [ s ] ]") : core_scope.

Definition repeat {A} (xs : array A) (count : int) : array (array A)
  := PArray.make count xs.
