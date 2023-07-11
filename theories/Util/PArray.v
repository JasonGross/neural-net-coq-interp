From Coq Require Import Bool ZArith NArith Sint63 Uint63 List PArray Wellfounded Lia.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63 Slice Arith.Classes Arith.Instances Default Notations Bool.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Open Scope uint63_scope.

#[local] Coercion is_true : bool >-> Sortclass.
Fixpoint fill_array_of_list_map {A B} (f : A -> B) (ls : list A) (start : int) (arr : array B) {struct ls} : array B
  := match ls with
     | [] => arr
     | x :: xs
       => fill_array_of_list_map f xs (start+1) arr.[ start <- f x ]
     end.
Definition fill_array_of_list {A} (ls : list A) (start : int) (arr : array A) : array A := fill_array_of_list_map (fun x => x) ls start arr.
Definition array_of_list_map {A B} {default : pointed B} (f : A -> B) (ls : list A) : array B
  := fill_array_of_list_map f ls 0 (PArray.make (Uint63.of_Z (List.length ls)) default).
Definition array_of_list {A} {default : pointed A} (ls : list A) : array A
  := array_of_list_map (fun x => x) ls.

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

Definition init_default {A} {default : pointed A} (len : int) (f : int -> A) : array A
  := let len := if len <=? PArray.max_length then len else PArray.max_length in
     with_state (PArray.make len default)
       for (i := 0;; i <? len;; i++) {{
           res <-- get;;
           set (res.[i <- f i])
       }}.

Definition init {A} (len : int) (f : int -> A) : array A
  := @init_default A (f 0) len f.

Definition map2_default {A B C} {default : pointed C} {reduce_len : with_default "reduce_len" (int -> int -> int) min} (f : A -> B -> C) (xs : array A) (ys : array B) : array C
  := let lenA := PArray.length xs in
     let lenB := PArray.length ys in
     let len := reduce_len lenA lenB in
     with_state (PArray.make len default)
       for (i := 0;; i <? len;; i++) {{
           res <-- get;;
           set (res.[i <- f xs.[i] ys.[i]])
       }}.

Definition map2 {A B C} {reduce_len} (f : A -> B -> C) (xs : array A) (ys : array B) : array C
  := map2_default (reduce_len:=reduce_len) (default:=f (PArray.default xs) (PArray.default ys)) f xs ys.

Definition broadcast_map2 {A B C} (f : A -> B -> C) (xs : array A) (ys : array B) : array C
  := let lenA := PArray.length xs in
     let lenB := PArray.length ys in
     let len1 := min lenA lenB in
     let len2 := max lenA lenB in
     let st :=
     with_state (PArray.make len2 (f (PArray.default xs) (PArray.default ys)))
       for (i := 0;; i <? len1;; i++) {{
           res <-- get;;
           set (res.[i <- f xs.[i] ys.[i]])
       }} in
     if lenA <? lenB
     then with_state st
            for (i := len1;; i <? len2;; i++) {{
                res <-- get;;
                set (res.[i <- f xs.[i] ys.[0]])
            }}
     else with_state st
            for (i := len1;; i <? len2;; i++) {{
                res <-- get;;
                set (res.[i <- f xs.[0] ys.[i]])
            }}.

(* TODO: make nary version *)
Definition broadcast_map3 {A B C D} (f : A -> B -> C -> D) (xs : array A) (ys : array B) (zs : array C) : array D
  := let lenA := PArray.length xs in
     let lenB := PArray.length ys in
     let lenC := PArray.length zs in
     let len := max (max lenA lenB) lenC in
     with_state (PArray.make len (f (PArray.default xs) (PArray.default ys) (PArray.default zs)))
       for (i := 0;; i <? len;; i++) {{
           res <-- get;;
           set (res.[i <- f xs.[i] ys.[i] zs.[i]])
       }}.

Definition reduce {A B} (f : B -> A -> B) (init : B) (xs : array A) : B
  := let len := PArray.length xs in
     with_state init
       for (i := 0;; i <? len;; i++) {{
           acc <-- get;;
           set (f acc xs.[i])
       }}.

Definition reduce_no_init {A} (f : A -> A -> A) (xs : array A) : A
  := let len := PArray.length xs in
     with_state xs.[0]
       for (i := 1;; i <? len;; i++) {{
           acc <-- get;;
           set (f acc xs.[i])
       }}.

Definition reduce_map {A A' B} (red : B -> A' -> B) (init : B) (f : A -> A') (xs : array A) : B
  := reduce (fun acc a => red acc (f a)) init xs.

Definition reduce_map_no_init {A A'} (red : A' -> A' -> A') (f : A -> A') (xs : array A) : A'
  := let len := PArray.length xs in
     with_state (f xs.[0])
       for (i := 1;; i <? len;; i++) {{
           acc <-- get;;
           set (red acc (f xs.[i]))
       }}.

(* TODO: probably want to broadcast torch.where without having to allocate arrays of bool, etc *)
Definition where_ {A} (condition : array bool) (input other : array A) : array A
  := broadcast_map3 Bool.where_ condition input other.

Notation "\sum_ ( xi <- xs ) F" := (reduce_map_no_init add (fun xi => F) xs) : core_scope.
Notation "∑_ ( xi <- xs ) F" := (reduce_map_no_init add (fun xi => F) xs) : core_scope.
Notation "\prod_ ( xi <- xs ) F" := (reduce_map_no_init mul (fun xi => F) xs) : core_scope.
Notation "∏_ ( xi <- xs ) F" := (reduce_map_no_init mul (fun xi => F) xs) : core_scope.

Definition sum {A} {zero : has_zero A} {add : has_add A} (xs : array A) : A
  := ∑_(xi <- xs) xi.
Definition prod {A} {one : has_one A} {mul : has_mul A} (xs : array A) : A
  := ∏_(xi <- xs) xi.
Definition max {A} {max : has_max A} (xs : array A) : A := reduce_no_init max xs.
Definition min {A} {min : has_max A} (xs : array A) : A := reduce_no_init min xs.
Definition mean {A B C} {zero : has_zero A} {add : has_add A} {div_by : has_div_by A B C} {coer : has_coer Z B} (xs : array A) : C
  := (sum xs / coer (Uint63.to_Z (PArray.length xs)))%core.
Definition var {A B} {zero : has_zero A} {add : has_add A} {mul : has_mul A} {sub : has_sub A} {div_by : has_div_by A B A} {coer : has_coer Z B} {correction : with_default "correction" Z 1%Z}
  (xs : array A) : A
  := (let xbar := mean xs in
     let N := Uint63.to_Z (PArray.length xs) in
     ((∑_(xi <- xs) (xi - xbar)²) / (coer (N - correction)))%core).

#[export] Instance has_add {A B C} {addA : has_add_with A B C} : has_add_with (array A) (array B) (array C) := broadcast_map2 add.
#[export] Instance has_sub {A B C} {subA : has_sub_with A B C} : has_sub_with (array A) (array B) (array C) := broadcast_map2 sub.
#[export] Instance has_mul {A B C} {mulA : has_mul_with A B C} : has_mul_with (array A) (array B) (array C) := broadcast_map2 mul.
#[export] Instance has_div_by {A B C} {div_byAB : has_div_by A B C} : has_div_by (array A) (array B) (array C) := broadcast_map2 div.
#[export] Instance has_sqrt {A} {sqrtA : has_sqrt A} : has_sqrt (array A) := map sqrt.
#[export] Instance has_opp {A} {oppA : has_opp A} : has_opp (array A) := map opp.

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
Notation "x .[ < s > ]" := (slice x s) (at level 2, s custom slice at level 60, format "x .[ < s > ]") : core_scope.

Definition repeat {A} (xs : A) (count : int) : array A
  := PArray.make count xs.

Definition to_bool {A} {zero : has_zero A} {eqb : has_eqb A} (xs : array A) : array bool
  := map (fun x => x ≠? 0)%core xs.

(** Quoting https://pytorch.org/docs/stable/generated/torch.tril.html

torch.tril(input, diagonal=0, *, out=None) → Tensor

Returns the lower triangular part of the matrix (2-D tensor) or batch
of matrices [input], the other elements of the result tensor [out] are
set to 0.

The lower triangular part of the matrix is defined as the elements on
and below the diagonal.

The argument [diagonal] controls which diagonal to consider. If
[diagonal = 0], all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the
main diagonal, and similarly a negative value excludes just as many
diagonals below the main diagonal. The main diagonal are the set of
indices {(i,i)} for i ∈ [0,min{d₁,d₂}−1] where d₁,d₂ are the
dimensions of the matrix. *)
Definition tril {A} {zero : has_zero A} {diagonal : with_default "diagonal" int 0} (input : array (array A)) : array (array A)
  := let len := PArray.length input in
     with_state input
       for (i := 0;; i <? len;; i++) {{
           out <-- get;;
           let row := out.[i] in
           let clen := PArray.length row in
           let row
             := with_state row
                  for (j := Sint63.max 0 (1 + i + diagonal);; j <? clen;; j++) {{
                      row <-- get;;
                      set (row.[j <- 0%core])
                  }}
           in
           set (out.[i <- row])
       }}.
#[global] Arguments tril {A%type_scope zero diagonal%sint63} input.

(** Quoting https://pytorch.org/docs/stable/generated/torch.triu.html

torch.triu(input, diagonal=0, *, out=None) → Tensor

Returns the upper triangular part of the matrix (2-D tensor) or batch
of matrices [input], the other elements of the result tensor [out] are
set to 0.

The upper triangular part of the matrix is defined as the elements on
and above the diagonal.

The argument [diagonal] controls which diagonal to consider. If
[diagonal = 0], all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the
main diagonal, and similarly a negative value includes just as many
diagonals below the main diagonal. The main diagonal are the set of
indices {(i,i)} for i ∈ [0,min{d₁,d₂}−1] where d₁,d₂ are the
dimensions of the matrix. *)
Definition triu {A} {zero : has_zero A} {diagonal : with_default "diagonal" int 0} (input : array (array A)) : array (array A)
  := let len := PArray.length input in
     with_state input
       for (i := 0;; i <? len;; i++) {{
           out <-- get;;
           let row := out.[i] in
           let clen := PArray.length row in
           let row
             := with_state row
                  for (j := 0;; j <? Sint63.max 0 (i + diagonal);; j++) {{
                      row <-- get;;
                      set (row.[j <- 0%core])
                  }}
           in
           set (out.[i <- row])
       }}.
#[global] Arguments triu {A%type_scope zero diagonal%sint63} input.
