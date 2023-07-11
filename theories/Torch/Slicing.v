From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Slice Arith.Classes PolymorphicOption.
Set Implicit Arguments.
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Set Boolean Equality Schemes.
Set Decidable Equality Schemes.

#[local] Coercion is_true : bool >-> Sortclass.

(* no tensor *)
Inductive SliceIndexType : Rank -> Rank -> Type :=
| slice_index (s : Slice IndexType) : SliceIndexType 1 0
| broadcast_one_index : SliceIndexType 0 1
| single_index (i : IndexType) : SliceIndexType 1 0
.

Inductive FancyIndexType {r} (s : Shape r) : Rank -> Rank -> Type :=
| tensor_index (_ : @tensor r IndexType s) : FancyIndexType s 1 0
| normal_index {ri ro} (_ : SliceIndexType ri ro) : FancyIndexType s ri ro
.
#[global] Arguments normal_index {r s ri ro} _.
#[global] Arguments tensor_index {r s} _.
#[export] Set Warnings Append "-uniform-inheritance".
#[global] Set Warnings Append "-uniform-inheritance".
#[global] Coercion slice_index : Slice >-> SliceIndexType.
#[global] Coercion tensor_index : tensor >-> FancyIndexType.
#[global] Coercion normal_index : SliceIndexType >-> FancyIndexType.
#[export] Set Warnings Append "uniform-inheritance".
#[global] Set Warnings Append "uniform-inheritance".
#[global] Coercion single_index : IndexType >-> SliceIndexType.
Module Type InjectIndexTypeHack.
  Definition t := IndexType.
End InjectIndexTypeHack.
Module InjectIndexType (T : InjectIndexTypeHack).
  #[global] Identity Coercion inject_raw : T.t >-> IndexType.
End InjectIndexType.
Module Export InjectRawIndexType := InjectIndexType RawIndex.RawIndexType.

Module FancySlicingNotations.
  Declare Custom Entry fancy_slice.
  Notation ":" := (slice_index (@Build_Slice _ None None None)) (in custom fancy_slice at level 5).
  Notation "start : stop : step" := (slice_index (@Build_Slice _ (Some start) (Some stop) (Some step))) (in custom fancy_slice at level 58, start constr at level 59, stop constr at level 59, step constr at level 59, format "start : stop : step").
  Notation "start :: step" := (slice_index (@Build_Slice _ (Some start) None (Some step))) (in custom fancy_slice at level 58, start constr at level 59, step constr at level 59, format "start :: step").
  Notation "start : : step" := (slice_index (@Build_Slice _ (Some start) None (Some step))) (in custom fancy_slice at level 58, start constr at level 59, step constr at level 59, format "start : : step").
  Notation "start :: step" := (slice_index (@Build_Slice _ (Some start) None (Some step))) (in custom fancy_slice at level 58, start constr at level 59, step constr at level 59, format "start :: step").
  Notation ": stop : step" := (slice_index (@Build_Slice _ None (Some stop) (Some step))) (in custom fancy_slice at level 58, stop constr at level 59, step constr at level 59, format ": stop : step").
  Notation ": : step" := (slice_index (@Build_Slice _ None None (Some step))) (in custom fancy_slice at level 58, step constr at level 59, format ": : step").
  Notation ":: step" := (slice_index (@Build_Slice _ None None (Some step))) (in custom fancy_slice at level 58, step constr at level 59, format ":: step").
  Notation "start : stop" := (slice_index (@Build_Slice _ (Some start) (Some stop) None)) (in custom fancy_slice at level 58, start constr at level 59, stop constr at level 59, format "start : stop").
  Notation "start :" := (slice_index (@Build_Slice _ (Some start) None None)) (in custom fancy_slice at level 58, start constr at level 59, format "start :").
  Notation ": stop" := (slice_index (@Build_Slice _ None (Some stop) None)) (in custom fancy_slice at level 58, stop constr at level 59, format ": stop").
  Notation "'None'" := broadcast_one_index (in custom fancy_slice at level 0).
  Notation "x" := x (in custom fancy_slice at level 58, x constr at level 59).
End FancySlicingNotations.

Module SliceIndex.
  Module SliceIndexType <: IndexType.
    Definition t@{u} : Type@{u} := SliceIndexType@{u u}.
    Definition eqb : has_eqb t := SliceIndexType_beq.
  End SliceIndexType.



  #[export] Set Warnings Append "-notation-overridden".
  Include IndexGen.Make SliceIndexType.
  #[export] Set Warnings Append "notation-overridden".

  Module Export SliceNotations.
    Declare Custom Entry slice_index.
    Reserved Infix "::'" (in custom slice_index at level 59, left associativity).
    Notation "xs ::' x" := (snoc xs x) (in custom slice_index).
    Notation "x ; .. ; z" :=  (snoc .. (snoc nil x) .. z) (in custom slice_index at level 99, x custom fancy_slice, z custom fancy_slice).
    (*Notation "x :: xs" := (cons x xs) : slice_scope.
    Notation "s1 ++ s2" := (app s1 s2) : slice_scope.
    Notation "s1 ++' s2" := (app s1 s2) : slice_scope.*)
  End SliceNotations.

  Fixpoint rank_of_slice (s :
End SliceIndex.



(*
Module FancyIndex.
  Module FancyIndexType.
    Definition t {r} {s : Shape r} : Type := @FancyIndexType r s.
    Notation IndexType := t.
  End FancyIndexType.
  Import (hints) FancyIndexType.
  Notation IndexType := FancyIndexType.t.

  Section with_shape.
    Context {r0 : Rank} {s0 : Shape r0}.

    Local Notation IndexType := (@IndexType r0 s0).

    Fixpoint t (r : Rank) : Type
      := match r with
         | O => unit
         | S r => t r * IndexType
         end.
    Notation Index := t.
  End with_shape.
  Notation Index := t.

  Definition nil : t 0 := tt.
  Definition snoc {r} (s : t r) x : t (S r) := (s, x).
  Module Import IndexNotations0.
    Declare Scope index_scope.
    Delimit Scope index_scope with index.
    Bind Scope index_scope with Index.
    Notation "xs ::' x" := (snoc xs x) : index_scope.
    Notation "[ ]" := nil : index_scope.
    Notation "[ x ]" := (snoc nil x) : index_scope.
    Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc nil x) y) .. z) : index_scope.
  End IndexNotations0.
    Module IndexPatternNotations.
      Declare Scope index_pattern_scope.
      Delimit Scope index_pattern_scope with index_pattern.
      Notation "xs ::' x" := (pair xs x) : index_pattern_scope.
      Notation "[ ]" := tt : index_pattern_scope.
      Notation "[ x ]" := (pair tt x) : index_pattern_scope.
      Notation "[ x ; y ; .. ; z ]" :=  (pair .. (pair (pair tt x) y) .. z) : index_pattern_scope.
    End IndexPatternNotations.
    #[local] Open Scope index_scope.
    Definition hd {r : Rank} : Index (S r) -> Index r := @fst _ _.
    Definition tl {r : Rank} : Index (S r) -> IndexType := @snd _ _.
    Fixpoint app {r1 r2 : Rank} {struct r2} : Index r1 -> Index r2 -> Index (r1 +' r2)
      := match r2 with
         | 0%nat => fun sz _tt => sz
         | S r2 => fun sz1 sz2 => @app r1 r2 sz1 (hd sz2) ::' tl sz2
         end%index.
    Definition cons {r : Rank} x (xs : Index r) : Index _ := app [x] xs.
    Module Export IndexNotations1.
      Include IndexNotations0.
      Notation "x :: xs" := (cons x xs) : index_scope.
      Notation "s1 ++ s2" := (app s1 s2) : index_scope.
      Notation "s1 ++' s2" := (app s1 s2) : index_scope.
    End IndexNotations1.



  Module SliceIndexNotations.
    Declare Scope slice_index_scope.
    Delimit Scope slice_index_scope with slice_index.
    Bind Scope slice_index_scope with t.
    Notation "xs ::' x" := (snoc xs x) (at level 59, left associativity, x custom fancy_slice) : raw_index_scope.
    Notation "[ ]" := nil : raw_index_scope.
    Notation "[ x ]" := (snoc nil x) : raw_index_scope.
    Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc nil x) y) .. z) : raw_index_scope.
    Notation "x :: xs" := (cons x xs) : raw_index_scope.
    Notation "s1 ++ s2" := (app s1 s2) : raw_index_scope.
    Notation "s1 ++' s2" := (app s1 s2) : raw_index_scope.

End SliceType.

*)
