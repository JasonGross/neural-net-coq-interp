From Coq Require Import Sint63 Uint63.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Slice Arith.Classes Arith.Instances PolymorphicOption Nat Notations.
Set Implicit Arguments.
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Set Boolean Equality Schemes.
Set Decidable Equality Schemes.

#[local] Coercion is_true : bool >-> Sortclass.

(* no tensor *)
Inductive SliceIndexType@{u} : Rank -> Rank -> Type@{u} :=
| slice_index (s : Slice IndexType@{u}) : SliceIndexType 1 1
| broadcast_one_index : SliceIndexType 0 1
| single_index (i : IndexType@{u}) : SliceIndexType 1 0
| condition_index (b : IndexType@{u} -> bool) : SliceIndexType 1 1
.
Inductive FancyIndexType {r} (s : Shape r) : Rank -> Rank -> Type :=
| tensor_index (_ : @tensor r IndexType s) : FancyIndexType s 1 0
| bool_tensor_index (_ : @tensor r bool s) : FancyIndexType s 1 0
| normal_index {ri ro} (_ : SliceIndexType ri ro) : FancyIndexType s ri ro
.
Set Printing Universes.
Set Printing All.
Print FancyIndexType.
Print tensor_of_rank.
Print R
TODO BOOL indexing
Print tensor_of_rank.
Set Printing All.
Print RawIndex.t.
#[global] Arguments normal_index {r s ri ro} _.
#[global] Arguments tensor_index {r s} _.
#[export] Set Warnings Append "-uniform-inheritance".
#[global] Set Warnings Append "-uniform-inheritance".
#[global] Coercion slice_index : Slice >-> SliceIndexType.
#[global] Coercion tensor_index : tensor >-> FancyIndexType.
#[global] Coercion bool_tensor_index : tensor >-> FancyIndexType.
#[global] Coercion normal_index : SliceIndexType >-> FancyIndexType.
#[export] Set Warnings Append "uniform-inheritance".
#[global] Set Warnings Append "uniform-inheritance".
#[global] Coercion single_index : IndexType >-> SliceIndexType.
#[global] Coercion inject_int (x : int) : IndexType := x.

Module FancySlicingNotations.
  Declare Custom Entry fancy_slice.
  Notation ":" := (slice_index (@Build_Slice _ None None None)) (in custom fancy_slice at level 59).
  Notation "start : stop : step" := (slice_index (@Build_Slice _ (Some start) (Some stop) (Some step))) (in custom fancy_slice at level 59, start constr at level 59, stop constr at level 59, step constr at level 59, format "start : stop : step").
  Notation "start :: step" := (slice_index (@Build_Slice _ (Some start) None (Some step))) (in custom fancy_slice at level 59, start constr at level 59, step constr at level 59, format "start :: step").
  Notation "start : : step" := (slice_index (@Build_Slice _ (Some start) None (Some step))) (in custom fancy_slice at level 59, start constr at level 59, step constr at level 59, format "start : : step").
  Notation "start :: step" := (slice_index (@Build_Slice _ (Some start) None (Some step))) (in custom fancy_slice at level 59, start constr at level 59, step constr at level 59, format "start :: step").
  Notation ": stop : step" := (slice_index (@Build_Slice _ None (Some stop) (Some step))) (in custom fancy_slice at level 59, stop constr at level 59, step constr at level 59, format ": stop : step").
  Notation ": : step" := (slice_index (@Build_Slice _ None None (Some step))) (in custom fancy_slice at level 59, step constr at level 59, format ": : step").
  Notation ":: step" := (slice_index (@Build_Slice _ None None (Some step))) (in custom fancy_slice at level 59, step constr at level 59, format ":: step").
  Notation "start : stop" := (slice_index (@Build_Slice _ (Some start) (Some stop) None)) (in custom fancy_slice at level 59, start constr at level 59, stop constr at level 59, format "start : stop").
  Notation "start :" := (slice_index (@Build_Slice _ (Some start) None None)) (in custom fancy_slice at level 59, start constr at level 59, format "start :").
  Notation ": stop" := (slice_index (@Build_Slice _ None (Some stop) None)) (in custom fancy_slice at level 59, stop constr at level 59, format ": stop").
  Notation "'None'" := broadcast_one_index (in custom fancy_slice at level 59).
  Notation "x" := x%sint63 (in custom fancy_slice at level 59, x constr at level 55).
End FancySlicingNotations.

Module SliceIndex.
  Module SliceIndexType.
    Definition t := SliceIndexType.
    Notation IndexType := t.
  End SliceIndexType.
  Notation IndexType := SliceIndexType.t.
  Notation SliceIndexType := SliceIndexType.t.

  Inductive t : Rank (* input *) -> Rank (* output *) -> Type :=
  | nil : t 0 0
  | elipsis {r} : t r r
  | snoc {ris ros ri ro} : t ris ros -> SliceIndexType ri ro -> t (ris +' ri) (ros +' ro).
  Notation SliceIndex := t.

  Module Import SliceIndexNotations0.
    Export FancySlicingNotations.
    Declare Scope slice_index_scope.
    Delimit Scope slice_index_scope with slice_index.
    Bind Scope slice_index_scope with SliceIndex.
    Notation "xs ::' x" := (snoc xs x) : slice_index_scope.
    Notation "[ ]" := nil : slice_index_scope.
    Notation "[ x ]" := (snoc nil x) : slice_index_scope.
    Notation "[ x ; y ; .. ; z ]" := (snoc .. (snoc (snoc nil x) y) .. z) : slice_index_scope.
    Notation "…" := elipsis : slice_index_scope.
    Declare Custom Entry slice_index.
    Notation "x" := (snoc nil x) (in custom slice_index at level 200, x custom fancy_slice at level 60).
    Notation "x , .. , z" := (snoc .. (snoc nil x) .. z) (in custom slice at level 200, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
    Notation "… , x , .. , z" := (snoc .. (snoc elipsis x) .. z) (in custom slice at level 60, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
  End SliceIndexNotations0.
  #[local] Open Scope slice_index_scope.

  Import Slice.SlicingNotations.
  Fixpoint transfer_shape {ri ro} (idxs : t ri ro) : Shape ri -> Shape ro
    := match idxs with
       | [] => fun tt => tt
       | … => fun x => x
       | @snoc ris ros ri ro idxs idx
         => match idx in Slicing.SliceIndexType ri ro return Shape (ris +' ri) -> Shape (ros +' ro) with
            | slice_index idx
              => fun s
                 => Shape.snoc
                      (@transfer_shape ris ros idxs (Shape.hd s))
                      match idx with
                      | slice[:] => Shape.tl s
                      | _ => Concrete.length (Slice.norm_concretize idx (Shape.tl s))
                      end
            | broadcast_one_index
              => fun s => Shape.snoc (@transfer_shape ris ros idxs s) 1
            | single_index idx
              => fun s => @transfer_shape ris ros idxs (Shape.hd s)
            end
       end.

  Fixpoint slice {A ri ro} (idxs : t ri ro) : forall {s : Shape ri}, tensor A s -> tensor A (transfer_shape idxs s)
    := match idxs with
       | [] => fun _s t idxs' => t tt
       | … => fun _s t idxs' => t idxs'
       | @snoc ris ros ri ro idxs idx
         => match idx in Slicing.SliceIndexType ri ro return forall s : Shape (ris +' ri), tensor A s -> tensor A (transfer_shape (idxs ::' idx) s) with
            | slice_index sl
              => fun s t idxs' (* adjust slice at last index *)
                 => let idx := RawIndex.tl idxs' in
                    @slice A ris ros idxs (Shape.hd s) (fun idxs' => t (RawIndex.snoc idxs' (Slice.invert_index sl (Shape.tl s) idx))) (RawIndex.hd idxs')
            | broadcast_one_index
              => fun s t idxs' (* ignore final idxs', which is just 1 *)
                 => @slice A ris ros idxs s t (RawIndex.hd idxs')
            | single_index idx
              => fun s t idxs' (* adjoin idx as final index *)
                 => @slice A ris ros idxs (Shape.hd s) (fun idxs' => t (RawIndex.snoc idxs' (adjust_index_for (Shape.tl s) idx))) idxs'
            end
       end.

  Module Import SliceIndexNotations.
    Export SliceIndexNotations0.
    Notation "t .[ x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc nil x) .. y) t%raw_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ x ,  .. ,  y ]")
        : raw_tensor_scope.
    Notation "t .[ x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc nil x) .. y) t%tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ x ,  .. ,  y ]")
        : tensor_scope.
    Notation "t .[ … , x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc elipsis x) .. y) t%raw_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ … ,  x ,  .. ,  y ]")
        : raw_tensor_scope.
    Notation "t .[ … , x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc elipsis x) .. y) t%tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ … ,  x ,  .. ,  y ]")
        : tensor_scope.
    Notation "t .[< i >]"
      := (SliceIndex.slice (snoc nil i) t%tensor)
           (at level 2, i custom fancy_slice at level 60, left associativity, format "t .[< i >]")
        : tensor_scope.
  End SliceIndexNotations.
End SliceIndex.

Module FancyIndex.
  About FancyIndexType.
  Module FancyIndexType.
    Definition t {r} := @FancyIndexType r.
    Notation IndexType := t.
  End FancyIndexType.
  Notation IndexType := FancyIndexType.t.
  Notation FancyIndexType := FancyIndexType.t.

  Inductive t : Rank (* input *) -> Rank (* output *) -> Type :=
  | nil : t 0 0
  | elipsis {r} : t r r
  | snoc {ris ros ri ro} : t ris ros -> SliceIndexType ri ro -> t (ris +' ri) (ros +' ro).
  Notation SliceIndex := t.

  Module Import SliceIndexNotations0.
    Export FancySlicingNotations.
    Declare Scope slice_index_scope.
    Delimit Scope slice_index_scope with slice_index.
    Bind Scope slice_index_scope with SliceIndex.
    Notation "xs ::' x" := (snoc xs x) : slice_index_scope.
    Notation "[ ]" := nil : slice_index_scope.
    Notation "[ x ]" := (snoc nil x) : slice_index_scope.
    Notation "[ x ; y ; .. ; z ]" := (snoc .. (snoc (snoc nil x) y) .. z) : slice_index_scope.
    Notation "…" := elipsis : slice_index_scope.
    Declare Custom Entry slice_index.
    Notation "x" := (snoc nil x) (in custom slice_index at level 200, x custom fancy_slice at level 60).
    Notation "x , .. , z" := (snoc .. (snoc nil x) .. z) (in custom slice at level 200, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
    Notation "… , x , .. , z" := (snoc .. (snoc elipsis x) .. z) (in custom slice at level 60, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
  End SliceIndexNotations0.
  #[local] Open Scope slice_index_scope.

  Import Slice.SlicingNotations.
  Fixpoint transfer_shape {ri ro} (idxs : t ri ro) : Shape ri -> Shape ro
    := match idxs with
       | [] => fun tt => tt
       | … => fun x => x
       | @snoc ris ros ri ro idxs idx
         => match idx in Slicing.SliceIndexType ri ro return Shape (ris +' ri) -> Shape (ros +' ro) with
            | slice_index idx
              => fun s
                 => Shape.snoc
                      (@transfer_shape ris ros idxs (Shape.hd s))
                      match idx with
                      | slice[:] => Shape.tl s
                      | _ => Concrete.length (Slice.norm_concretize idx (Shape.tl s))
                      end
            | broadcast_one_index
              => fun s => Shape.snoc (@transfer_shape ris ros idxs s) 1
            | single_index idx
              => fun s => @transfer_shape ris ros idxs (Shape.hd s)
            end
       end.

  Fixpoint slice {A ri ro} (idxs : t ri ro) : forall {s : Shape ri}, tensor A s -> tensor A (transfer_shape idxs s)
    := match idxs with
       | [] => fun _s t idxs' => t tt
       | … => fun _s t idxs' => t idxs'
       | @snoc ris ros ri ro idxs idx
         => match idx in Slicing.SliceIndexType ri ro return forall s : Shape (ris +' ri), tensor A s -> tensor A (transfer_shape (idxs ::' idx) s) with
            | slice_index sl
              => fun s t idxs' (* adjust slice at last index *)
                 => let idx := RawIndex.tl idxs' in
                    @slice A ris ros idxs (Shape.hd s) (fun idxs' => t (RawIndex.snoc idxs' (Slice.invert_index sl (Shape.tl s) idx))) (RawIndex.hd idxs')
            | broadcast_one_index
              => fun s t idxs' (* ignore final idxs', which is just 1 *)
                 => @slice A ris ros idxs s t (RawIndex.hd idxs')
            | single_index idx
              => fun s t idxs' (* adjoin idx as final index *)
                 => @slice A ris ros idxs (Shape.hd s) (fun idxs' => t (RawIndex.snoc idxs' (adjust_index_for (Shape.tl s) idx))) idxs'
            end
       end.

  Module Import SliceIndexNotations.
    Export SliceIndexNotations0.
    Notation "t .[ x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc nil x) .. y) t%raw_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ x ,  .. ,  y ]")
        : raw_tensor_scope.
    Notation "t .[ x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc nil x) .. y) t%tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ x ,  .. ,  y ]")
        : tensor_scope.
    Notation "t .[ … , x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc elipsis x) .. y) t%raw_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ … ,  x ,  .. ,  y ]")
        : raw_tensor_scope.
    Notation "t .[ … , x , .. , y ]"
      := (SliceIndex.slice (snoc .. (snoc elipsis x) .. y) t%tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ … ,  x ,  .. ,  y ]")
        : tensor_scope.
    Notation "t .[< i >]"
      := (SliceIndex.slice (snoc nil i) t%tensor)
           (at level 2, i custom fancy_slice at level 60, left associativity, format "t .[< i >]")
        : tensor_scope.
  End SliceIndexNotations.
End SliceIndex.

Export SliceIndex.SliceIndexNotations.

(*
Local Open Scope tensor_scope.
Eval cbn in  _.[:, None, 0].
*)

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
(*FIXME FANCY SLICING*)
