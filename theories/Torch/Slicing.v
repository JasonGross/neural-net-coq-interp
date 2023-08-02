From Coq Require Import Sint63 Uint63.
From NeuralNetInterp.Torch Require Import Tensor.
From NeuralNetInterp.Util Require Import Slice Arith.Classes Arith.Instances PolymorphicOption Nat Notations.
Set Implicit Arguments.
(*
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
*)
Set Boolean Equality Schemes.
Set Decidable Equality Schemes.

#[local] Coercion is_true : bool >-> Sortclass.

(* no tensor *)
Inductive SliceIndexType : Rank -> Rank -> Type :=
| slice_index (s : Slice IndexType) : SliceIndexType 1 1
| broadcast_one_index : SliceIndexType 0 1
| single_index (i : IndexType) : SliceIndexType 1 0
.
Inductive FancyIndexType {r} (s : Shape r) : Rank -> Rank -> Type :=
| tensor_index (_ : tensor s IndexType) : FancyIndexType s 1 0
(*| bool_tensor_index {s'} (_ : tensor IndexType (s ::' s')) : FancyIndexType s 1 1*)
| normal_index {ri ro} (_ : SliceIndexType ri ro) : FancyIndexType s ri ro
.
#[global] Arguments normal_index {r s ri ro} _.
#[global] Arguments tensor_index {r s} _.

(* TODO: Figure out bool index tensor; I don't understand how pytorch does boolean broadcasting *)
(*
Definition bool_index_tensor {r} {s : Shape r} {s'} (t : tensor bool (s ::' s')) : FancyIndexType s 1 1.
*)

#[export] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
#[global] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
#[global] Coercion slice_index : Slice >-> SliceIndexType.
#[global] Coercion tensor_index : tensor >-> FancyIndexType.
(*#[global] Coercion bool_tensor_index : tensor >-> FancyIndexType.*)
#[global] Coercion normal_index : SliceIndexType >-> FancyIndexType.
#[export] Set Warnings Append "uniform-inheritance,ambiguous-paths".
#[global] Set Warnings Append "uniform-inheritance,ambiguous-paths".
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
  Notation "'None'" := broadcast_one_index (in custom fancy_slice at level 0). (* need to avoid breaking Datatypes.None *)
  Notation "x" := x%sint63 (in custom fancy_slice at level 59, x constr at level 55).
End FancySlicingNotations.

Module SliceIndex.
  Module SliceIndexType.
    Definition t := SliceIndexType.
    Notation IndexType := t.

    Definition transfer_shape_single_index {ris ros} (ri:Rank:=1%nat) (ro:Rank:=0%nat)
      (transfer_shape_idxs : Shape ris -> Shape ros)
      : Shape (ris +' ri) -> Shape (ros +' ro)
      := fun s => transfer_shape_idxs (Shape.hd s).

    Definition transfer_shape {ris ros ri ro}
      (transfer_shape_idxs : Shape ris -> Shape ros)
      (idx : SliceIndexType ri ro)
      : Shape (ris +' ri) -> Shape (ros +' ro)
      := match idx in Slicing.SliceIndexType ri ro return Shape (ris +' ri) -> Shape (ros +' ro) with
         | slice_index idx
           => fun s
              => Shape.snoc
                   (transfer_shape_idxs (Shape.hd s))
                   match idx with
                   | slice[:] => Shape.tl s
                   | _ => Concrete.length (Slice.norm_concretize idx (Shape.tl s))
                   end
         (*| @slice_tensor_index s' idx
              => fun s
                 => Shape.snoc
                      (transfer_shape_idxs (Shape.hd s))
                      (Shape.item s')*)
         | broadcast_one_index
           => fun s => Shape.snoc (transfer_shape_idxs s) 1
         | single_index _
           => transfer_shape_single_index transfer_shape_idxs
         end.

    Definition slice {ris ros ri ro}
      (transfer_shape_idxs : Shape ris -> Shape ros)
      (slice_idxs : forall {s : Shape ris} {A}, tensor s A -> tensor (transfer_shape_idxs s) A)
      (idx : SliceIndexType ri ro)
      : forall {s : Shape (ris +' ri)} {A}, tensor s A -> tensor (transfer_shape transfer_shape_idxs idx s) A
      := match idx in Slicing.SliceIndexType ri ro return forall (s : Shape (ris +' ri)) {A}, tensor s A -> tensor (transfer_shape transfer_shape_idxs idx s) A with
         | slice_index sl
           => fun s A t idxs' (* adjust slice at last index *)
              => let idx := RawIndex.tl idxs' in
                 @slice_idxs (Shape.hd s) A (fun idxs' => t (RawIndex.snoc idxs' (Slice.invert_index sl (Shape.tl s) idx))) (RawIndex.hd idxs')
         (*| @slice_tensor_index s' sidx
              => fun s t idxs' (* lookup index *)
                 => let idx := RawIndex.tl idxs' in
                    slice_idxs (Shape.hd s) (fun idxs' => t (RawIndex.snoc idxs' (adjust_index_for (Shape.tl s) (Tensor.raw_get sidx [idx])))) (RawIndex.hd idxs')*)
         | broadcast_one_index
           => fun s A t idxs' (* ignore final idxs', which is just 1 *)
              => @slice_idxs s A t (RawIndex.hd idxs')
         | single_index idx
           => fun s A t idxs' (* adjoin idx as final index *)
              => @slice_idxs (Shape.hd s) A (fun idxs' => t (RawIndex.snoc idxs' (adjust_index_for (Shape.tl s) idx))) idxs'
         end.
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
    Notation "x , .. , z" := (snoc .. (snoc nil x) .. z) (in custom slice_index at level 200, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
    Notation "… , x , .. , z" := (snoc .. (snoc elipsis x) .. z) (in custom slice_index at level 60, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
  End SliceIndexNotations0.
  #[local] Open Scope slice_index_scope.

  Import Slice.SlicingNotations.
  Fixpoint transfer_shape {ri ro} (idxs : t ri ro) : Shape ri -> Shape ro
    := match idxs with
       | [] => fun tt => tt
       | … => fun x => x
       | idxs ::' idx => SliceIndexType.transfer_shape (transfer_shape idxs) idx
       end.

  Fixpoint slice {ri ro} {s : Shape ri} (idxs : t ri ro) {A} : tensor s A -> tensor (transfer_shape idxs s) A
    := match idxs in t ri ro return forall {s : Shape ri}, tensor s A -> tensor (transfer_shape idxs s) A with
       | [] => fun _s t idxs' => t tt
       | … => fun _s t idxs' => t idxs'
       | idxs ::' idx
         => fun s => SliceIndexType.slice (transfer_shape idxs) (fun s A => slice idxs) idx
       end s.

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
  Module FancyIndexType.
    Definition t {r} := @FancyIndexType r.
    Notation IndexType := t.
    Definition broadcast {rb} {s_broadcast : Shape rb} {ri ro} (idx : @t rb s_broadcast ri ro) : tensor s_broadcast (SliceIndex.IndexType ri ro)
      := match idx with
         | tensor_index idx
           => Tensor.map single_index idx
         | normal_index idx
           => Tensor.repeat' idx
         end.
    Definition transfer_inner_shape {rb sb ris ros ri ro}
      (transfer_inner_shape_idxs : Shape ris -> Shape ros)
      (idx : @t rb sb ri ro)
      : Shape (ris +' ri) -> Shape (ros +' ro)
      := match idx with
         | normal_index idx
           => SliceIndex.SliceIndexType.transfer_shape transfer_inner_shape_idxs idx
         | tensor_index _
           => SliceIndex.SliceIndexType.transfer_shape_single_index transfer_inner_shape_idxs
         end.
  End FancyIndexType.
  Notation IndexType := FancyIndexType.t.
  Notation FancyIndexType := FancyIndexType.t.

  Inductive t {rb} (s_broadcast : Shape rb) : Rank (* input *) -> Rank (* output *) -> Type :=
  | nil : t s_broadcast 0 0
  | elipsis {r} : t s_broadcast r r
  | snoc {ris ros ri ro} : t s_broadcast ris ros -> FancyIndexType s_broadcast ri ro -> t s_broadcast (ris +' ri) (ros +' ro).
  #[global] Arguments nil {rb s_broadcast}.
  #[global] Arguments elipsis {rb s_broadcast r}.
  #[global] Arguments snoc {rb s_broadcast ris ros ri ro} _ _.
  Notation FancyIndex := t.

  Module Import FancyIndexNotations0.
    Export FancySlicingNotations.
    Declare Scope fancy_index_scope.
    Delimit Scope fancy_index_scope with fancy_index.
    Bind Scope fancy_index_scope with FancyIndex.
    Notation "xs ::' x" := (snoc xs x) : fancy_index_scope.
    Notation "[ ]" := nil : fancy_index_scope.
    Notation "[ x ]" := (snoc nil x) : fancy_index_scope.
    Notation "[ x ; y ; .. ; z ]" := (snoc .. (snoc (snoc nil x) y) .. z) : fancy_index_scope.
    Notation "…" := elipsis : fancy_index_scope.
    Declare Custom Entry fancy_index.
    Notation "x" := (snoc nil x) (in custom fancy_index at level 200, x custom fancy_slice at level 60).
    Notation "x , .. , z" := (snoc .. (snoc nil x) .. z) (in custom fancy_index at level 200, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
    Notation "… , x , .. , z" := (snoc .. (snoc elipsis x) .. z) (in custom fancy_index at level 60, x custom fancy_slice at level 60, z custom fancy_slice at level 60).
  End FancyIndexNotations0.
  #[local] Open Scope fancy_index_scope.
  Import SliceIndex.SliceIndexNotations.
  Import Slice.SlicingNotations.

  Fixpoint transfer_inner_shape {rb sb ri ro} (idxs : @t rb sb ri ro) : Shape ri -> Shape ro
    := match idxs with
       | [] => fun tt => tt
       | … => fun x => x
       | idxs ::' idx => FancyIndexType.transfer_inner_shape (transfer_inner_shape idxs) idx
       end.
  Definition transfer_shape {rb sb ri ro} (idxs : @t rb sb ri ro) (s : Shape ri) : Shape (rb +' ro)
    := sb ++' transfer_inner_shape idxs s.

  Fixpoint broadcast {rb} {s_broadcast : Shape rb} {ri ro} (idxs : @t rb s_broadcast ri ro) : tensor s_broadcast (SliceIndex.t ri ro)
    := match idxs with
       | [] => Tensor.repeat' []%slice_index
       | … => Tensor.repeat' …%slice_index
       | idxs ::' idx
         => Tensor.map2
              (fun idxs idx => idxs ::' idx)%slice_index
              (@broadcast _ _ _ _ idxs)
              (FancyIndexType.broadcast idx)
       end.

  Definition slice_ {rb} {s_broadcast : Shape rb} {ri ro} {s : Shape ri} {A} (idxs : @t rb s_broadcast ri ro) (x : tensor s A)
    : tensor_dep (fun i => tensor (SliceIndex.transfer_shape i s) A) (broadcast idxs)
    := Tensor.map_dep (fun i => SliceIndex.slice i x) (broadcast idxs).

  Definition slice
    {rb} {s_broadcast : Shape rb} {ri ro} {s : Shape ri} {A} (idxs : @t rb s_broadcast ri ro) (x : tensor s A)
    : tensor (transfer_shape idxs s) A
    := reshape_app_combine' (slice_ idxs x).

  Module Import FancyIndexNotations.
    Export FancyIndexNotations0.
    Declare Scope fancy_tensor_scope.
    Declare Scope fancy_raw_tensor_scope.
    Delimit Scope fancy_tensor_scope with fancy_tensor.
    Delimit Scope fancy_raw_tensor_scope with fancy_raw_tensor.
    Notation "t .[ x , .. , y ]"
      := (FancyIndex.slice (snoc .. (snoc nil x) .. y) t%fancy_raw_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ x ,  .. ,  y ]")
        : fancy_raw_tensor_scope.
    Notation "t .[ x , .. , y ]"
      := (FancyIndex.slice (snoc .. (snoc nil x) .. y) t%fancy_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ x ,  .. ,  y ]")
        : fancy_tensor_scope.
    Notation "t .[ … , x , .. , y ]"
      := (FancyIndex.slice (snoc .. (snoc elipsis x) .. y) t%fancy_raw_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ … ,  x ,  .. ,  y ]")
        : fancy_raw_tensor_scope.
    Notation "t .[ … , x , .. , y ]"
      := (FancyIndex.slice (snoc .. (snoc elipsis x) .. y) t%fancy_tensor)
           (at level 2, x custom fancy_slice at level 60, y custom fancy_slice at level 60, left associativity, format "t .[ … ,  x ,  .. ,  y ]")
        : fancy_tensor_scope.
    Notation "t .[< i >]"
      := (FancyIndex.slice (snoc nil i) t%fancy_tensor)
           (at level 2, i custom fancy_slice at level 60, left associativity, format "t .[< i >]")
        : fancy_tensor_scope.
  End FancyIndexNotations.
End FancyIndex.
Export SliceIndex.SliceIndexNotations.
Export FancyIndex.FancyIndexNotations.

(*
Local Open Scope tensor_scope.
Eval cbn in  _.[1:-1:1].
Eval cbn in  _.[1:-1:1,1:-1:1].
Eval cbn in  _.[1:-1].
Eval cbn in  _.[:-1].
Eval cbn in  _.[1:].
Eval cbn in  _.[1::1].
Eval cbn in  _.[1::1,:1].
Eval cbn in  _.[1::1,1:].
Eval cbn in  _.[1::1,1].
Eval cbn in  _.[1:-1:1, None, 0].
Eval cbn in  _.[:, None, 0].
Eval cbn in  _.[:1, None, 0].
Eval cbn in  _.[:-1:1, None, 0].
*)
