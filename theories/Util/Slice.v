(* Implements a view of an array, a la Python lists *)
From NeuralNetInterp.Util Require Import ErrorT.
#[local] Set Primitive Projections.
#[local] Set Implicit Arguments.
#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.
Record Slice I := { start : option I ; stop : option I ; step : option I }.

Definition map {A B} (f : A -> B) (s : Slice A) : Slice B
  := {| start := option_map f s.(start)
     ; stop := option_map f s.(stop)
     ; step := option_map f s.(step) |}.

Module Export SliceNotations.
  Declare Custom Entry slice.
  Notation ":" := (@Build_Slice _ None None None) (in custom slice at level 5).
  Notation "start : stop : step" := (@Build_Slice _ (Some start) (Some stop) (Some step)) (in custom slice at level 5, start constr at level 59, stop constr at level 59, step constr at level 59, format "start : stop : step").
  Notation "start :: step" := (@Build_Slice _ (Some start) None (Some step)) (in custom slice at level 5, start constr at level 59, step constr at level 59, format "start :: step").
  Notation "start : : step" := (@Build_Slice _ (Some start) None (Some step)) (in custom slice at level 5, start constr at level 59, step constr at level 59, format "start : : step").
  Notation "start :: step" := (@Build_Slice _ (Some start) None (Some step)) (in custom slice at level 5, start constr at level 59, step constr at level 59, format "start :: step").
  Notation ": stop : step" := (@Build_Slice _ None (Some stop) (Some step)) (in custom slice at level 5, stop constr at level 59, step constr at level 59, format ": stop : step").
  Notation ": : step" := (@Build_Slice _ None None (Some step)) (in custom slice at level 5, step constr at level 59, format ": : step").
  Notation ":: step" := (@Build_Slice _ None None (Some step)) (in custom slice at level 5, step constr at level 59, format ":: step").
  Notation "start : stop" := (@Build_Slice _ (Some start) (Some stop) None) (in custom slice at level 5, start constr at level 59, stop constr at level 59, format "start : stop").
  Notation "start :" := (@Build_Slice _ (Some start) None None) (in custom slice at level 5, start constr at level 59, format "start :").
  Notation ": stop" := (@Build_Slice _ None (Some stop) None) (in custom slice at level 5, stop constr at level 59, format ": stop").
End SliceNotations.


Module Export SlicingNotations.
  Declare Custom Entry slicing.
  Notation "x" := x (in custom slicing at level 70, x custom slice at level 69).
End SlicingNotations.
(*
Class SliceableBy ArrayType SliceType ValueType
*)
