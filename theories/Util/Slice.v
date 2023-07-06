(* Implements a view of an array, a la Python lists *)
From NeuralNetInterp Require Import Util.Classes.
From NeuralNetInterp.Util Require Import ErrorT Arith.Classes Option.
#[local] Set Primitive Projections.
#[local] Set Implicit Arguments.
#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.
Record Slice I := { start : option I ; stop : option I ; step : option I }.

Module Concrete.
  Record Slice I := { start : I ; stop : I ; step : I }.

  Definition map {A B} (f : A -> B) (s : Slice A) : Slice B
    := {| start := f s.(start)
       ; stop := f s.(stop)
       ; step := f s.(step) |}.

  Definition normalize {I} {modulo : has_mod I} {add : has_add I} {sub : has_sub I} {one : has_one I} (s : Slice I) (len : I) : Slice I
    := {| start := s.(start) mod len
       ; stop := ((s.(stop) - 1) mod len) + 1
       ; step := s.(step) |}%core.

  Definition length {I} {sub : has_sub I} {div : has_int_div I} {one : has_one I} {add : has_add I}
    (s : Slice I) : I
    := (1 + (s.(stop) - s.(start) - 1) // s.(step))%core.
End Concrete.

Module ConcreteProjections.
  Export Concrete (start, stop, step).
End ConcreteProjections.

Definition map {A B} (f : A -> B) (s : Slice A) : Slice B
  := {| start := option_map f s.(start)
     ; stop := option_map f s.(stop)
     ; step := option_map f s.(step) |}.

Definition concretize {I} {zero : has_zero I} {one : has_one I} (s : Slice I) (len : I) : Concrete.Slice I
  := {| Concrete.start := Option.value s.(start) zero
     ; Concrete.stop := Option.value s.(stop) len
     ; Concrete.step := Option.value s.(step) one |}.

Definition norm_concretize {I} {modulo : has_mod I} {add : has_add I} {sub : has_sub I} {one : has_one I} {zero : has_zero I} (s : Slice I) (len : I) : Concrete.Slice I
  := Concrete.normalize (concretize s len) len.

Definition unconcretize {I} (s : Concrete.Slice I) : Slice I
  := {| start := Some s.(Concrete.start)
     ; stop := Some s.(Concrete.stop)
     ; step := Some s.(Concrete.step) |}.
#[global] Coercion unconcretize : Concrete.Slice >-> Slice.

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
