(* Implements a view of an array, a la Python lists *)
From NeuralNetInterp Require Import Util.Classes.
From NeuralNetInterp.Util Require Import ErrorT Arith.Classes PolymorphicOption.
#[local] Set Primitive Projections.
#[local] Set Implicit Arguments.
#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.
Record Slice I := { start : option I ; stop : option I ; step : option I }.

Module Concrete.
  Record Slice I := { start : I ; stop : option I ; step : I ; base_len : I }.

  Definition map {A B} (f : A -> B) (base_len_map : A -> B) (s : Slice A) : Slice B
    := {| start := f s.(start)
       ; stop := option_map f s.(stop)
       ; step := f s.(step)
       ; base_len := base_len_map s.(base_len) |}.

  Section with_ops.
    Context {I}
      {modulo : has_mod I} {add : has_add I} {sub : has_sub I} {div : has_int_div I} {one : has_one I} {zero : has_zero I} {ltb : has_ltb I} {opp : has_opp I}.
    Local Open Scope core_scope.
    Definition normalize (s : Slice I) : Slice I
      := {| start := s.(start) mod s.(base_len)
         ; stop := let stop := if s.(step) <? 1 then s.(stop) else Some (Option.value s.(stop) s.(base_len)) in
                   option_map (fun stop => ((stop - 1) mod s.(base_len)) + 1) stop
         ; step := s.(step)
         ; base_len := s.(base_len) |}.

    Definition raw_length (s : Slice I) : I
      := if s.(step) <? 1
         then match s.(stop) with
              | Some stop => 1 + ((stop+1) - s.(start)) // s.(step)
              | None => 1 + (0 - s.(start)) // s.(step)
              end
         else 1 + (Option.value s.(stop) s.(base_len) - s.(start) - 1) // s.(step).

    Definition length (s : Slice I) : I
      := raw_length (normalize s).
  End with_ops.
End Concrete.

Module ConcreteProjections.
  Export Concrete (start, stop, step, base_len).
End ConcreteProjections.

Definition map {A B} (f : A -> B) (s : Slice A) : Slice B
  := {| start := option_map f s.(start)
     ; stop := option_map f s.(stop)
     ; step := option_map f s.(step) |}.

Definition concretize {I} {zero : has_zero I} {one : has_one I} {ltb : has_ltb I} {sub : has_sub I} (s : Slice I) (len : I) : Concrete.Slice I
  := let step := Option.value s.(step) one in
     {| Concrete.start := Option.value s.(start) (if step <? zero then (len - 1) else zero)
     ; Concrete.stop := s.(stop)
     ; Concrete.step := step
     ; Concrete.base_len := len |}%core.

Definition norm_concretize {I} {modulo : has_mod I} {add : has_add I} {sub : has_sub I} {one : has_one I} {zero : has_zero I} {ltb : has_ltb I} (s : Slice I) (len : I) : Concrete.Slice I
  := Concrete.normalize (concretize s len).

Definition unconcretize {I} (s : Concrete.Slice I) : Slice I
  := {| start := Some s.(Concrete.start)
     ; stop := s.(Concrete.stop)
     ; step := Some s.(Concrete.step) |}.
#[global] Coercion unconcretize : Concrete.Slice >-> Slice.

Definition invert_index {I} {modulo : has_mod I} {add : has_add I} {sub : has_sub I} {one : has_one I} {mul : has_mul I} {div : has_int_div I} {zero : has_zero I} {ltb : has_ltb I} (s : Slice I) (base_len : I) (idx : I) : I
  := let via_concretize _
       := let len := Concrete.length (concretize s base_len) in
          match norm_concretize s base_len with
          | Concrete.Build_Slice start _ step _
            => (start + (idx mod len) * step) mod base_len
          end%core in
     match s with
     | Build_Slice None _ None => idx
     | Build_Slice (Some start) _ None
       => (start + idx) mod base_len
     | Build_Slice None _ (Some step)
       => if step <? 0
          then via_concretize tt
          else if idx <? 0
               then (idx * step) mod base_len
               else idx * step
     | Build_Slice (Some start) None (Some step)
       => if step <? 0
          then via_concretize tt
          else if idx <? 0
               then (start + idx * step) mod base_len
               else start + idx * step
     | Build_Slice (Some _) (Some _) (Some _)
       => via_concretize tt
     end%core.

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
  Notation "'slice[' x ]" := x (x custom slicing).
End SlicingNotations.
(*
Class SliceableBy ArrayType SliceType ValueType
*)
