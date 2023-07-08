From Coq Require Import String.
Export StringSyntax.
Definition with_default (name : string) {A} (x : A) := A.
#[global] Arguments with_default _ {_} _, _ _ _.
Existing Class with_default.
#[global] Typeclasses Opaque with_default.
Ltac fill_default _ :=
  lazymatch goal with
  | [ |- @with_default ?name ?A ?x ]
    => match goal with
       | [ H : @with_default ?name' ?A' _ |- _ ] => constr_eq A A'; constr_eq name name'; fail 1
       | _ => exact x
       end
  end.
#[global] Hint Extern 0 (with_default _ _) => fill_default () : typeclass_instances.
#[global] Hint Unfold with_default : typeclass_instances.
