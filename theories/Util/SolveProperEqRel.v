From Coq Require Import Morphisms.

Ltac make_eq_rel T :=
  lazymatch T with
  | (?A -> ?B)
    => let RB := make_eq_rel B in
      constr:(@respectful A B (@eq A) RB)
  | (forall a : ?A, ?B)
    => let B' := fresh in
      constr:(@forall_relation A (fun a : A => B) (fun a : A => match B with B' => ltac:(let B'' := (eval cbv delta [B'] in B') in
                                                                                  let RB := make_eq_rel B in
                                                                                  exact RB) end))
  | _ => constr:(@eq T)
  end.
Ltac solve_Proper_eq :=
  match goal with
  | [ |- @Proper ?A ?R ?f ]
    => let R' := make_eq_rel A in
      unify R R';
      apply (@reflexive_proper A R')
  end.

#[export] Hint Extern 0 (Proper _ _) => solve_Proper_eq : typeclass_instances.
