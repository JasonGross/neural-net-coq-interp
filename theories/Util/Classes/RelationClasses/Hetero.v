(* -*- coding: utf-8 -*- *)
(* Mostly copied from Coq *)
From Coq.Classes Require Export Init.
From Coq.Program Require Import Basics Tactics.
From Coq.Relations Require Import Relation_Definitions.
From Coq.Classes Require Import RelationClasses.
From NeuralNetInterp.Util.Relations Require Import Relation_Definitions.Hetero.

Generalizable Variables A B C D R S T U l eqA eqB eqC eqD.

Section Defs.
  (** We rebind relational properties in separate classes to be able to overload each proof. *)

  Definition complement {A A'} (R : relation A A') : relation A A' := fun x y => R x y -> False.

  (** Opaque for proof-search. *)
  Typeclasses Opaque complement.

  (** These are convertible. *)
  Lemma complement_inverse {A A'} (R : relation A A') : complement (flip R) = flip (complement R).
  Proof. reflexivity. Qed.

  Class subrelation {A A'} (R R' : relation A A') : Prop :=
    is_subrelation : forall {x y}, R x y -> R' x y.

End Defs.

Unset Implicit Arguments.

Local Obligation Tactic := try solve [ simpl_relation ].

(** We now develop a generalization of results on relations for arbitrary predicates.
   The resulting theory can be applied to homogeneous binary relations but also to
   arbitrary n-ary predicates. *)

Local Open Scope list_scope.

(** A compact representation of non-dependent arities, with the codomain singled-out. *)

(* Note, we do not use [list Type] because it imposes unnecessary universe constraints *)
Local Infix "::" := Tcons.
(*
(** We can define abbreviations for operation and relation types based on [arrows]. *)

Definition unary_operation R A := arrows (A::Tnil) R.
Definition binary_operation R A B := arrows (A::B::Tnil) R.
Definition ternary_operation R A B C := arrows (A::B::C::Tnil) R.

(** Heterogeneous binary relations, equivalent to [Hetero.relation A B]. *)

Definition binary_relation A B := predicate (A::B::Tnil).

(** Pointwise extension of a binary operation on [T] to a binary operation
   on functions whose codomain is [T].
   For an operator on [Prop] this lifts the operator to a binary operation. *)

Fixpoint pointwise_extension {X A B : Type} (op : binary_operation X A B)
  (l : Tlist) : binary_operation (arrows l X) (arrows l A) (arrows l B) :=
  match l with
    | Tnil => fun R R' => op R R'
    | A :: tl => fun R R' =>
      fun x => pointwise_extension op tl (R x) (R' x)
  end.

(*
(** Pointwise lifting, equivalent to doing [pointwise_extension] and closing using [predicate_all]. *)

Fixpoint pointwise_lifting (op : binary_relation Prop)  (l : Tlist) : binary_relation (predicate l) :=
  match l with
    | Tnil => fun R R' => op R R'
    | A :: tl => fun R R' =>
      forall x, pointwise_lifting op tl (R x) (R' x)
  end.

(** The n-ary equivalence relation, defined by lifting the 0-ary [iff] relation. *)

Definition predicate_equivalence {l : Tlist} : binary_relation (predicate l) :=
  pointwise_lifting iff l.

(** The n-ary implication relation, defined by lifting the 0-ary [impl] relation. *)

Definition predicate_implication {l : Tlist} :=
  pointwise_lifting impl l.
*)
(** Notations for pointwise equivalence and implication of predicates. *)
Declare Scope hetero_predicate_scope.
(*

Infix "<∙>" := predicate_equivalence (at level 95, no associativity) : hetero_predicate_scope.
Infix "-∙>" := predicate_implication (at level 70, right associativity) : hetero_predicate_scope.
*)
Local Open Scope hetero_predicate_scope.

(** The pointwise liftings of conjunction and disjunctions.
   Note that these are [binary_operation]s, building new relations out of old ones. *)
(*
Definition predicate_intersection := pointwise_extension and.
Definition predicate_union := pointwise_extension or.

Infix "/∙\" := predicate_intersection (at level 80, right associativity) : hetero_predicate_scope.
Infix "\∙/" := predicate_union (at level 85, right associativity) : hetero_predicate_scope.
*)
(** The always [True] and always [False] predicates. *)
(*
Fixpoint true_predicate {l : Tlist} : predicate l :=
  match l with
    | Tnil => True
    | A :: tl => fun _ => @true_predicate tl
  end.

Fixpoint false_predicate {l : Tlist} : predicate l :=
  match l with
    | Tnil => False
    | A :: tl => fun _ => @false_predicate tl
  end.
*)
Notation "∙⊤∙" := true_predicate : hetero_predicate_scope.
Notation "∙⊥∙" := false_predicate : hetero_predicate_scope.

(** We define the various operations which define the algebra on binary relations,
   from the general ones. *)
*)
Section Binary.
  Context {A A' : Type}.

  (*Definition relation_equivalence : relation (relation A) :=
    @predicate_equivalence (_::_::Tnil).
   *)

  Definition relation_conjunction (R : Hetero.relation A A') (R' : Hetero.relation A A') : Hetero.relation A A' :=
    @predicate_intersection (A::A'::Tnil) R R'.

  Definition relation_disjunction (R : Hetero.relation A A') (R' : Hetero.relation A A') : Hetero.relation A A' :=
    @predicate_union (A::A'::Tnil) R R'.
End Binary.
(*
Global Typeclasses Opaque arrows predicate_implication predicate_equivalence
            relation_equivalence pointwise_lifting.
*)
