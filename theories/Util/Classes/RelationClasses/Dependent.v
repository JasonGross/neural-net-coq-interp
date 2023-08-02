(* -*- coding: utf-8 -*- *)
(* Mostly copied from Coq *)
From Coq.Classes Require Export Init.
From Coq.Program Require Import Basics Tactics.
From Coq.Relations Require Import Relation_Definitions.
From Coq.Classes Require Import RelationClasses.
From NeuralNetInterp.Util.Program Require Import Basics.Dependent.
From NeuralNetInterp.Util.Relations Require Import Relation_Definitions.Dependent.
From NeuralNetInterp.Util.Classes Require Import Morphisms.Dependent.

Generalizable Variables A B C D R S T U l eqA eqB eqC eqD.

Section Defs.
  Context {A : Type -> Type}.

  (** We rebind relational properties in separate classes to be able to overload each proof. *)

  Class Reflexive (R : Dependent.relation A) :=
    reflexivity : forall T R', @RelationClasses.Reflexive T R' -> @RelationClasses.Reflexive (A T) (R T T R').
  #[export] Existing Instance reflexivity.

  Definition complement (R : Dependent.relation A) : Dependent.relation A
    := fun x y R' a b => R x y R' a b -> False.

  (** Opaque for proof-search. *)
  Typeclasses Opaque complement.

  (*
  (** These are convertible. *)
  Lemma complement_inverse R : complement (flip R) = flip (complement R).
  Proof. reflexivity. Qed.
*)
  Class Irreflexive (R : Dependent.relation A) :=
    irreflexivity : Reflexive (complement R).

  Class Symmetric (R : Dependent.relation A) :=
    symmetry : forall {x y} {R' : Hetero.relation x y} {a b}, R x y R' a b -> R y x (flip R') b a.

  Class Asymmetric (R : Dependent.relation A) :=
    asymmetry : forall {x y} {R' : Hetero.relation x y} {a b}, R x y R' a b -> R y x (flip R') b a -> False.

  Class Transitive (R : Dependent.relation A) :=
    transitivity : forall {x y z} {Rxy : Hetero.relation x y} {Ryz : Hetero.relation y z} {Rxz : Hetero.relation x z}, (forall a b c, Rxy a b -> Ryz b c -> Rxz a c) -> forall {a b c}, R x y Rxy a b -> R y z Ryz b c -> R x z Rxz a c.

  (** Various combinations of reflexivity, symmetry and transitivity. *)

  (** A [PreOrder] is both Reflexive and Transitive. *)

  Class PreOrder (R : Dependent.relation A) : Prop := {
    #[global] PreOrder_Reflexive :: Reflexive R | 2 ;
    #[global] PreOrder_Transitive :: Transitive R | 2 }.

  (** A [StrictOrder] is both Irreflexive and Transitive. *)

  Class StrictOrder (R : Dependent.relation A) : Prop := {
    #[global] StrictOrder_Irreflexive :: Irreflexive R ;
    #[global] StrictOrder_Transitive :: Transitive R }.

  (** By definition, a strict order is also asymmetric *)
  Global Instance StrictOrder_Asymmetric `(H : StrictOrder R) : Asymmetric R.
  Proof using Type.
    repeat intro; eapply irreflexivity; [ | eapply transitivity; [ | eassumption .. ]; cbv ].
    hnf.
    (* cheat *)
    Unshelve.
    3: refine (fun X Y => X = Y \/ exists b0, ?[R'] b0 X /\ ?R' b0 Y).
    1: left; reflexivity.
    1: right; eexists; split; eassumption.
  Qed.

  (** A partial equivalence relation is Symmetric and Transitive. *)

  Class PER (R : Dependent.relation A) : Prop := {
    #[global] PER_Symmetric :: Symmetric R | 3 ;
    #[global] PER_Transitive :: Transitive R | 3 }.

  (** Equivalence Dependent.relations. *)

  Class Equivalence (R : Dependent.relation A) : Prop := {
    #[global] Equivalence_Reflexive :: Reflexive R ;
    #[global] Equivalence_Symmetric :: Symmetric R ;
    #[global] Equivalence_Transitive :: Transitive R }.

  (** An Equivalence is a PER plus reflexivity. *)

  Global Instance Equivalence_PER {R} `(E:Equivalence R) : PER R | 10 :=
    { }.

  (** An Equivalence is a PreOrder plus symmetry. *)

  Global Instance Equivalence_PreOrder {R} `(E:Equivalence R) : PreOrder R | 10 :=
    { }.

  (*
  (** We can now define antisymmetry w.r.t. an equivalence relation on the carrier. *)

  Class Antisymmetric eqA `{equ : Equivalence eqA} (R : Dependent.relation A) :=
    antisymmetry : forall {x y}, R x y -> R y x -> eqA x y.
*)
  Class subrelation (R R' : Dependent.relation A) : Prop :=
    is_subrelation : forall {x y R0 a b}, R x y R0 a b -> R' x y R0 a b.

  (*
  (** Any symmetric relation is equal to its inverse. *)

  Lemma subrelation_symmetric R `(Symmetric R) : subrelation (flip R) R.
  Proof. hnf. intros x y H0. red in H0. apply symmetry. assumption. Qed.

  Section flip.

    Lemma flip_Reflexive `{Reflexive R} : Reflexive (flip R).
    Proof. tauto. Qed.

    Program Definition flip_Irreflexive `(Irreflexive R) : Irreflexive (flip R) :=
      irreflexivity (R:=R).

    Program Definition flip_Symmetric `(Symmetric R) : Symmetric (flip R) :=
      fun x y H => symmetry (R:=R) H.

    Program Definition flip_Asymmetric `(Asymmetric R) : Asymmetric (flip R) :=
      fun x y H H' => asymmetry (R:=R) H H'.

    Program Definition flip_Transitive `(Transitive R) : Transitive (flip R) :=
      fun x y z H H' => transitivity (R:=R) H' H.

    Program Definition flip_Antisymmetric `(Antisymmetric eqA R) :
      Antisymmetric eqA (flip R).
    Proof. firstorder. Qed.

    (** Inversing the larger structures *)

    Lemma flip_PreOrder `(PreOrder R) : PreOrder (flip R).
    Proof. firstorder. Qed.

    Lemma flip_StrictOrder `(StrictOrder R) : StrictOrder (flip R).
    Proof. firstorder. Qed.

    Lemma flip_PER `(PER R) : PER (flip R).
    Proof. firstorder. Qed.

    Lemma flip_Equivalence `(Equivalence R) : Equivalence (flip R).
    Proof. firstorder. Qed.

  End flip.

  Section complement.

    Definition complement_Irreflexive `(Reflexive R)
      : Irreflexive (complement R).
    Proof. firstorder. Qed.

    Definition complement_Symmetric `(Symmetric R) : Symmetric (complement R).
    Proof. firstorder. Qed.
  End complement.
*)
End Defs.


#[export] Hint Extern 2 (@RelationClasses.Reflexive _ (?R ?T ?T ?R')) => notypeclasses refine (@reflexivity _ R _ T R' _) : typeclass_instances.

#[export] Instance const_Reflexive {T R} {_ : @RelationClasses.Reflexive T R} : @Reflexive (fun _ => T) (Dependent.const R).
Proof. cbv in *; eauto. Qed.

(** Hints to drive the typeclass resolution avoiding loops
 due to the use of full unification. *)
#[global]
Hint Extern 1 (Reflexive (complement _)) => class_apply @irreflexivity : typeclass_instances.
#[global]
Hint Extern 3 (Symmetric (complement _)) => class_apply complement_Symmetric : typeclass_instances.
#[global]
Hint Extern 3 (Irreflexive (complement _)) => class_apply complement_Irreflexive : typeclass_instances.

#[global]
Hint Extern 3 (Reflexive (flip _)) => apply flip_Reflexive : typeclass_instances.
#[global]
Hint Extern 3 (Irreflexive (flip _)) => class_apply flip_Irreflexive : typeclass_instances.
#[global]
Hint Extern 3 (Symmetric (flip _)) => class_apply flip_Symmetric : typeclass_instances.
#[global]
Hint Extern 3 (Asymmetric (flip _)) => class_apply flip_Asymmetric : typeclass_instances.
#[global]
Hint Extern 3 (Antisymmetric (flip _)) => class_apply flip_Antisymmetric : typeclass_instances.
#[global]
Hint Extern 3 (Transitive (flip _)) => class_apply flip_Transitive : typeclass_instances.
#[global]
Hint Extern 3 (StrictOrder (flip _)) => class_apply flip_StrictOrder : typeclass_instances.
#[global]
Hint Extern 3 (PreOrder (flip _)) => class_apply flip_PreOrder : typeclass_instances.

#[global]
Hint Extern 4 (subrelation (flip _) _) =>
  class_apply @subrelation_symmetric : typeclass_instances.

Arguments irreflexivity {A R Irreflexive} [x] _ : rename.
Arguments symmetry {A} {R} {_} [x] [y] _.
Arguments asymmetry {A} {R} {_} [x] [y] _ _.
Arguments transitivity {A} {R} {_} [x] [y] [z] _ _.
Arguments Antisymmetric A eqA {_} _.

#[global]
Hint Resolve irreflexivity : ord.

Unset Implicit Arguments.

Local Obligation Tactic := try solve [ simpl_relation ].

(** We now develop a generalization of results on relations for arbitrary predicates.
   The resulting theory can be applied to homogeneous binary relations but also to
   arbitrary n-ary predicates. *)

Local Open Scope list_scope.

(** A compact representation of non-dependent arities, with the codomain singled-out. *)

(* Note, we do not use [list (Type -> Type)] because it imposes unnecessary universe constraints *)
Inductive Tlist : Type := Tnil : Tlist | Tcons : type_function -> Tlist -> Tlist.
Local Infix "::" := Tcons.

Fixpoint arrows (l : Tlist) (r : type_function) : type_function :=
  match l with
    | Tnil => r
    | A :: l' => A -> arrows l' r
  end.
(*
(** We can define abbreviations for operation and relation types based on [arrows]. *)

Definition unary_operation A := arrows (A::Tnil) A.
Definition binary_operation A := arrows (A::A::Tnil) A.
Definition ternary_operation A := arrows (A::A::A::Tnil) A.

(** We define n-ary [predicate]s as functions into [Prop]. *)

Notation predicate l := (arrows l (fun _ => Prop)).

(** Unary predicates, or sets. *)

Definition unary_predicate A := predicate (A::Tnil).

(** Homogeneous binary relations, equivalent to [relation A]. *)

Definition binary_relation A := predicate (A::A::Tnil).

(** We can close a predicate by universal or existential quantification. *)
(*
Fixpoint predicate_all (l : Tlist) : forall T, predicate l T -> Prop :=
  match l with
    | Tnil => fun T f => f
    | A :: tl => fun f => forall x : A, predicate_all tl (f x)
  end.

Fixpoint predicate_exists (l : Tlist) : predicate l -> Prop :=
  match l with
    | Tnil => fun f => f
    | A :: tl => fun f => exists x : A, predicate_exists tl (f x)
  end.
*)
(** Pointwise extension of a binary operation on [T] to a binary operation
   on functions whose codomain is [T].
   For an operator on [Prop] this lifts the operator to a binary operation. *)

Fixpoint pointwise_extension {T : type_function} (op : forall X, binary_operation T X)
  (l : Tlist) : forall X, binary_operation (arrows l T) X :=
  match l return forall X, binary_operation (arrows l T) X with
    | Tnil => fun X R R' => op X R R'
    | A :: tl => fun X R R' =>
      fun x => pointwise_extension op tl (R x) (R' x)
  end.
*)
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

Declare Scope dependent_predicate_scope.

Infix "<∙>" := predicate_equivalence (at level 95, no associativity) : dependent_predicate_scope.
Infix "-∙>" := predicate_implication (at level 70, right associativity) : dependent_predicate_scope.

Local Open Scope dependent_predicate_scope.

(** The pointwise liftings of conjunction and disjunctions.
   Note that these are [binary_operation]s, building new relations out of old ones. *)

Definition predicate_intersection := pointwise_extension and.
Definition predicate_union := pointwise_extension or.

Infix "/∙\" := predicate_intersection (at level 80, right associativity) : dependent_predicate_scope.
Infix "\∙/" := predicate_union (at level 85, right associativity) : dependent_predicate_scope.
(*
(** The always [True] and always [False] predicates. *)

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

Notation "∙⊤∙" := true_predicate : dependent_predicate_scope.
Notation "∙⊥∙" := false_predicate : dependent_predicate_scope.
 *)
(** We define the various operations which define the algebra on binary relations,
   from the general ones. *)

Section Binary.
  Context {A : Type -> Type}.

  (*Definition relation_equivalence : relation (relation A) :=
    @predicate_equivalence (_::_::Tnil).
   *)

  Definition relation_conjunction (R : Dependent.relation A) (R' : Dependent.relation A) : Dependent.relation A
    := fun x y R0 a b => R x y R0 a b /\ R' x y R0 a b.
  Definition relation_disjunction (R : Dependent.relation A) (R' : Dependent.relation A) : Dependent.relation A
    := fun x y R0 a b => R x y R0 a b \/ R' x y R0 a b.

(*
  (** Relation equivalence is an equivalence, and subrelation defines a partial order. *)

  Global Instance relation_equivalence_equivalence :
    Equivalence relation_equivalence.
  Proof. exact (@predicate_equivalence_equivalence (A::A::Tnil)). Qed.

  Global Instance relation_implication_preorder : PreOrder (@subrelation A).
  Proof. exact (@predicate_implication_preorder (A::A::Tnil)). Qed.

  (** *** Partial Order.
   A partial order is a preorder which is additionally antisymmetric.
   We give an equivalent definition, up-to an equivalence relation
   on the carrier. *)

  Class PartialOrder eqA `{equ : Equivalence A eqA} R `{preo : PreOrder A R} :=
    partial_order_equivalence : relation_equivalence eqA (relation_conjunction R (flip R)).

  (** The equivalence proof is sufficient for proving that [R] must be a
   morphism for equivalence (see Morphisms).  It is also sufficient to
   show that [R] is antisymmetric w.r.t. [eqA] *)

  Global Instance partial_order_antisym `(PartialOrder eqA R) : Antisymmetric A eqA R.
  Proof with auto.
    reduce_goal.
    pose proof partial_order_equivalence as poe. do 3 red in poe.
    apply <- poe. firstorder.
  Qed.


  Lemma PartialOrder_inverse `(PartialOrder eqA R) : PartialOrder eqA (flip R).
  Proof. firstorder. Qed.
  *)
End Binary.
(*
#[global]
Hint Extern 3 (PartialOrder (flip _)) => class_apply PartialOrder_inverse : typeclass_instances.

(** The partial order defined by subrelation and relation equivalence. *)

#[global]
Program Instance subrelation_partial_order {A} :
  PartialOrder (@relation_equivalence A) subrelation.

Next Obligation.
Proof.
  unfold relation_equivalence in *. compute; firstorder.
Qed.

Global Typeclasses Opaque arrows predicate_implication predicate_equivalence
            relation_equivalence pointwise_lifting.
*)
