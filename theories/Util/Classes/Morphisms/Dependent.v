From Coq.Program Require Import Basics Tactics.
From Coq.Classes Require Import Morphisms RelationClasses.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Dependent.
(*From NeuralNetInterp.Util.Classes Require Export RelationClasses.Hetero.*)

Generalizable Variables A eqA B C D R RA RB RC m f x y.
Local Obligation Tactic := try solve [ simpl_relation ].

#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

Section Proper.
  Context {F : Type -> Type}.

  Class Proper (R : Dependent.relation F) (m : forall T, F T) : Prop :=
    proper_prf : forall {A B} (R0 : Hetero.relation A B), R A B R0 (m A) (m B).

  Definition respectful {A B : Type -> Type} (R : Dependent.relation A) (R' : Dependent.relation B) : Dependent.relation (fun T => A T -> B T)
    := fun a0 b0 R0 f g => forall x y, R _ _ R0 x y -> R' _ _ R0 (f x) (g y).

End Proper.

Section Proper2.
  Context {F : Type -> Type -> Type}.

  Class Proper2 (R : Dependent.relation2 F) (m : forall A B, F A B) : Prop :=
    proper2_prf : forall {A B} (R0 : Hetero.relation A B) {A' B'} (R1 : Hetero.relation A' B'), R A B R0 A' B' R1 (m A A') (m B B').

  Definition respectful2 {A B : Type -> Type -> Type} (R : Dependent.relation2 A) (R' : Dependent.relation2 B) : Dependent.relation2 (fun T U => A T U -> B T U)
    := fun a0 b0 R0 a1 b1 R1 f g => forall x y, R _ _ R0 _ _ R1 x y -> R' _ _ R0 _ _ R1 (f x) (g y).

End Proper2.

Section Proper3.
  Context {F : Type -> Type -> Type -> Type}.

  Class Proper3 (R : Dependent.relation3 F) (m : forall A B C, F A B C) : Prop :=
    proper3_prf : forall {A B} (R0 : Hetero.relation A B) {A' B'} (R1 : Hetero.relation A' B') {A'' B''} (R2 : Hetero.relation A'' B''), R A B R0 A' B' R1 A'' B'' R2 (m A A' A'') (m B B' B'').

  Definition respectful3 {A B : Type -> Type -> Type -> Type} (R : Dependent.relation3 A) (R' : Dependent.relation3 B) : Dependent.relation3 (fun T U V => A T U V -> B T U V)
    := fun a0 b0 R0 a1 b1 R1 a2 b2 R2 f g => forall x y, R _ _ R0 _ _ R1 _ _ R2 x y -> R' _ _ R0 _ _ R1 _ _ R2 (f x) (g y).

End Proper3.
(*
(** Non-dependent pointwise lifting *)
Definition pointwise_relation A {B} (R : relation B) : relation (A -> B) :=
  fun f g => forall a, R (f a) (g a).
 *)

(** Notations reminiscent of the old syntax for declaring morphisms. *)
Declare Scope dependent_signature_scope.
Delimit Scope dependent_signature_scope with dependent_signature.
Delimit Scope dependent_signature_scope with signatureD.

Declare Scope dependent2_signature_scope.
Delimit Scope dependent2_signature_scope with dependent2_signature.
Delimit Scope dependent2_signature_scope with signatureD2.

Declare Scope dependent3_signature_scope.
Delimit Scope dependent3_signature_scope with dependent3_signature.
Delimit Scope dependent3_signature_scope with signatureD3.

Module ProperNotations.

  Notation " R ++> R' " := (@respectful _ _ (R%dependent_signature) (R'%dependent_signature))
    (right associativity, at level 55) : dependent_signature_scope.

  Notation " R ==> R' " := (@respectful _ _ (R%dependent_signature) (R'%dependent_signature))
    (right associativity, at level 55) : dependent_signature_scope.

  Notation " R --> R' " := (@respectful _ _ (flip (R%dependent_signature)) (R'%dependent_signature))
    (right associativity, at level 55) : dependent_signature_scope.

  Notation " R ++> R' " := (@respectful2 _ _ (R%dependent2_signature) (R'%dependent2_signature))
    (right associativity, at level 55) : dependent2_signature_scope.

  Notation " R ==> R' " := (@respectful2 _ _ (R%dependent2_signature) (R'%dependent2_signature))
    (right associativity, at level 55) : dependent2_signature_scope.

  Notation " R --> R' " := (@respectful2 _ _ (flip (R%dependent2_signature)) (R'%dependent2_signature))
    (right associativity, at level 55) : dependent2_signature_scope.

  Notation " R ++> R' " := (@respectful3 _ _ (R%dependent3_signature) (R'%dependent3_signature))
    (right associativity, at level 55) : dependent3_signature_scope.

  Notation " R ==> R' " := (@respectful3 _ _ (R%dependent3_signature) (R'%dependent3_signature))
    (right associativity, at level 55) : dependent3_signature_scope.

  Notation " R --> R' " := (@respectful3 _ _ (flip (R%dependent3_signature)) (R'%dependent3_signature))
    (right associativity, at level 55) : dependent3_signature_scope.

End ProperNotations.

Notation idR := (fun (A B : Type) (R : Hetero.relation A B) => R).
Notation const R := (match _ as F return forall A B, Hetero.relation A B -> Hetero.relation F F with
                     | F => fun (A B : Type) (_ : Hetero.relation A B) => R
                     end) (only parsing).
Notation constR R := (fun (A B : Type) (_ : Hetero.relation A B) => R).
Arguments Proper {F}%type R%dependent_signature m.
Arguments respectful {A B}%type (R R')%dependent_signature (_ _)%type _ _.
Arguments Proper2 {F}%type R%dependent2_signature m.
Arguments respectful2 {A B}%type (R R')%dependent2_signature (_ _)%type _ _.
Arguments Proper3 {F}%type R%dependent3_signature m.
Arguments respectful3 {A B}%type (R R')%dependent3_signature (_ _)%type _ _.

Export ProperNotations.

Local Open Scope dependent_signature_scope.
(*
Section Relations.
  Context {A B : Type -> Type} (P : A -> U).

  (** [forall_def] reifies the dependent product as a definition. *)

  Definition forall_def : Type := forall x : A, P x.

  (** Dependent pointwise lifting of a relation on the range. *)

  Definition forall_relation
             (sig : forall a, relation (P a)) : relation (forall x, P x) :=
    fun f g => forall a, sig a (f a) (g a).

  Lemma pointwise_pointwise (R : relation B) :
    relation_equivalence (pointwise_relation A R) (@eq A ==> R).
  Proof. intros. split; reduce; subst; firstorder. Qed.

  (** Subrelations induce a morphism on the identity. *)

  Global Instance subrelation_id_proper `(subrelation A RA RA') : Proper (RA ==> RA') id.
  Proof. firstorder. Qed.

  (** The subrelation property goes through products as usual. *)

  Lemma subrelation_respectful `(subl : subrelation A RA' RA, subr : subrelation B RB RB') :
    subrelation (RA ==> RB) (RA' ==> RB').
  Proof. unfold subrelation in *; firstorder. Qed.

  (** And of course it is reflexive. *)

  Lemma subrelation_refl R : @subrelation A R R.
  Proof. unfold subrelation; firstorder. Qed.

  (** [Proper] is itself a covariant morphism for [subrelation].
   We use an unconvertible premise to avoid looping.
   *)

  Lemma subrelation_proper `(mor : Proper A R' m)
        `(unc : Unconvertible (relation A) R R')
        `(sub : subrelation A R' R) : Proper R m.
  Proof.
    intros. apply sub. apply mor.
  Qed.

  Global Instance proper_subrelation_proper :
    Proper (subrelation ++> eq ==> impl) (@Proper A).
  Proof. reduce. subst. firstorder. Qed.

  Global Instance pointwise_subrelation `(sub : subrelation B R R') :
    subrelation (pointwise_relation A R) (pointwise_relation A R') | 4.
  Proof. intros x y H a. unfold pointwise_relation in *. apply sub. apply H. Qed.

  (** For dependent function types. *)
  Lemma forall_subrelation (R S : forall x : A, relation (P x)) :
    (forall a, subrelation (R a) (S a)) -> subrelation (forall_relation R) (forall_relation S).
  Proof. intros H x y H0 a. apply H. apply H0. Qed.
End Relations.

Global Typeclasses Opaque respectful pointwise_relation forall_relation.
Arguments forall_relation {A P}%type sig%signature _ _.
Arguments pointwise_relation A%type {B}%type R%signature _ _.

#[global]
Hint Unfold Reflexive : core.
#[global]
Hint Unfold Symmetric : core.
#[global]
Hint Unfold Transitive : core.

(** Resolution with subrelation: favor decomposing products over applying reflexivity
  for unconstrained goals. *)
Ltac subrelation_tac T U :=
  (is_ground T ; is_ground U ; class_apply @subrelation_refl) ||
    class_apply @subrelation_respectful || class_apply @subrelation_refl.

#[global]
Hint Extern 3 (@subrelation _ ?T ?U) => subrelation_tac T U : typeclass_instances.

CoInductive apply_subrelation : Prop := do_subrelation.

Ltac proper_subrelation :=
  match goal with
    [ H : apply_subrelation |- _ ] => clear H ; class_apply @subrelation_proper
  end.

#[global]
Hint Extern 5 (@Proper _ ?H _) => proper_subrelation : typeclass_instances.

(** Essential subrelation instances for [iff], [impl] and [pointwise_relation]. *)

#[global]
Instance iff_impl_subrelation : subrelation iff impl | 2.
Proof. firstorder. Qed.

#[global]
Instance iff_flip_impl_subrelation : subrelation iff (flip impl) | 2.
Proof. firstorder. Qed.

(** We use an extern hint to help unification. *)

#[global]
Hint Extern 4 (subrelation (@forall_relation ?A ?B ?R) (@forall_relation _ _ ?S)) =>
  apply (@forall_subrelation A B R S) ; intro : typeclass_instances.

Section GenericInstances.
  (* Share universes *)
  Let U := Type.
  Context {A B C : U}.

  (** We can build a PER on the Coq function space if we have PERs on the domain and
   codomain. *)

  Program Instance respectful_per `(PER A R, PER B R') : PER (R ==> R').

  Next Obligation.
  Proof with auto.
    intros R H R' H0 x y z H1 H2 x0 y0 H3.
    assert(R x0 x0).
    - transitivity y0... symmetry...
    - transitivity (y x0)...
  Qed.

  (** The complement of a relation conserves its proper elements. *)

  Program Definition complement_proper
          `(mR : Proper (A -> A -> Prop) (RA ==> RA ==> iff) R) :
    Proper (RA ==> RA ==> iff) (complement R) := _.

  Next Obligation.
  Proof.
    intros RA R mR x y H x0 y0 H0.
    unfold complement.
    pose (mR x y H x0 y0 H0).
    intuition.
  Qed.

  (** The [flip] too, actually the [flip] instance is a bit more general. *)

  Program Definition flip_proper
          `(mor : Proper (A -> B -> C) (RA ==> RB ==> RC) f) :
    Proper (RB ==> RA ==> RC) (flip f) := _.

  Next Obligation.
  Proof.
    intros RA RB RC f mor x y H x0 y0 H0; apply mor ; auto.
  Qed.


  (** Every Transitive relation gives rise to a binary morphism on [impl],
   contravariant in the first argument, covariant in the second. *)

  Global Program
  Instance trans_contra_co_morphism
    `(Transitive A R) : Proper (R --> R ++> impl) R.

  Next Obligation.
  Proof with auto.
    intros R H x y H0 x0 y0 H1 H2.
    transitivity x...
    transitivity x0...
  Qed.

  (** Proper declarations for partial applications. *)

  Global Program
  Instance trans_contra_inv_impl_morphism
  `(Transitive A R) {x} : Proper (R --> flip impl) (R x) | 3.

  Next Obligation.
  Proof with auto.
    intros R H x x0 y H0 H1.
    transitivity y...
  Qed.

  Global Program
  Instance trans_co_impl_morphism
    `(Transitive A R) {x} : Proper (R ++> impl) (R x) | 3.

  Next Obligation.
  Proof with auto.
    intros R H x x0 y H0 H1.
    transitivity x0...
  Qed.

  Global Program
  Instance trans_sym_co_inv_impl_morphism
    `(PER A R) {x} : Proper (R ++> flip impl) (R x) | 3.

  Next Obligation.
  Proof with auto.
    intros R H x x0 y H0 H1.
    transitivity y... symmetry...
  Qed.

  Global Program Instance trans_sym_contra_impl_morphism
    `(PER A R) {x} : Proper (R --> impl) (R x) | 3.

  Next Obligation.
  Proof with auto.
    intros R H x x0 y H0 H1.
    transitivity x0... symmetry...
  Qed.

  Global Program Instance per_partial_app_morphism
  `(PER A R) {x} : Proper (R ==> iff) (R x) | 2.

  Next Obligation.
  Proof with auto.
    intros R H x x0 y H0.
    split.
    - intros ; transitivity x0...
    - intros.
      transitivity y...
      symmetry...
  Qed.

  (** Every Transitive relation induces a morphism by "pushing" an [R x y] on the left of an [R x z] proof to get an [R y z] goal. *)

  Global Program
  Instance trans_co_eq_inv_impl_morphism
  `(Transitive A R) : Proper (R ==> (@eq A) ==> flip impl) R | 2.

  Next Obligation.
  Proof with auto.
    intros R H x y H0 y0 y1 e H2; destruct e.
    transitivity y...
  Qed.

  (** Every Symmetric and Transitive relation gives rise to an equivariant morphism. *)

  Global Program
  Instance PER_morphism `(PER A R) : Proper (R ==> R ==> iff) R | 1.

  Next Obligation.
  Proof with auto.
    intros R H x y H0 x0 y0 H1.
    split ; intros.
    - transitivity x0... transitivity x... symmetry...

    - transitivity y... transitivity y0... symmetry...
  Qed.

  Lemma symmetric_equiv_flip `(Symmetric A R) : relation_equivalence R (flip R).
  Proof. firstorder. Qed.

  Global Program Instance compose_proper RA RB RC :
    Proper ((RB ==> RC) ==> (RA ==> RB) ==> (RA ==> RC)) (@compose A B C).

  Next Obligation.
  Proof.
    intros RA RB RC x y H x0 y0 H0 x1 y1 H1.
    unfold compose. apply H. apply H0. apply H1.
  Qed.

  Global Instance reflexive_eq_dom_reflexive `{Reflexive B R'}:
    Reflexive (respectful (@Logic.eq A) R').
  Proof. simpl_relation. Qed.

  (** [respectful] is a morphism for relation equivalence. *)

  Global Instance respectful_morphism :
    Proper (relation_equivalence ++> relation_equivalence ++> relation_equivalence)
           (@respectful A B).
  Proof.
    intros x y H x0 y0 H0 x1 x2.
    unfold respectful, relation_equivalence, predicate_equivalence in * ; simpl in *.
    split ; intros H1 x3 y1 H2.

    - rewrite <- H0.
      apply H1.
      rewrite H.
      assumption.

    - rewrite H0.
      apply H1.
      rewrite <- H.
      assumption.
  Qed.

  (** [R] is Reflexive, hence we can build the needed proof. *)

  Lemma Reflexive_partial_app_morphism `(Proper (A -> B) (R ==> R') m, ProperProxy A R x) :
    Proper R' (m x).
  Proof. simpl_relation. Qed.

  Lemma flip_respectful (R : relation A) (R' : relation B) :
    relation_equivalence (flip (R ==> R')) (flip R ==> flip R').
  Proof.
    intros.
    unfold flip, respectful.
    split ; intros ; intuition.
  Qed.


  (** Treating flip: can't make them direct instances as we
   need at least a [flip] present in the goal. *)

  Lemma flip1 `(subrelation A R' R) : subrelation (flip (flip R')) R.
  Proof. firstorder. Qed.

  Lemma flip2 `(subrelation A R R') : subrelation R (flip (flip R')).
  Proof. firstorder. Qed.

  (** That's if and only if *)

  Lemma eq_subrelation `(Reflexive A R) : subrelation (@eq A) R.
  Proof. simpl_relation. Qed.

  (** Once we have normalized, we will apply this instance to simplify the problem. *)

  Definition proper_flip_proper `(mor : Proper A R m) : Proper (flip R) m := mor.

  Lemma proper_eq (x : A) : Proper (@eq A) x.
  Proof. intros. reflexivity. Qed.

End GenericInstances.

Class PartialApplication.

CoInductive normalization_done : Prop := did_normalization.

Class Params {A : Type} (of : A) (arity : nat).
#[global] Instance eq_pars : Params (@eq) 1 := {}.
#[global] Instance iff_pars : Params (@iff) 0 := {}.
#[global] Instance impl_pars : Params (@impl) 0 := {}.
#[global] Instance flip_pars : Params (@flip) 4 := {}.

Ltac partial_application_tactic :=
  let rec do_partial_apps H m cont :=
    match m with
      | ?m' ?x => class_apply @Reflexive_partial_app_morphism ;
        [(do_partial_apps H m' ltac:(idtac))|clear H]
      | _ => cont
    end
  in
  let rec do_partial H ar m :=
    lazymatch ar with
      | 0%nat => do_partial_apps H m ltac:(fail 1)
      | S ?n' =>
        match m with
          ?m' ?x => do_partial H n' m'
        end
    end
  in
  let params m sk fk :=
    (let m' := fresh in head_of_constr m' m ;
     let n := fresh in evar (n:nat) ;
     let v := eval compute in n in clear n ;
      let H := fresh in
        assert(H:Params m' v) by (subst m'; once typeclasses eauto) ;
          let v' := eval compute in v in subst m';
            (sk H v' || fail 1))
    || fk
  in
  let on_morphism m cont :=
    params m ltac:(fun H n => do_partial H n m)
      ltac:(cont)
  in
  match goal with
    | [ _ : normalization_done |- _ ] => fail 1
    | [ _ : @Params _ _ _ |- _ ] => fail 1
    | [ |- @Proper ?T _ (?m ?x) ] =>
      match goal with
        | [ H : PartialApplication |- _ ] =>
          class_apply @Reflexive_partial_app_morphism; [|clear H]
        | _ => on_morphism (m x)
          ltac:(class_apply @Reflexive_partial_app_morphism)
      end
  end.

(** Bootstrap !!! *)

#[global]
Instance proper_proper {A} : Proper (relation_equivalence ==> eq ==> iff) (@Proper A).
Proof.
  intros x y H y0 y1 e; destruct e.
  reduce in H.
  split ; red ; intros H0.
  - setoid_rewrite <- H.
    apply H0.
  - setoid_rewrite H.
    apply H0.
Qed.

Ltac proper_reflexive :=
  match goal with
    | [ _ : normalization_done |- _ ] => fail 1
    | _ => class_apply proper_eq || class_apply @reflexive_proper
  end.

#[global]
Hint Extern 1 (subrelation (flip _) _) => class_apply @flip1 : typeclass_instances.
#[global]
Hint Extern 1 (subrelation _ (flip _)) => class_apply @flip2 : typeclass_instances.

#[global]
Hint Extern 1 (Proper _ (complement _)) => apply @complement_proper
  : typeclass_instances.
#[global]
Hint Extern 1 (Proper _ (flip _)) => apply @flip_proper
  : typeclass_instances.
#[global]
Hint Extern 2 (@Proper _ (flip _) _) => class_apply @proper_flip_proper
  : typeclass_instances.
#[global]
Hint Extern 4 (@Proper _ _ _) => partial_application_tactic
  : typeclass_instances.
#[global]
Hint Extern 7 (@Proper _ _ _) => proper_reflexive
  : typeclass_instances.

(** Special-purpose class to do normalization of signatures w.r.t. flip. *)

Section Normalize.
  Context (A : Type).

  Class Normalizes (m : relation A) (m' : relation A) : Prop :=
    normalizes : relation_equivalence m m'.

  (** Current strategy: add [flip] everywhere and reduce using [subrelation]
   afterwards. *)

  Lemma proper_normalizes_proper `(Normalizes R0 R1, Proper A R1 m) : Proper R0 m.
  Proof.
    rewrite normalizes.
    assumption.
  Qed.

  Lemma flip_atom R : Normalizes R (flip (flip R)).
  Proof.
    firstorder.
  Qed.

End Normalize.

Lemma flip_arrow {A : Type} {B : Type}
      `(NA : Normalizes A R (flip R'''), NB : Normalizes B R' (flip R'')) :
  Normalizes (A -> B) (R ==> R') (flip (R''' ==> R'')%signature).
Proof.
  unfold Normalizes in *.
  unfold relation_equivalence in *.
  unfold predicate_equivalence in *. simpl in *.
  unfold respectful. unfold flip in *.
  intros x x0; split; intros H x1 y H0.
  - apply NB. apply H. apply NA. apply H0.
  - apply NB. apply H. apply NA. apply H0.
Qed.

Ltac normalizes :=
  match goal with
    | [ |- Normalizes _ (respectful _ _) _ ] => class_apply @flip_arrow
    | _ => class_apply @flip_atom
  end.

Ltac proper_normalization :=
  match goal with
    | [ _ : normalization_done |- _ ] => fail 1
    | [ _ : apply_subrelation |- @Proper _ ?R _ ] =>
      let H := fresh "H" in
      set(H:=did_normalization) ; class_apply @proper_normalizes_proper
  end.

#[global]
Hint Extern 1 (Normalizes _ _ _) => normalizes : typeclass_instances.
#[global]
Hint Extern 6 (@Proper _ _ _) => proper_normalization
  : typeclass_instances.

(** When the relation on the domain is symmetric, we can
    flip the relation on the codomain. Same for binary functions. *)

Lemma proper_sym_flip :
 forall `(Symmetric A R1)`(Proper (A->B) (R1==>R2) f),
 Proper (R1==>flip R2) f.
Proof.
intros A R1 Sym B R2 f Hf.
intros x x' Hxx'. apply Hf, Sym, Hxx'.
Qed.

Lemma proper_sym_flip_2 :
 forall `(Symmetric A R1)`(Symmetric B R2)`(Proper (A->B->C) (R1==>R2==>R3) f),
 Proper (R1==>R2==>flip R3) f.
Proof.
intros A R1 Sym1 B R2 Sym2 C R3 f Hf.
intros x x' Hxx' y y' Hyy'. apply Hf; auto.
Qed.

(** When the relation on the domain is symmetric, a predicate is
  compatible with [iff] as soon as it is compatible with [impl].
  Same with a binary relation. *)

Lemma proper_sym_impl_iff : forall `(Symmetric A R)`(Proper _ (R==>impl) f),
 Proper (R==>iff) f.
Proof.
intros A R Sym f Hf x x' Hxx'. repeat red in Hf. split; eauto.
Qed.

Lemma proper_sym_impl_iff_2 :
 forall `(Symmetric A R)`(Symmetric B R')`(Proper _ (R==>R'==>impl) f),
 Proper (R==>R'==>iff) f.
Proof.
intros A R Sym B R' Sym' f Hf x x' Hxx' y y' Hyy'.
repeat red in Hf. split; eauto.
Qed.

(** A [PartialOrder] is compatible with its underlying equivalence. *)

#[global]
Instance PartialOrder_proper `(PartialOrder A eqA R) :
  Proper (eqA==>eqA==>iff) R.
Proof.
intros.
apply proper_sym_impl_iff_2. 1-2: auto with relations.
intros x x' Hx y y' Hy Hr.
transitivity x.
- generalize (partial_order_equivalence x x'); compute; intuition.
- transitivity y; auto.
  generalize (partial_order_equivalence y y'); compute; intuition.
Qed.

(** From a [PartialOrder] to the corresponding [StrictOrder]:
     [lt = le /\ ~eq].
    If the order is total, we could also say [gt = ~le]. *)

Lemma PartialOrder_StrictOrder `(PartialOrder A eqA R) :
  StrictOrder (relation_conjunction R (complement eqA)).
Proof.
split; compute.
- intros x (_,Hx). apply Hx, Equivalence_Reflexive.
- intros x y z (Hxy,Hxy') (Hyz,Hyz'). split.
  + apply PreOrder_Transitive with y; assumption.
  + intro Hxz.
    apply Hxy'.
    apply partial_order_antisym; auto.
    rewrite Hxz; auto.
Qed.


(** From a [StrictOrder] to the corresponding [PartialOrder]:
     [le = lt \/ eq].
    If the order is total, we could also say [ge = ~lt]. *)

Lemma StrictOrder_PreOrder
 `(Equivalence A eqA, StrictOrder A R, Proper _ (eqA==>eqA==>iff) R) :
 PreOrder (relation_disjunction R eqA).
Proof.
split.
- intros x. right. reflexivity.
- intros x y z [Hxy|Hxy] [Hyz|Hyz].
  + left. transitivity y; auto.
  + left. rewrite <- Hyz; auto.
  + left. rewrite Hxy; auto.
  + right. transitivity y; auto.
Qed.

#[global]
Hint Extern 4 (PreOrder (relation_disjunction _ _)) =>
  class_apply StrictOrder_PreOrder : typeclass_instances.

Lemma StrictOrder_PartialOrder
  `(Equivalence A eqA, StrictOrder A R, Proper _ (eqA==>eqA==>iff) R) :
  PartialOrder eqA (relation_disjunction R eqA).
Proof.
intros. intros x y. compute. intuition auto with relations.
elim (StrictOrder_Irreflexive x).
transitivity y; auto.
Qed.

#[global]
Hint Extern 4 (StrictOrder (relation_conjunction _ _)) =>
  class_apply PartialOrder_StrictOrder : typeclass_instances.

#[global]
Hint Extern 4 (PartialOrder _ (relation_disjunction _ _)) =>
  class_apply StrictOrder_PartialOrder : typeclass_instances.
*)
