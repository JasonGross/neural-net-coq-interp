(* Adapted from Coq *)
From Coq Require Import SetoidList Relations Morphisms RelationPairs.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Hetero Relation_Definitions.Dependent.
From NeuralNetInterp.Util.Classes Require RelationClasses.Dependent Morphisms.Dependent.
From NeuralNetInterp.Util.Program Require Import Basics.Dependent.
Local Open Scope type_scope.

Local Notation Fst := (fun _ => @fst _ _).
Local Notation Snd := (fun _ => @snd _ _).

Generalizable Variables A B RA RB Ri Ro f.

(** Any function from [A] to [B] allow to obtain a relation over [A]
    out of a relation over [B]. *)

Definition RelCompFun {A B : type_function} (R:Dependent.relation B)(f:forall T, A T -> B T) : Dependent.relation A
  := fun _ _ R' a a' => R _ _ R' (f _ a) (f _ a').

(** Instances on RelCompFun must match syntactically *)
Global Typeclasses Opaque RelCompFun.

(** We define a product relation over [A*B]: each components should
    satisfy the corresponding initial relation. *)

Definition RelProd {A B} (RA:Dependent.relation A)(RB:Dependent.relation B) : Dependent.relation (fun T => A T * B T) :=
  Dependent.relation_conjunction
    (@RelCompFun (A * B) A RA Fst)
    (@RelCompFun (A * B) B RB Snd).

Global Typeclasses Opaque RelProd.

Module Export RelationPairsNotations.
  Export Morphisms.Dependent.ProperNotations.
  Infix "@@" := RelCompFun (at level 30, right associativity) : dependent_signature_scope.

  Notation "R @@1" := (R @@ Fst)%dependent_signature (at level 30) : dependent_signature_scope.
  Notation "R @@2" := (R @@ Snd)%dependent_signature (at level 30) : dependent_signature_scope.

  Infix "*" := RelProd : dependent_signature_scope.
End RelationPairsNotations.

Section RelProd_Instances.

  Context {A : type_function} {B : type_function} (RA : Dependent.relation A) (RB : Dependent.relation B).

  #[export] Instance RelProd_Reflexive `(Dependent.Reflexive _ RA, Dependent.Reflexive _ RB) : Dependent.Reflexive (RA*RB).
  Proof using Type. firstorder. Qed.

  #[export] Instance RelProd_Symmetric `(Dependent.Symmetric _ RA, Dependent.Symmetric _ RB)
  : Dependent.Symmetric (RA*RB).
  Proof using Type. cbv in *; firstorder. Qed.

  (*
  #[export] Instance RelProd_Transitive
           `(Dependent.Transitive _ RA, Dependent.Transitive _ RB) : Dependent.Transitive (RA*RB).
  Proof. cbv in *; firstorder. Qed.

  #[export] Program Instance RelProd_Equivalence
          `(Equivalence _ RA, Equivalence _ RB) : Equivalence (RA*RB).

  Lemma FstRel_ProdRel :
    relation_equivalence (RA @@1) (RA*(fun _ _ : B => True)).
  Proof. firstorder. Qed.

  Lemma SndRel_ProdRel :
    relation_equivalence (RB @@2) ((fun _ _ : A =>True) * RB).
  Proof. firstorder. Qed.

  #[export] Instance FstRel_sub :
    subrelation (RA*RB) (RA @@1).
  Proof. firstorder. Qed.

  #[export] Instance SndRel_sub :
    subrelation (RA*RB) (RB @@2).
  Proof. firstorder. Qed.

  #[export] Instance pair_compat :
    Proper (RA==>RB==> RA*RB) (@pair _ _).
  Proof. firstorder. Qed.

  #[export] Instance fst_compat :
    Proper (RA*RB ==> RA) Fst.
  Proof.
    intros (x,y) (x',y') (Hx,Hy); compute in *; auto.
  Qed.

  #[export] Instance snd_compat :
    Proper (RA*RB ==> RB) Snd.
  Proof.
    intros (x,y) (x',y') (Hx,Hy); compute in *; auto.
  Qed.

  #[export] Instance RelCompFun_compat (f:A->B)
           `(Proper _ (Ri==>Ri==>Ro) RB) :
    Proper (Ri@@f==>Ri@@f==>Ro) (RB@@f)%signature.
  Proof. unfold RelCompFun; firstorder. Qed.
*)
End RelProd_Instances.

#[export]
Hint Unfold RelProd RelCompFun : core.
#[export]
Hint Extern 2 (RelProd _ _ _ _) => split : core.
