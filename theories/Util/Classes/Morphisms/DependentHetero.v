From Coq.Program Require Import Basics Tactics.
From Coq.Classes Require Import Morphisms RelationClasses.
From NeuralNetInterp.Util.Program Require Import Basics.Dependent.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Dependent Relation_Definitions.DependentHetero.
Export Relation_Definitions.Dependent.RelationsNotations.
(*From NeuralNetInterp.Util.Classes Require Export RelationClasses.Hetero.*)

Generalizable Variables A eqA B C D R RA RB RC m f x y.
Local Obligation Tactic := try solve [ simpl_relation ].

#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

Section Propern.
  Fixpoint Propern {n : nat} : forall {F G : type_functionn n} (R : DependentHetero.relationn F G) (mF : Dependent.foralln F) (mG : Dependent.foralln G), Prop
    := match n with
       | O => fun F G R mF mG => R mF mG
       | S n => fun F G R mF mG => forall A B (R0 : Hetero.relation A B),
                    @Propern n (F A) (G B) (R A B R0) (mF A) (mG B)
       end.
  Existing Class Propern.

  Fixpoint respectfuln {n} : forall {A A' B B' : type_functionn n} (R : DependentHetero.relationn A A') (R' : DependentHetero.relationn B B'), DependentHetero.relationn (Dependent.arrown A B) (Dependent.arrown A' B')
    := match n with
       | O => fun A B A' B' R R' f g => forall x y, R x y -> R' (f x) (g y)
       | S n => fun A B A' B' R R' a b r => @respectfuln n _ _ _ _ (R a b r) (R' a b r)
       end.

(*
  Definition respectful_hetero {A : Type -> Type} {B : forall x, A x -> Type}
    (R : Dependent.relation A)
    (R' : forall x y, Hetero.relation x y -> forall x' y', Hetero.relation (B x x') (B y y'))
    : Dependent.relation (fun T => forall a : A T, B T a)
    := fun a0 b0 R0 f g => forall x y, R _ _ R0 x y -> R' _ _ R0 _ _ (f x) (g y).

  Definition respectful {A B : Type -> Type} (R : Dependent.relation A) (R' : Dependent.relation B) : Dependent.relation (fun T => A T -> B T)
    := Eval cbv [respectful_hetero] in
      @respectful_hetero A (fun T _ => B T) R (fun x y R0 _ _ => R' x y R0).


    hnf in R.
R mF mG

 *)
End Propern.
