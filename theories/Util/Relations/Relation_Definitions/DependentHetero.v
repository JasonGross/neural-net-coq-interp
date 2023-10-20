From Coq Require Import Relations.Relation_Definitions.
From NeuralNetInterp.Util.Program Require Import Basics.Dependent.
From NeuralNetInterp.Util.Relations Require Import Relation_Definitions.Hetero.
#[local] Set Implicit Arguments.
#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

Fixpoint relationn {n : nat} : type_functionn n -> type_functionn n -> Type
  := match n with
     | O => Hetero.relation
     | S n => fun F G => forall A B, Hetero.relation A B -> relationn (F A) (G B)
     end.
