From Coq Require Import Morphisms Setoid.
From NeuralNetInterp.Util.Arith Require Import Classes.
#[local] Set Implicit Arguments.

Class Associative {A} (R : relation A) (op : A -> A -> A)
  := assoc : forall x y z, R (op x (op y z)) (op (op x y) z).
Class Commutative {A} (R : relation A) (op : A -> A -> A)
  := comm : forall x y, R (op x y) (op y x).
Class LeftId {A} (R : relation A) (op : A -> A -> A) (id : A)
  := id_l : forall x, R (op id x) x.
Class RightId {A} (R : relation A) (op : A -> A -> A) (id : A)
  := id_r : forall x, R (op x id) x.
Class LeftZero {A} (R : relation A) (op : A -> A -> A) (zero : A)
  := zero_l : forall x, R (op zero x) zero.
Class RightZero {A} (R : relation A) (op : A -> A -> A) (zero : A)
  := zero_r : forall x, R (op x zero) zero.
Class Distributive12 {A} (R : relation A) (op_outer : A -> A) (op_inner : A -> A -> A)
  := distr_1 : forall x y, R (op_outer (op_inner x y)) (op_inner (op_outer x) (op_outer y)).
Class LeftDistributive {A} (R : relation A) (op_outer : A -> A -> A) (op_inner : A -> A -> A)
  := distr_l : forall x y z, R (op_outer x (op_inner y z)) (op_inner (op_outer x y) (op_outer x z)).
Class RightDistributive {A} (R : relation A) (op_outer : A -> A -> A) (op_inner : A -> A -> A)
  := distr_r : forall x y z, R (op_outer (op_inner x y) z) (op_inner (op_outer x z) (op_outer y z)).
