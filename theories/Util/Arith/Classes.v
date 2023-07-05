From Coq Require Import Bool.
From NeuralNetInterp.Util Require Import Notations.
Set Implicit Arguments.

Class has_ltb A := ltb : A -> A -> bool.
Class has_eqb A := eqb : A -> A -> bool.
Class has_leb A := leb : A -> A -> bool.
Class has_add A := add : A -> A -> A.
Class has_sub A := sub : A -> A -> A.
Class has_mul A := mul : A -> A -> A.
Class has_div A := div : A -> A -> A.
Class has_opp A := opp : A -> A.
Class has_zero A := zero : A.
Class has_one A := one : A.
Class has_mod A := modulo : A -> A -> A.

Definition gtb {A} {ltb : has_ltb A} (x y : A) : bool := ltb y x.
Definition geb {A} {leb : has_leb A} (x y : A) : bool := leb y x.
Definition neqb {A} {eqb : has_eqb A} (x y : A) : bool := negb (eqb x y).

Infix "<?" := ltb : core_scope.
Infix "<=?" := leb : core_scope.
Infix "≤?" := leb : core_scope.
Infix ">?" := gtb : core_scope.
Infix ">=?" := geb : core_scope.
Infix "≥?" := geb : core_scope.
Infix "=?" := eqb : core_scope.
Infix "!=" := neqb : core_scope.
Infix "<>?" := neqb : core_scope.
Infix "≠?" := neqb : core_scope.
Infix "+" := add : core_scope.
Infix "-" := sub : core_scope.
Infix "*" := mul : core_scope.
Infix "/" := div : core_scope.
Notation "- x" := (opp x) : core_scope.
Infix "mod" := modulo : core_scope.
Notation "0" := zero : core_scope.
Notation "1" := one : core_scope.

#[export] Hint Mode has_ltb ! : typeclass_instances.
#[export] Hint Mode has_leb ! : typeclass_instances.
#[export] Hint Mode has_eqb ! : typeclass_instances.
#[export] Hint Mode has_add ! : typeclass_instances.
#[export] Hint Mode has_sub ! : typeclass_instances.
#[export] Hint Mode has_mul ! : typeclass_instances.
#[export] Hint Mode has_div ! : typeclass_instances.
#[export] Hint Mode has_opp ! : typeclass_instances.
#[export] Hint Mode has_zero ! : typeclass_instances.
#[export] Hint Mode has_one ! : typeclass_instances.
#[export] Hint Mode has_mod ! : typeclass_instances.

Local Open Scope core_scope.

#[export] Instance has_default_sub {A} {add : has_add A} {opp : has_opp A} : has_sub A | 10
  := fun x y => x + (-y).
#[export] Instance has_default_opp {A} {zero : has_zero A} {sub : has_sub A} : has_opp A | 10
  := fun x => 0 - x.
#[export] Instance has_default_eqb {A} {leb : has_leb A} : has_eqb A | 10
  := fun x y => (x <=? y) && (y <=? x).
#[export] Instance has_default_leb {A} {eqb : has_eqb A} {ltb : has_ltb A} : has_leb A | 10
  := fun x y => (x =? y) || (x <? y).
#[export] Instance has_default_ltb {A} {eqb : has_eqb A} {leb : has_leb A} : has_ltb A | 10
  := fun x y => negb (x =? y) && (x <=? y).

#[export] Hint Cut [ has_default_sub * has_default_sub * ].
#[export] Hint Cut [ has_default_opp * has_default_opp * ].
#[export] Hint Cut [ has_default_eqb * has_default_eqb * ].
#[export] Hint Cut [ has_default_leb * has_default_leb * ].
#[export] Hint Cut [ has_default_ltb * has_default_ltb * ].
