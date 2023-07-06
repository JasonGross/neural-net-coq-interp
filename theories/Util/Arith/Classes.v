From Coq Require Import Bool.
From NeuralNetInterp.Util Require Import Notations.
Set Implicit Arguments.

Cumulative Polymorphic Inductive TyList := tynil | tycons (_ : Type) (_ : TyList).

Class has_ltb A := ltb : A -> A -> bool.
Class has_eqb A := eqb : A -> A -> bool.
Class has_leb A := leb : A -> A -> bool.
Class has_add A := add : A -> A -> A.
Class has_sub A := sub : A -> A -> A.
Class has_mul A := mul : A -> A -> A.
Class has_opp A := opp : A -> A.
Class has_zero A := zero : A.
Class has_one A := one : A.
Class has_mod A := modulo : A -> A -> A.
Class has_max A := max : A -> A -> A.
Class has_min A := min : A -> A -> A.
Class has_int_div_by A B := int_div : A -> B -> A.
Class has_abs A := abs : A -> A.
Notation has_int_div A := (has_int_div_by A A).
Class has_div_by A B := div : A -> B -> A.
Notation has_div A := (has_div_by A A).
Class has_coer A B := coer : A -> B.
Definition has_coer_from (avoid : TyList) := has_coer.
Definition has_coer_to (avoid : TyList) := has_coer.
Definition has_coer_can_trans := has_coer.
Existing Class has_coer_from.
Existing Class has_coer_to.
Existing Class has_coer_can_trans.
#[export] Typeclasses Opaque has_coer_can_trans has_coer_from has_coer_to.

Definition gtb {A} {ltb : has_ltb A} (x y : A) : bool := ltb y x.
Definition geb {A} {leb : has_leb A} (x y : A) : bool := leb y x.
Definition neqb {A} {eqb : has_eqb A} (x y : A) : bool := negb (eqb x y).
Definition sqr {A} {mul : has_mul A} (x : A) : A := mul x x.

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
Infix "//" := int_div : core_scope.
Notation "x ²" := (sqr x) : core_scope.
Notation "- x" := (opp x) : core_scope.
Infix "mod" := modulo : core_scope.
Notation "0" := zero : core_scope.
Notation "1" := one : core_scope.

#[export] Hint Mode has_ltb ! : typeclass_instances.
#[export] Hint Mode has_leb ! : typeclass_instances.
#[export] Hint Mode has_eqb ! : typeclass_instances.
#[export] Hint Mode has_max ! : typeclass_instances.
#[export] Hint Mode has_min ! : typeclass_instances.
#[export] Hint Mode has_add ! : typeclass_instances.
#[export] Hint Mode has_sub ! : typeclass_instances.
#[export] Hint Mode has_mul ! : typeclass_instances.
#[export] Hint Mode has_int_div_by ! - : typeclass_instances.
#[export] Hint Mode has_div_by ! - : typeclass_instances.
#[export] Hint Mode has_opp ! : typeclass_instances.
#[export] Hint Mode has_zero ! : typeclass_instances.
#[export] Hint Mode has_one ! : typeclass_instances.
#[export] Hint Mode has_mod ! : typeclass_instances.
#[export] Hint Mode has_abs ! : typeclass_instances.
#[export] Hint Mode has_coer ! ! : typeclass_instances.
#[export] Hint Mode has_coer_can_trans ! ! : typeclass_instances.
#[export] Hint Mode has_coer_from + ! - : typeclass_instances.
#[export] Hint Mode has_coer_to + - ! : typeclass_instances.
#[export] Hint Unfold has_coer_from has_coer_to : typeclass_instances.

Ltac check_tylist_free ls B :=
  lazymatch ls with
  | context[tycons B _] => fail
  | _ => idtac
  end.
Ltac check_unify_has_coer_from B
  := lazymatch goal with
     | [ |- has_coer_from ?avoid _ ?b ]
       => is_evar b; check_tylist_free avoid B; unify b B
     end.
Ltac check_unify_has_coer_to A
  := lazymatch goal with
     | [ |- has_coer_to ?avoid ?a _ ]
       => is_evar a; check_tylist_free avoid A; unify a A
     end.

#[export] Instance coer_trans {A B C} : has_coer_from (tycons C tynil) A B -> has_coer_to (tycons A tynil) B C -> has_coer_can_trans A C | 10
  := fun ab bc a => bc (ab a).
#[export] Instance coer_refl {A} : has_coer A A := fun x => x.
#[export] Hint Cut [ ( _ * ) coer_trans ( _ * ) coer_refl ( _ * ) ] : typeclass_instances.
#[export] Hint Extern 10 (has_coer ?A ?B)
=> tryif first [ is_evar A | is_evar B | unify A B ] then fail else change (has_coer_can_trans A B) : typeclass_instances.

Local Open Scope core_scope.

#[export] Instance has_default_sub {A} {add : has_add A} {opp : has_opp A} : has_sub A | 10
  := fun x y => x + (-y).
#[export] Instance has_default_opp {A} {zero : has_zero A} {sub : has_sub A} : has_opp A | 10
  := fun x => 0 - x.
#[export] Instance has_default_eqb {A} {leb : has_leb A} : has_eqb A | 10
  := fun x y => (x ≤? y) && (y ≤? x).
#[export] Instance has_default_leb {A} {eqb : has_eqb A} {ltb : has_ltb A} : has_leb A | 10
  := fun x y => (x =? y) || (x <? y).
#[export] Instance has_default_ltb {A} {eqb : has_eqb A} {leb : has_leb A} : has_ltb A | 10
  := fun x y => (x ≠? y) && (x ≤? y).
#[export] Instance has_default_max_ltb {A} {ltb : has_ltb A} : has_max A | 10
  := fun x y => if x <? y then y else x.
#[export] Instance has_default_min_ltb {A} {ltb : has_ltb A} : has_min A | 10
  := fun x y => if x <? y then x else y.
#[export] Instance has_default_max_leb {A} {leb : has_leb A} : has_max A | 10
  := fun x y => if x ≤? y then y else x.
#[export] Instance has_default_min_leb {A} {leb : has_leb A} : has_min A | 10
  := fun x y => if x ≤? y then x else y.
#[export] Instance has_default_abs {A} {opp : has_opp A} {max : has_max A} : has_abs A
  := fun x => max x (opp x).

#[export] Hint Cut [ ( _ * ) has_default_sub ( _ * ) has_default_sub ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_opp ( _ * ) has_default_opp ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_eqb ( _ * ) has_default_eqb ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_leb ( _ * ) has_default_leb ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_ltb ( _ * ) has_default_ltb ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_max_ltb has_default_ltb ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_min_ltb has_default_ltb ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_max_leb has_default_leb ( _ * ) ] : typeclass_instances.
#[export] Hint Cut [ ( _ * ) has_default_min_leb has_default_leb ( _ * ) ] : typeclass_instances.
