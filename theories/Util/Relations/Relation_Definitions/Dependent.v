From Coq Require Import Relations.Relation_Definitions.
From NeuralNetInterp.Util.Program Require Import Basics.Dependent.
From NeuralNetInterp.Util.Relations Require Import Relation_Definitions.Hetero.
#[local] Set Implicit Arguments.
#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

#[export] Set Warnings Append "-overwriting-delimiting-key".
Declare Scope dependent_signature_scope.
Declare Scope dependent2_signature_scope.
Declare Scope dependent3_signature_scope.
Declare Scope dependent4_signature_scope.
#[export] Set Warnings Append "overwriting-delimiting-key".

Definition relation (F : type_function)
  := forall A B, Hetero.relation A B -> Hetero.relation (F A) (F B).
#[global] Arguments relation F%type_function.
Definition relation2 (F : type_function2)
  := forall A B, Hetero.relation A B -> forall A' B', Hetero.relation A' B' -> Hetero.relation (F A A') (F B B').
#[global] Arguments relation2 F%type_function2.
Definition relation3 (F : type_function3)
  := forall A B, Hetero.relation A B -> forall A' B', Hetero.relation A' B' -> forall A'' B'', Hetero.relation A'' B'' -> Hetero.relation (F A A' A'') (F B B' B'').
#[global] Arguments relation3 F%type_function3.
Definition relation4 (F : type_function4)
  := forall A B, Hetero.relation A B -> forall A' B', Hetero.relation A' B' -> forall A'' B'', Hetero.relation A'' B'' -> forall A''' B''', Hetero.relation A''' B''' -> Hetero.relation (F A A' A'' A''') (F B B' B'' B''').
#[global] Arguments relation4 F%type_function4.

Definition lift_comp {F} (R : relation F) {G} (R' : relation G) : relation (F ∘ G)
  := fun A B RAB => R _ _ (R' _ _ RAB).

Module Export RelationsNotations.

  #[export] Set Warnings Append "-overwriting-delimiting-key".
  Delimit Scope dependent_signature_scope with dependent_signature.
  Delimit Scope dependent_signature_scope with signatureD.
  Bind Scope dependent_signature_scope with relation.

  Delimit Scope dependent2_signature_scope with dependent2_signature.
  Delimit Scope dependent2_signature_scope with signatureD2.
  Bind Scope dependent2_signature_scope with relation2.

  Delimit Scope dependent3_signature_scope with dependent3_signature.
  Delimit Scope dependent3_signature_scope with signatureD3.
  Bind Scope dependent3_signature_scope with relation3.

  Delimit Scope dependent4_signature_scope with dependent4_signature.
  Delimit Scope dependent4_signature_scope with signatureD4.
  Bind Scope dependent4_signature_scope with relation4.
  #[export] Set Warnings Append "overwriting-delimiting-key".

  Notation "R ∘ R'" := (fun (A B : Type) (RAB : Hetero.relation A B) => R%dependent_signature _ _ (R'%dependent_signature A B RAB)) : dependent_signature_scope.
End RelationsNotations.

Definition lift2_1 {F} (R : relation F) : relation2 (fun A _ => F A)
  := fun _ _ R' _ _ _ => R _ _ R'.
Definition lift2_2 {F} (R : relation F) : relation2 (fun _ A => F A)
  := fun _ _ _ _ _ R' => R _ _ R'.
Definition lift3_1 {F} (R : relation F) : relation3 (fun A _ _ => F A)
  := fun _ _ R' _ _ _ _ _ _ => R _ _ R'.
Definition lift3_2 {F} (R : relation F) : relation3 (fun _ A _ => F A)
  := fun _ _ _ _ _ R' _ _ _ => R _ _ R'.
Definition lift3_3 {F} (R : relation F) : relation3 (fun _ _ A => F A)
  := fun _ _ _ _ _ _ _ _ R' => R _ _ R'.
Definition lift3_12 {F} (R : relation2 F) : relation3 (fun A B _ => F A B)
  := fun _ _ R' _ _ R'' _ _ _ => R _ _ R' _ _ R''.
Definition lift3_13 {F} (R : relation2 F) : relation3 (fun A _ B => F A B)
  := fun _ _ R' _ _ _ _ _ R'' => R _ _ R' _ _ R''.
Definition lift3_23 {F} (R : relation2 F) : relation3 (fun _ A B => F A B)
  := fun _ _ _ _ _ R' _ _ R'' => R _ _ R' _ _ R''.
Definition lift4_1 {F} (R : relation F) : relation4 (fun A _ _ _ => F A)
  := fun _ _ R' _ _ _ _ _ _ _ _ _ => R _ _ R'.
Definition lift4_2 {F} (R : relation F) : relation4 (fun _ A _ _ => F A)
  := fun _ _ _ _ _ R' _ _ _ _ _ _ => R _ _ R'.
Definition lift4_3 {F} (R : relation F) : relation4 (fun _ _ A _ => F A)
  := fun _ _ _ _ _ _ _ _ R' _ _ _ => R _ _ R'.
Definition lift4_4 {F} (R : relation F) : relation4 (fun _ _ _ A => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ R' => R _ _ R'.
Definition lift4_12 {F} (R : relation2 F) : relation4 (fun A B _ _ => F A B)
  := fun _ _ R' _ _ R'' _ _ _ _ _ _ => R _ _ R' _ _ R''.
Definition lift4_13 {F} (R : relation2 F) : relation4 (fun A _ B _ => F A B)
  := fun _ _ R' _ _ _ _ _ R'' _ _ _ => R _ _ R' _ _ R''.
Definition lift4_14 {F} (R : relation2 F) : relation4 (fun A _ _ B => F A B)
  := fun _ _ R' _ _ _ _ _ _ _ _ R'' => R _ _ R' _ _ R''.
Definition lift4_23 {F} (R : relation2 F) : relation4 (fun _ A B _ => F A B)
  := fun _ _ _ _ _ R' _ _ R'' _ _ _ => R _ _ R' _ _ R''.
Definition lift4_24 {F} (R : relation2 F) : relation4 (fun _ A _ B => F A B)
  := fun _ _ _ _ _ R' _ _ _ _ _ R'' => R _ _ R' _ _ R''.
Definition lift4_34 {F} (R : relation2 F) : relation4 (fun _ _ A B => F A B)
  := fun _ _ _ _ _ _ _ _ R' _ _ R'' => R _ _ R' _ _ R''.
Definition lift4_123 {F} (R : relation3 F) : relation4 (fun A B C _ => F A B C)
  := fun _ _ R' _ _ R'' _ _ R''' _ _ _ => R _ _ R' _ _ R'' _ _ R'''.
Definition lift4_124 {F} (R : relation3 F) : relation4 (fun A B _ C => F A B C)
  := fun _ _ R' _ _ R'' _ _ _ _ _ R''' => R _ _ R' _ _ R'' _ _ R'''.
Definition lift4_134 {F} (R : relation3 F) : relation4 (fun A _ B C => F A B C)
  := fun _ _ R' _ _ _ _ _ R'' _ _ R''' => R _ _ R' _ _ R'' _ _ R'''.
Definition lift4_234 {F} (R : relation3 F) : relation4 (fun _ A B C => F A B C)
  := fun _ _ _ _ _ R' _ _ R'' _ _ R''' => R _ _ R' _ _ R'' _ _ R'''.

Section Relation_Definition.

  Variable F : type_function.


  (*
  Variable R : relation.

  Section General_Properties_of_Relations.

    Definition reflexive : Prop := forall A RA, @reflexive A RA -> forall x, R RA x x.
    Definition transitive : Prop := forall A RA, @transitive A RA -> forall x, R RA x x.
    Definition transitive : Prop := forall (Ri:Relation_Definitions.relation I) i j k (x:A i) (y:A j) (z:A k), (Ri i j -> Ri j k -> Ri i k) -> R Ri x y -> R Ri y z -> R Ri x z.
    Definition symmetric : Prop := forall (Ri:Relation_Definitions.relation I) i j (x:A i) (y:A j), (Ri i j -> Ri j i) -> R Ri x y -> R Ri y x.
    Definition antisymmetric : Prop := forall (Ri:Relation_Definitions.relation I) i (x y:A i), R Ri x y -> R Ri y x -> x = y.

    (* for compatibility with Equivalence in  ../PROGRAMS/ALG/  *)
    Definition equiv := reflexive /\ transitive /\ symmetric.

  End General_Properties_of_Relations.



  Section Sets_of_Relations.

    Record preorder : Prop :=
      { preord_refl : reflexive; preord_trans : transitive}.

    Record order : Prop :=
      { ord_refl : reflexive;
	ord_trans : transitive;
	ord_antisym : antisymmetric}.

    Record equivalence : Prop :=
      { equiv_refl : reflexive;
	equiv_trans : transitive;
	equiv_sym : symmetric}.

    Record PER : Prop :=  {per_sym : symmetric; per_trans : transitive}.

  End Sets_of_Relations.


  Section Relations_of_Relations.

    Definition inclusion (R1 R2:relation) : Prop :=
      forall Ri i j x y, R1 Ri i j x y -> R2 Ri i j x y.

    Definition same_relation (R1 R2:relation) : Prop :=
      inclusion R1 R2 /\ inclusion R2 R1.
(*
    Definition commut (R1 R2:relation) : Prop :=
      forall i j (x:A i) (y:A j),
	R1 _ _ y x -> forall k (z:A k), R2 _ _ z y ->  exists2 y' : A, R2 y' x & R1 z y'.
*)
  End Relations_of_Relations.

*)
End Relation_Definition.
(*
#[export]
Hint Unfold reflexive transitive antisymmetric symmetric: sets.

#[export]
Hint Resolve Build_preorder Build_order Build_equivalence Build_PER
  preord_refl preord_trans ord_refl ord_trans ord_antisym equiv_refl
  equiv_trans equiv_sym per_sym per_trans: sets.

#[export]
Hint Unfold inclusion same_relation commut: sets.
*)
