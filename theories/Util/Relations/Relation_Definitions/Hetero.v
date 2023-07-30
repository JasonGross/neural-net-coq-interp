From Coq Require Import Relations.Relation_Definitions.
#[local] Set Implicit Arguments.

Definition relation A B := A -> B -> Prop.

Section Relations_of_Relations.
  Context {A B : Type}.

  Definition inclusion (R1 R2:relation A B) : Prop :=
    forall x y, R1 x y -> R2 x y.

  Definition same_relation (R1 R2:relation A B) : Prop :=
    inclusion R1 R2 /\ inclusion R2 R1.

End Relations_of_Relations.

#[export]
Hint Unfold inclusion same_relation commut: sets.
