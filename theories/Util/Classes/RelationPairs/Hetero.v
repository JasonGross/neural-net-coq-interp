(* Adapted from Coq *)
From Coq Require Import SetoidList Relations Morphisms RelationPairs.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Hetero.
From NeuralNetInterp.Util.Classes Require RelationClasses.Hetero.
(*From NeuralNetInterp.Util.Classes Require Morphisms.Hetero.*)

Local Notation Fst := (@fst _ _).
Local Notation Snd := (@snd _ _).

Generalizable Variables A B RA RB Ri Ro f.

(** Any function from [A] to [B] allow to obtain a relation over [A]
    out of a relation over [B]. *)

Definition RelCompFun {A A'} {B B' : Type}(R:Hetero.relation B B')(f:A->B) (f':A'->B') : Hetero.relation A A' :=
 fun a a' => R (f a) (f' a').

(** Instances on RelCompFun must match syntactically *)
Global Typeclasses Opaque RelCompFun.

(*
Infix "@@" := RelCompFun (at level 30, right associativity) : hetero_signature_scope.

Notation "R @@1" := (R @@ Fst)%signature (at level 30) : hetero_signature_scope.
Notation "R @@2" := (R @@ Snd)%signature (at level 30) : hetero_signature_scope.
*)

(** We define a product relation over [A*B]: each components should
    satisfy the corresponding initial relation. *)
Definition RelProd {A A' B B'} (RA:Hetero.relation A A')(RB:Hetero.relation B B') : Hetero.relation (A*B) (A'*B') :=
  Hetero.relation_conjunction
    (@RelCompFun (A * B) (A' * B') A A' RA fst fst)
    (@RelCompFun (A * B) (A' * B') B B' RB snd snd).

Global Typeclasses Opaque RelProd.

(*Infix "*" := RelProd : hetero_signature_scope.*)
(*
Section RelProd_Instances.

  Context {A : Type} {B : Type} (RA : relation A) (RB : relation B).

  Global Instance FstRel_sub :
    subrelation (RA*RB) (RA @@1).
  Proof. firstorder. Qed.

  Global Instance SndRel_sub :
    subrelation (RA*RB) (RB @@2).
  Proof. firstorder. Qed.

  Global Instance pair_compat :
    Proper (RA==>RB==> RA*RB) (@pair _ _).
  Proof. firstorder. Qed.

  Global Instance fst_compat :
    Proper (RA*RB ==> RA) Fst.
  Proof.
    intros (x,y) (x',y') (Hx,Hy); compute in *; auto.
  Qed.

  Global Instance snd_compat :
    Proper (RA*RB ==> RB) Snd.
  Proof.
    intros (x,y) (x',y') (Hx,Hy); compute in *; auto.
  Qed.

  Global Instance RelCompFun_compat (f:A->B)
           `(Proper _ (Ri==>Ri==>Ro) RB) :
    Proper (Ri@@f==>Ri@@f==>Ro) (RB@@f)%signature.
  Proof. unfold RelCompFun; firstorder. Qed.
End RelProd_Instances.
 *)
(*
#[global]
Hint Unfold RelProd RelCompFun : core.
#[global]
Hint Extern 2 (RelProd _ _ _ _) => split : core.
*)
