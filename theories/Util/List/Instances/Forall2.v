From Coq Require Import List Morphisms Program.Basics.
From NeuralNetInterp.Util Require RelationClasses.Dependent.
Import ListNotations.
#[local] Open Scope list_scope.

Module List.
  #[export] Instance Forall2_refl {A R} {_ : Reflexive R} : Reflexive (@Forall2 A A R).
  Proof.
    intro ls; induction ls as [|?? IH]; constructor; eauto.
  Qed.

  #[export] Instance Forall2_refl_dep : Dependent.Reflexive (@Forall2) := @Forall2_refl.

  Lemma Forall2_flip {A B R xs ys} : @Forall2 A B R xs ys -> @Forall2 B A (flip R) ys xs.
  Proof.
    cbv [flip]; induction 1; constructor; eauto.
  Qed.

  #[export] Instance Forall2_sym {A R} {_ : Symmetric R} : Symmetric (@Forall2 A A R).
  Proof.
    repeat intro; eapply Forall2_flip, Forall2_impl; eauto.
  Qed.

  Lemma Forall2_trans_hetero {A B C} {RAB RBC RAC : _ -> _ -> Prop} {xs ys zs} : (forall x y z, RAB x y -> RBC y z -> RAC x z) -> @Forall2 A B RAB xs ys -> @Forall2 B C RBC ys zs -> @Forall2 A C RAC xs zs.
  Proof.
    intros HT H; revert zs; induction H; intro zs; inversion 1; subst; constructor; eauto.
  Qed.

  #[export] Instance Forall2_trans {A R} {_ : Transitive R} : Transitive (@Forall2 A A R).
  Proof. repeat intro; eapply Forall2_trans_hetero; eassumption. Qed.
End List.
Export (hints) List.
