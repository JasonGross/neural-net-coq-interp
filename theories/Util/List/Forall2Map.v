From Coq Require Import List.
Import ListNotations.
#[local] Open Scope list_scope.

Module List.
  Lemma Forall2_map {A B A' B' f g P xs ys}
    : @Forall2 A' B' P (map f xs) (map g ys) <-> @Forall2 A B (fun x y => P (f x) (g y)) xs ys.
  Proof.
    split; revert ys; induction xs as [|x xs IH], ys as [|y ys]; cbn [map]; intro H;
      inversion H; clear H; subst; constructor; eauto.
  Qed.
End List.
