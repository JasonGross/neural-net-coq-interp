From Coq Require Import List Morphisms.
From NeuralNetInterp.Util Require Import Option.
Import ListNotations.
#[local] Open Scope list_scope.

Module List.
  #[export] Instance nth_error_Proper_Forall2 {A R}
  : Proper (Forall2 R ==> eq ==> option_eq R) (@nth_error A).
  Proof.
    intros ?? H n ? <-; revert n.
    now induction H, n; cbn.
  Qed.
End List.
Export (hints) List.
