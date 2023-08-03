From Coq Require Import List Morphisms.
From NeuralNetInterp.Util Require Import Option.
From NeuralNetInterp.Util Require Morphisms.Dependent.
Import ListNotations.
Import Dependent.ProperNotations.
#[local] Open Scope list_scope.

Module List.
  #[export] Instance nth_error_Proper_dep_Forall2
  : Dependent.Proper (Forall2 ==> Dependent.const eq ==> @option_eq) (@nth_error).
  Proof.
    intros ?? HR ?? H n ? <-; revert n.
    now induction H, n; cbn.
  Qed.

  #[export] Instance nth_error_Proper_Forall2 {A R}
  : Proper (Forall2 R ==> eq ==> option_eq R) (@nth_error A).
  Proof. apply nth_error_Proper_dep_Forall2. Qed.
End List.
Export (hints) List.
