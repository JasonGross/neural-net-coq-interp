From Coq Require Import Lia Arith List.
From NeuralNetInterp.Util Require Import List.
Import ListNotations.
Open Scope list_scope.
Set Implicit Arguments.

Module List.
  Lemma nth_error_seq n start len : nth_error (seq start len) n = if n <? len then Some (start + n) else None.
  Proof.
    revert len start; induction n as [|n IH], len as [|len]; cbn; intros; try reflexivity.
    { f_equal; lia. }
    { rewrite IH; destruct len; cbn; try reflexivity.
      destruct (n <=? len); f_equal; lia. }
  Qed.
End List.
