From Coq Require Import List.
#[local] Open Scope list_scope.
#[local] Set Implicit Arguments.
Module List.
  Section __.
    Context (A : Type).

    Lemma nth_error_match l n
      : nth_error l n
        = match n, l with
          | O, x :: _ => Some x
          | S n, _ :: l => @nth_error A l n
          | _, _ => None
          end.
    Proof using Type. destruct n; reflexivity. Qed.

    Lemma nth_error_nil n : @nth_error A nil n = None.
    Proof using Type. destruct n; reflexivity. Qed.

    Lemma nth_error_cons x xs n
      : nth_error (x :: xs) n
        = match n with
          | O => Some x
          | S n => @nth_error A xs n
          end.
    Proof using Type. apply nth_error_match. Qed.

    Lemma nth_error_O l
      : nth_error l O = @hd_error A l.
    Proof using Type. destruct l; reflexivity. Qed.

    Lemma nth_error_S l n
      : nth_error l (S n) = @nth_error A (tl l) n.
    Proof using Type. destruct l; rewrite ?nth_error_nil; reflexivity. Qed.
  End __.
End List.
