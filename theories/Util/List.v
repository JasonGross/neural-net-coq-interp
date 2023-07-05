From Coq Require Export List.
Import ListNotations.
Open Scope list_scope.

Definition rev {A} (ls : list A) := rev_append ls [].
Definition snoc {A} (xs : list A) (x : A) : list A := xs ++ [x].
