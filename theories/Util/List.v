From Coq Require Export List.
Import ListNotations.
Open Scope list_scope.

Definition rev {A} (ls : list A) := rev_append ls [].
Definition snoc {A} (xs : list A) (x : A) : list A := xs ++ [x].
Fixpoint droplast {A} (ls : list A) : list A
  := match ls with
     | [] => []
     | [x] => []
     | x :: xs => droplast xs
     end.
