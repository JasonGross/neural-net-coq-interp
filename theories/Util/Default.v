Definition with_default {A} (x : A) := A.
#[global] Arguments with_default {_} _, _ _.
Existing Class with_default.
#[global] Hint Extern 0 (with_default ?x) => exact x : typeclass_instances.
#[global] Hint Unfold with_default : typeclass_instances.
