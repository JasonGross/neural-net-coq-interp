Set Implicit Arguments.
Inductive value_eq A : A -> Type := the_value (x : A) : value_eq x.
Definition extract {A x} (val : value_eq x) : A := match val with the_value x => x end.
Lemma extract_eq {A x} (val : @value_eq A x) : extract val = x.
Proof. destruct val; reflexivity. Qed.
Notation native_compute x := (match x return value_eq x with y => ltac:(native_compute; constructor) end)
                               (only parsing).
Notation vm_compute x := (match x return value_eq x with y => ltac:(vm_compute; constructor) end)
                           (only parsing).
