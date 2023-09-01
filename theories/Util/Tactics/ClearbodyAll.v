Ltac clearbody_all :=
  repeat match goal with
         | [ H := _ |- _ ] => clearbody H
         end.

Ltac clearbody_all_has_evar :=
  repeat match goal with
         | [ H := ?T |- _ ] => has_evar T; clearbody H
         end.
