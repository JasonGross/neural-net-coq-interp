From Flocq.IEEE754 Require Import BinarySingleNaN.
From NeuralNetInterp.Util Require Import Notations.

Declare Scope binary_float_scope.
Delimit Scope binary_float_scope with binary_float.
#[global] Bind Scope binary_float_scope with binary_float.

Infix "+" := (Bplus mode_NE) : binary_float_scope.
Infix "*" := (Bmult mode_NE) : binary_float_scope.
Infix "/" := (Bdiv mode_NE) : binary_float_scope.
Infix "-" := (Bminus mode_NE) : binary_float_scope.
Notation "- x" := (Bopp x) : binary_float_scope.
Infix "<?" := Bltb : binary_float_scope.
Infix "<=?" := Bleb : binary_float_scope.
Infix "=?" := Beqb : binary_float_scope.
