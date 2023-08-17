From Coq Require Import Floats.
From NeuralNetInterp.Util.Arith Require Import Classes.
#[local] Open Scope core_scope.
#[local] Set Universe Polymorphism.
#[local] Set Primitive Projections.
#[local] Set Polymorphic Inductive Cumulativity.
#[local] Unset Universe Minimization ToSet.
#[local] Set Implicit Arguments.

Record Interval A := { lower : A ; upper : A }.
Record XInterval A := { inan : bool ; iinf : bool ; range : Interval A }.
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion range : XInterval >-> Interval.
#[local] Set Warnings Append "unsupported-attributes".
Record XBounded A {is_nanA : has_is_nan A} {is_infA : has_is_infinity A} {lebA : has_leb A}
  := { value : A
     ; xrange : XInterval A
     ; value_nan : True -> implb (is_nan value) xrange.(inan) = true
     ; value_inf : True -> implb (is_infinity value) xrange.(iinf) = true
     ; value_range : True -> implb (negb (is_nan value)) ((xrange.(lower) <=? value) && (value <=? xrange.(upper))) = true }.
#[global] Arguments XBounded A {_} {_} {_}.

Section generic.
  Context {A}
    {is_nanA : has_is_nan A} {is_infA : has_is_infinity A} {lebA : has_leb A}
    {minA : has_min A} {maxA : has_max A}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}.
End generic.
