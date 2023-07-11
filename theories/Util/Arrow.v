Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Definition feed {A} {B : A -> Type} (x : A) (f : forall a : A, B a) : B x := f x.
Definition feed_nd {A B} (x : A) (f : A -> B) : B := f x.
