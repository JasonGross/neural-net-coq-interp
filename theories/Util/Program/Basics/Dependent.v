From Coq Require Import Program.Basics Utf8.
Declare Scope type_function_scope.
Declare Scope type_function2_scope.
Declare Scope type_function3_scope.
Declare Scope type_function4_scope.
Delimit Scope type_function_scope with type_function.
Delimit Scope type_function2_scope with type_function2.
Delimit Scope type_function3_scope with type_function3.
Delimit Scope type_function4_scope with type_function4.
Local Open Scope type_scope.

#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

Definition type_function := Type -> Type.
Definition type_function2 := Type -> type_function.
Definition type_function3 := Type -> type_function2.
Definition type_function4 := Type -> type_function3.
Bind Scope type_function_scope with type_function.
Bind Scope type_function2_scope with type_function2.
Bind Scope type_function3_scope with type_function3.
Bind Scope type_function4_scope with type_function4.

Notation "X -> Y" := (fun A : Type => X%type_function A -> Y%type_function A) : type_function_scope.
Notation "X * Y" := (fun A : Type => prod (X%type_function A) (Y%type_function A)) : type_function_scope.
Notation "X + Y" := (fun A : Type => sum (X%type_function A) (Y%type_function A)) : type_function_scope.
Notation "F ∘ G" := (fun A : Type => F%type_function (G%type_function A)) : type_function_scope.
Notation "∀ x , P" := (fun A : Type => forall x : Type, P%type_function2 A x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function_scope.

Notation "X -> Y" := (fun A B : Type => X%type_function2 A B -> Y%type_function2 A B) : type_function2_scope.
Notation "X * Y" := (fun A B : Type => prod (X%type_function2 A B) (Y%type_function2 A B)) : type_function2_scope.
Notation "X + Y" := (fun A B : Type => sum (X%type_function2 A B) (Y%type_function2 A B)) : type_function2_scope.
Notation "F ∘ G" := (fun A B : Type => F%type_function2 (G%type_function2 A B)) : type_function2_scope.
Notation "∀ x , P" := (fun A B : Type => forall x : Type, P%type_function3 A B x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function2_scope.


Notation "X -> Y" := (fun A B C : Type => X%type_function3 A B C -> Y%type_function3 A B C) : type_function3_scope.
Notation "X * Y" := (fun A B C : Type => prod (X%type_function3 A B C) (Y%type_function3 A B C)) : type_function3_scope.
Notation "X + Y" := (fun A B C : Type => sum (X%type_function3 A B C) (Y%type_function3 A B C)) : type_function3_scope.
Notation "F ∘ G" := (fun A B C : Type => F%type_function3 (G%type_function3 A B C)) : type_function3_scope.
Notation "∀ x , P" := (fun A B C : Type => forall x : Type, P%type_function4 A B C x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function3_scope.

Notation "X -> Y" := (fun A B C D : Type => X%type_function4 A B C D -> Y%type_function4 A B C D) : type_function4_scope.
Notation "X * Y" := (fun A B C D : Type => prod (X%type_function4 A B C D) (Y%type_function4 A B C D)) : type_function4_scope.
Notation "X + Y" := (fun A B C D : Type => sum (X%type_function4 A B C D) (Y%type_function4 A B C D)) : type_function4_scope.
Notation "F ∘ G" := (fun A B C D : Type => F%type_function4 (G%type_function4 A B C D)) : type_function4_scope.
