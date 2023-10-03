From Coq Require Import Program.Basics Utf8.
Declare Scope type_function_scope.
Declare Scope type_function2_scope.
Declare Scope type_function3_scope.
Declare Scope type_function4_scope.
Declare Scope type_function5_scope.
Declare Scope type_function6_scope.
Declare Scope type_functionn_scope.
Delimit Scope type_function_scope with type_function.
Delimit Scope type_function2_scope with type_function2.
Delimit Scope type_function3_scope with type_function3.
Delimit Scope type_function4_scope with type_function4.
Delimit Scope type_function5_scope with type_function5.
Delimit Scope type_function6_scope with type_function6.
Delimit Scope type_functionn_scope with type_functionn.
Local Open Scope type_scope.

#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

Definition type_function := Type -> Type.
Definition type_function2 := Type -> type_function.
Definition type_function3 := Type -> type_function2.
Definition type_function4 := Type -> type_function3.
Definition type_function5 := Type -> type_function4.
Definition type_function6 := Type -> type_function5.
Fixpoint type_functionn (n : nat)
  := match n with
     | O => Type
     | S n => Type -> type_functionn n
     end.
Bind Scope type_function_scope with type_function.
Bind Scope type_function2_scope with type_function2.
Bind Scope type_function3_scope with type_function3.
Bind Scope type_function4_scope with type_function4.
Bind Scope type_function5_scope with type_function5.
Bind Scope type_function6_scope with type_function6.
Bind Scope type_functionn_scope with type_functionn.

Fixpoint Constn' {n : nat} {m : nat} : type_functionn n -> type_functionn (m + n)
  := match m with
     | O => fun T => T
     | S m => fun T _ => @Constn' n m T
     end.

Fixpoint Constn {n : nat} : Type -> type_functionn n
  := match n with
     | O => fun T => T
     | S n => fun T _ => @Constn n T
     end.

Fixpoint Constn'r {n : nat} {m : nat} : type_functionn n -> type_functionn (n + m)
  := match n with
     | O => @Constn m
     | S n => fun T t => @Constn'r n m (T t)
     end.

Section liftTn1.
  Context (F : Type -> Type).
  Fixpoint liftTn1 {n : nat} : type_functionn n -> type_functionn n
    := match n return type_functionn n -> type_functionn n with
       | O => F
       | S n => fun X A => @liftTn1 n (X A)
       end.
End liftTn1.

Section liftTn2.
  Context (F : Type -> Type -> Type).
  Fixpoint liftTn2 {n : nat} : type_functionn n -> type_functionn n -> type_functionn n
    := match n return type_functionn n -> type_functionn n -> type_functionn n with
       | O => fun X Y => F X Y
       | S n => fun X Y A => @liftTn2 n (X A) (Y A)
       end.
End liftTn2.

Definition arrown {n : nat} : type_functionn n -> type_functionn n -> type_functionn n
  := liftTn2 (fun X Y => X -> Y).
Definition prodn {n : nat} : type_functionn n -> type_functionn n -> type_functionn n
  := liftTn2 (fun X Y => X * Y).
Definition sumn {n : nat} : type_functionn n -> type_functionn n -> type_functionn n
  := liftTn2 (fun X Y => X + Y).
Definition composen {n : nat} : type_function -> type_functionn n -> type_functionn n
  := fun F => liftTn1 F.
Fixpoint foralln0 {n : nat} : type_functionn (S n) -> type_functionn n
  := match n with
     | O => fun P => forall x, P x
     | S n => fun P A => @foralln0 n (P A)
     end.
Fixpoint foralln {n : nat} : type_functionn n -> Type
  := match n with
     | O => fun P => P
     | S n => fun P => forall A, @foralln n (P A)
     end.

Notation "X -> Y" := (fun A : Type => X%type_function A -> Y%type_function A) : type_function_scope.
Notation "X * Y" := (fun A : Type => prod (X%type_function A) (Y%type_function A)) : type_function_scope.
Notation "X + Y" := (fun A : Type => sum (X%type_function A) (Y%type_function A)) : type_function_scope.
Notation "F ∘ G" := (fun A : Type => F%type_function (G%type_function A)) : type_function_scope.
Notation "∀ x , P" := (fun A : Type => forall x : Type, P%type_function2 A x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function_scope.

Notation "X -> Y" := (fun A B : Type => X%type_function2 A B -> Y%type_function2 A B) : type_function2_scope.
Notation "X * Y" := (fun A B : Type => prod (X%type_function2 A B) (Y%type_function2 A B)) : type_function2_scope.
Notation "X + Y" := (fun A B : Type => sum (X%type_function2 A B) (Y%type_function2 A B)) : type_function2_scope.
Notation "F ∘ G" := (fun A B : Type => F%type_function (G%type_function2 A B)) : type_function2_scope.
Notation "∀ x , P" := (fun A B : Type => forall x : Type, P%type_function3 A B x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function2_scope.


Notation "X -> Y" := (fun A B C : Type => X%type_function3 A B C -> Y%type_function3 A B C) : type_function3_scope.
Notation "X * Y" := (fun A B C : Type => prod (X%type_function3 A B C) (Y%type_function3 A B C)) : type_function3_scope.
Notation "X + Y" := (fun A B C : Type => sum (X%type_function3 A B C) (Y%type_function3 A B C)) : type_function3_scope.
Notation "F ∘ G" := (fun A B C : Type => F%type_function (G%type_function3 A B C)) : type_function3_scope.
Notation "∀ x , P" := (fun A B C : Type => forall x : Type, P%type_function4 A B C x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function3_scope.

Notation "X -> Y" := (fun A B C D : Type => X%type_function4 A B C D -> Y%type_function4 A B C D) : type_function4_scope.
Notation "X * Y" := (fun A B C D : Type => prod (X%type_function4 A B C D) (Y%type_function4 A B C D)) : type_function4_scope.
Notation "X + Y" := (fun A B C D : Type => sum (X%type_function4 A B C D) (Y%type_function4 A B C D)) : type_function4_scope.
Notation "F ∘ G" := (fun A B C D : Type => F%type_function (G%type_function4 A B C D)) : type_function4_scope.
Notation "∀ x , P" := (fun A B C D : Type => forall x : Type, P%type_function5 A B C D x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function4_scope.

Notation "X -> Y" := (fun A B C D E : Type => X%type_function5 A B C D E -> Y%type_function5 A B C D E) : type_function5_scope.
Notation "X * Y" := (fun A B C D E : Type => prod (X%type_function5 A B C D E) (Y%type_function5 A B C D E)) : type_function5_scope.
Notation "X + Y" := (fun A B C D E : Type => sum (X%type_function5 A B C D E) (Y%type_function5 A B C D E)) : type_function5_scope.
Notation "F ∘ G" := (fun A B C D E : Type => F%type_function (G%type_function5 A B C D E)) : type_function5_scope.
Notation "∀ x , P" := (fun A B C D E : Type => forall x : Type, P%type_function6 A B C D E x)
                        (at level 200, x binder, right associativity, format "∀  x ,  P") : type_function5_scope.

Notation "X -> Y" := (fun A B C D E F : Type => X%type_function6 A B C D E F -> Y%type_function6 A B C D E F) : type_function6_scope.
Notation "X * Y" := (fun A B C D E F : Type => prod (X%type_function6 A B C D E F) (Y%type_function6 A B C D E F)) : type_function6_scope.
Notation "X + Y" := (fun A B C D E F : Type => sum (X%type_function6 A B C D E F) (Y%type_function6 A B C D E F)) : type_function6_scope.
Notation "F ∘ G" := (fun A B C D E F : Type => F%type_function (G%type_function6 A B C D E F)) : type_function6_scope.

Infix "->" := arrown : type_functionn_scope.
Infix "*" := prodn : type_functionn_scope.
Infix "+" := sumn : type_functionn_scope.
Infix "∘" := composen : type_functionn_scope.
