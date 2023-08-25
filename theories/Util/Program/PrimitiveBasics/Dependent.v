From Coq Require Import Program.Basics Utf8.
From NeuralNetInterp.Util Require Import PrimitiveProd Program.Basics.Dependent.

Declare Scope primproj_type_function_scope.
Declare Scope primproj_type_function2_scope.
Declare Scope primproj_type_function3_scope.
Declare Scope primproj_type_function4_scope.
Delimit Scope primproj_type_function_scope with primproj_type_function.
Delimit Scope primproj_type_function2_scope with primproj_type_function2.
Delimit Scope primproj_type_function3_scope with primproj_type_function3.
Delimit Scope primproj_type_function4_scope with primproj_type_function4.

Notation "X * Y" := (fun A : Type => Primitive.prod (X%primproj_type_function%type_function A) (Y%primproj_type_function%type_function A)) : primproj_type_function_scope.
Notation "X * Y" := (fun A B : Type => Primitive.prod (X%primproj_type_function2%type_function2 A B) (Y%primproj_type_function2%type_function2 A B)) : primproj_type_function2_scope.
Notation "X * Y" := (fun A B C : Type => Primitive.prod (X%primproj_type_function3%type_function3 A B C) (Y%primproj_type_function3%type_function3 A B C)) : primproj_type_function3_scope.
Notation "X * Y" := (fun A B C D : Type => Primitive.prod (X%primproj_type_function4%type_function4 A B C D) (Y%primproj_type_function4%type_function4 A B C D)) : primproj_type_function4_scope.

Module Primitive.
  Export PrimitiveProd.Primitive.
  Module Notations.
    Export PrimitiveProd.Primitive.Notations.

    #[export] Set Warnings Append "-notation-overridden".
    Notation "X * Y" := (fun A : Type => prod (X%type_function A) (Y%type_function A)) : type_function_scope.
    Notation "X * Y" := (fun A B : Type => prod (X%type_function2 A B) (Y%type_function2 A B)) : type_function2_scope.
    Notation "X * Y" := (fun A B C : Type => prod (X%type_function3 A B C) (Y%type_function3 A B C)) : type_function3_scope.
    Notation "X * Y" := (fun A B C D : Type => prod (X%type_function4 A B C D) (Y%type_function4 A B C D)) : type_function4_scope.
    #[export] Set Warnings Append "notation-overridden".
  End Notations.
End Primitive.
