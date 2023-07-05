From Coq Require Import List.
From NeuralNetInterp.Util Require Import Notations.
Import ListNotations.
Local Open Scope list_scope.

Set Universe Polymorphism.
Set Polymorphic Inductive Cumulativity.
Unset Universe Minimization ToSet.

Class Monad@{d c} (m : Type@{d} -> Type@{c}) : Type :=
{ ret : forall {t : Type@{d}}, t -> m t
; bind : forall {t u : Type@{d}}, m t -> (t -> m u) -> m u
}.

Module MonadNotation.
  Declare Scope monad_scope.
  Delimit Scope monad_scope with monad.

  Notation "c >>= f" := (@bind _ _ _ _ c%monad f%monad) : monad_scope.
  Notation "f =<< c" := (@bind _ _ _ _ c%monad f%monad) : monad_scope.
  Notation "x <-- c1 ;; c2" := (@bind _ _ _ _ c1%monad (fun x => c2%monad)) : monad_scope.
  Notation "' pat <-- c1 ;; c2" := (@bind _ _ _ _ c1%monad (fun 'pat => c2%monad)) : monad_scope.
  Notation "e1 ;; e2" := (_ <-- e1%monad ;; e2%monad)%monad : monad_scope.
End MonadNotation.

Import MonadNotation.

#[global] Instance option_monad : Monad (fun x => option x) :=
  {| ret A a := Some a ;
     bind A B m f :=
       match m with
       | Some a => f a
       | None => None
       end
  |}.

#[global] Instance id_monad : Monad (fun x => x) :=
  {| ret A a := a ;
     bind A B m f := f m
  |}.

Local Open Scope monad_scope.

(* state is underneath the other monad *)
Definition StateT S M T := S -> M (T * S)%type.
Definition StateT_Monad {S T} {TM : Monad T} : Monad (StateT S T) :=
  {| ret A a st := ret (a, st) ;
     bind A B m f st := '(m, st) <-- m st;; f m st
  |}.
#[export] Hint Extern 1 (Monad (StateT ?S ?T)) => simple apply (@StateT_Monad S T) : typeclass_instances.

Module State.
  Definition get {S T} {TM : Monad T} : StateT S T S
    := fun s => ret (s, s).
  Definition update {S T} {TM : Monad T} (f : S -> S) : StateT S T unit
    := fun s => ret (tt, f s).
  Definition set {S T} {TM : Monad T} (s : S) : StateT S T unit
    := update (fun _ => s).
  Definition lift {S T} {TM : Monad T} {A} (m : T A) : StateT S T A
    := fun s => v <-- m;; ret (v, s).
  Definition evalStateT {S T A} {TM : Monad T} (p : StateT S T A) (st : S) : T A
    := '(v, st) <-- p st;;
       ret v.
End State.

Section MapOpt.
  Context {A} {B} (f : A -> option B).

  Fixpoint mapopt (l : list A) : option (list B) :=
    match l with
    | nil => ret nil
    | x :: xs => x' <-- f x ;;
                xs' <-- mapopt xs ;;
                ret (x' :: xs')
    end.
End MapOpt.

Section MonadOperations.
  Context {T : Type -> Type} {M : Monad T}.
  Context {A B} (f : A -> T B).
  Fixpoint monad_map (l : list A)
    : T (list B)
    := match l with
       | nil => ret nil
       | x :: l => x' <-- f x ;;
                  l' <-- monad_map l ;;
                  ret (x' :: l')
       end.

  Definition monad_option_map (l : option A)
    : T (option B)
    := match l with
       | None => ret None
       | Some x => x' <-- f x ;;
                   ret (Some x')
       end.

  Context (g : A -> B -> T A).
  Fixpoint monad_fold_left (l : list B) (x : A) : T A
    := match l with
       | nil => ret x
       | y :: l => x' <-- g x y ;;
                   monad_fold_left l x'
       end.

  Fixpoint monad_fold_right (l : list B) (x : A) : T A
       := match l with
          | nil => ret x
          | y :: l => l' <-- monad_fold_right l x ;;
                      g l' y
          end.

  Context (h : nat -> A -> T B).
  Fixpoint monad_map_i_aux (n0 : nat) (l : list A) : T (list B)
    := match l with
       | nil => ret nil
       | x :: l => x' <-- (h n0 x) ;;
                   l' <-- (monad_map_i_aux (S n0) l) ;;
                   ret (x' :: l')
       end.

  Definition monad_map_i := @monad_map_i_aux 0.
End MonadOperations.

Definition monad_iter {T : Type -> Type} {M A} (f : A -> T unit) (l : list A) : T unit
  := @monad_fold_left T M _ _ (fun _ => f) l tt.
