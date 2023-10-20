From Coq Require Import PeanoNat Relations.Relation_Definitions.
From NeuralNetInterp.Util Require Import PolymorphicList.
From NeuralNetInterp.Util.Program Require Import Basics.Dependent.
From NeuralNetInterp.Util.Relations Require Import Relation_Definitions.Hetero Relation_Definitions.DependentHetero.
Import ListNotations.
#[local] Open Scope polymorphic_list_scope.
#[local] Set Implicit Arguments.
#[local] Set Universe Polymorphism.
#[local] Unset Universe Minimization ToSet.
#[local] Set Polymorphic Inductive Cumulativity.

#[export] Set Warnings Append "-overwriting-delimiting-key".
Declare Scope dependent_signature_scope.
Declare Scope dependent2_signature_scope.
Declare Scope dependent3_signature_scope.
Declare Scope dependent4_signature_scope.
Declare Scope dependent5_signature_scope.
Declare Scope dependent6_signature_scope.
Declare Scope dependentn_signature_scope.
#[export] Set Warnings Append "overwriting-delimiting-key".

Definition relationn {n : nat} (F : type_functionn n) : Type
  := DependentHetero.relationn F F.
Definition relation : type_function -> Type
  := Eval cbv [relationn DependentHetero.relationn] in @relationn 1.
Definition relation2 : type_function2 -> Type
  := Eval cbv [relationn DependentHetero.relationn] in @relationn 2.
Definition relation3 : type_function3 -> Type
  := Eval cbv [relationn DependentHetero.relationn] in @relationn 3.
Definition relation4 : type_function4 -> Type
  := Eval cbv [relationn DependentHetero.relationn] in @relationn 4.
Definition relation5 : type_function5 -> Type
  := Eval cbv [relationn DependentHetero.relationn] in @relationn 5.
Definition relation6 : type_function6 -> Type
  := Eval cbv [relationn DependentHetero.relationn] in @relationn 6.

Definition lift_comp {F} (R : relation F) {G} (R' : relation G) : relation (F ∘ G)
  := fun A B RAB => R _ _ (R' _ _ RAB).

Module Export RelationsNotations.

  #[export] Set Warnings Append "-overwriting-delimiting-key".
  Delimit Scope dependent_signature_scope with dependent_signature.
  Delimit Scope dependent_signature_scope with signatureD.
  Bind Scope dependent_signature_scope with relation.

  Delimit Scope dependent2_signature_scope with dependent2_signature.
  Delimit Scope dependent2_signature_scope with signatureD2.
  Bind Scope dependent2_signature_scope with relation2.

  Delimit Scope dependent3_signature_scope with dependent3_signature.
  Delimit Scope dependent3_signature_scope with signatureD3.
  Bind Scope dependent3_signature_scope with relation3.

  Delimit Scope dependent4_signature_scope with dependent4_signature.
  Delimit Scope dependent4_signature_scope with signatureD4.
  Bind Scope dependent4_signature_scope with relation4.

  Delimit Scope dependent5_signature_scope with dependent5_signature.
  Delimit Scope dependent5_signature_scope with signatureD5.
  Bind Scope dependent5_signature_scope with relation5.

  Delimit Scope dependent6_signature_scope with dependent6_signature.
  Delimit Scope dependent6_signature_scope with signatureD6.
  Bind Scope dependent6_signature_scope with relation6.

  Delimit Scope dependentn_signature_scope with dependentn_signature.
  Delimit Scope dependentn_signature_scope with signatureDn.
  Bind Scope dependentn_signature_scope with relationn.
  #[export] Set Warnings Append "overwriting-delimiting-key".

  Notation "R ∘ R'" := (fun (A B : Type) (RAB : Hetero.relation A B) => R%dependent_signature _ _ (R'%dependent_signature A B RAB)) : dependent_signature_scope.
End RelationsNotations.

Module Import Internal.
  Variant arg {T} := argn (n : nat) | argv (v : T).
  Definition map_arg {A B} (f : A -> B) (v : @arg A) : @arg B
    := match v with
       | argn n => argn n
       | argv v => argv (f v)
       end.
  Record relation_type_arg := { A : Type ; B : Type ; R : Hetero.relation A B }.
  Fixpoint apply_default {n} (default : nat -> Type) (args : list (@arg Type)) : type_functionn (length args + n) -> type_functionn n
    := match args with
       | [] => fun f => f
       | argn i :: xs => fun f => @apply_default n default xs (f (default i))
       | argv v :: xs => fun f => @apply_default n default xs (f v)
       end.
  Definition transparentify {n m : nat} (pf : n = m) : n = m
    := Eval cbv in match Nat.eq_dec n m with
                   | left pf => pf
                   | right npf => match npf pf with end
                   end.
  Import EqNotations.
  (*
  Fixpoint apply_default_relation {n} (default : nat -> relation_type_arg) (args : list (@arg relation_type_arg))
    : forall (pf1 pf2 : _ = length _) (F G : type_functionn (length args + n)),
      DependentHetero.relationn F G
      -> DependentHetero.relationn
           (apply_default (fun i => (default i).(A)) (map (map_arg A) args) (rew [fun la => type_functionn (la + n)] (transparentify pf1) in F))
           (apply_default (fun i => (default i).(B)) (map (map_arg B) args) (rew [fun la => type_functionn (la + n)] (transparentify pf2) in G)).
    refine match args with
           | [] => fun pf1 pf2 F G R => R
           | argn i :: xs => fun pf1 pf2 F G Rf => _
           | argv v :: xs => _
           end; cbn in *.
    epose (@apply_default_relation n default xs _ _ _ _ (Rf _ _ (default i).(R))).

      -> type_functionn (length args + n) -> type_functionn n
    := match args with
       | [] => fun f => f
       | argn i :: xs => fun f => @apply_default n default xs (f (default i))
       | argv v :: xs => fun f => @apply_default n default xs (f v)
       end.
*)
  Definition arg_use {T} (val : T) (a : arg) : arg
    := match a with
       | argn O => argv val
       | argv v => argv v
       | argn (S n) => argn n
       end.
End Internal.
#[local] Set Warnings "-uniform-inheritance".
#[local] Coercion argn : nat >-> arg.
#[local] Set Warnings "uniform-inheritance".
Import EqNotations.
Fixpoint lift_specialize {n m : nat} : forall args : list arg, type_functionn (length args + n) -> type_functionn (m + n)
  := match m with
     | O => apply_default (fun _ => unit)
     | S m => fun args f A => @lift_specialize n m (PolymorphicList.map (arg_use A) args) (rew <- [fun la => type_functionn (la + n)] transparentify (PolymorphicList.map_length _ _) in f)
     end.

#[local] Notation "'cbv!' x" := (match x return _ with y => ltac:(let v := (eval cbv [y lift_specialize apply_default eq_rect eq_rect_r eq_sym transparentify arg_use PolymorphicList.length PolymorphicList.map] in y) in exact v) end) (only parsing, at level 100).

(*Check cbv! fun (F : type_function) => @lift_specialize 0 2 (map argn [0]) F.*)
(**
<<<
# %%
import itertools

LETTERS_NO_F = 'ABCDEGHIHJKLMNOPQRSTUVWXYZ'
def make_args(n, subset):
    args = ['_'] * n
    for i, l in zip(subset, LETTERS_NO_F):
        args[i] = l
    return ' '.join(args)

def make_args_rel(n, subset):
    args = ['_'] * 3 * n
    for iv, i in enumerate(subset):
        args[3 * i + 2] = f'R{iv}'
    return ' '.join(args)

def make_app_rel(subset):
    return ' '.join(f'_ _ R{i}' for i in range(len(subset)))

for i in range(2, 7):
    # get all strict subsets of {0, ..., i-1}
    for leni in range(1, i):
        for subset in itertools.combinations(range(i), leni):
            print(f"""Definition lift{i}_{''.join(str(k+1) for k in subset)} {{F}} (R : relation{'' if len(subset) == 1 else len(subset)} F) : relation{i} (fun {make_args(i, subset)} => F {' '.join(LETTERS_NO_F[:len(subset)])})
  := fun {make_args_rel(i, subset)} => R {make_app_rel(subset)}.""")

# %%
*)
Definition lift2_1 {F} (R : relation F) : relation2 (fun A _ => F A)
  := fun _ _ R0 _ _ _ => R _ _ R0.
Definition lift2_2 {F} (R : relation F) : relation2 (fun _ A => F A)
  := fun _ _ _ _ _ R0 => R _ _ R0.
Definition lift3_1 {F} (R : relation F) : relation3 (fun A _ _ => F A)
  := fun _ _ R0 _ _ _ _ _ _ => R _ _ R0.
Definition lift3_2 {F} (R : relation F) : relation3 (fun _ A _ => F A)
  := fun _ _ _ _ _ R0 _ _ _ => R _ _ R0.
Definition lift3_3 {F} (R : relation F) : relation3 (fun _ _ A => F A)
  := fun _ _ _ _ _ _ _ _ R0 => R _ _ R0.
Definition lift3_12 {F} (R : relation2 F) : relation3 (fun A B _ => F A B)
  := fun _ _ R0 _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift3_13 {F} (R : relation2 F) : relation3 (fun A _ B => F A B)
  := fun _ _ R0 _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift3_23 {F} (R : relation2 F) : relation3 (fun _ A B => F A B)
  := fun _ _ _ _ _ R0 _ _ R1 => R _ _ R0 _ _ R1.
Definition lift4_1 {F} (R : relation F) : relation4 (fun A _ _ _ => F A)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ => R _ _ R0.
Definition lift4_2 {F} (R : relation F) : relation4 (fun _ A _ _ => F A)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ => R _ _ R0.
Definition lift4_3 {F} (R : relation F) : relation4 (fun _ _ A _ => F A)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ => R _ _ R0.
Definition lift4_4 {F} (R : relation F) : relation4 (fun _ _ _ A => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 => R _ _ R0.
Definition lift4_12 {F} (R : relation2 F) : relation4 (fun A B _ _ => F A B)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift4_13 {F} (R : relation2 F) : relation4 (fun A _ B _ => F A B)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift4_14 {F} (R : relation2 F) : relation4 (fun A _ _ B => F A B)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift4_23 {F} (R : relation2 F) : relation4 (fun _ A B _ => F A B)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift4_24 {F} (R : relation2 F) : relation4 (fun _ A _ B => F A B)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift4_34 {F} (R : relation2 F) : relation4 (fun _ _ A B => F A B)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 => R _ _ R0 _ _ R1.
Definition lift4_123 {F} (R : relation3 F) : relation4 (fun A B C _ => F A B C)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift4_124 {F} (R : relation3 F) : relation4 (fun A B _ C => F A B C)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift4_134 {F} (R : relation3 F) : relation4 (fun A _ B C => F A B C)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift4_234 {F} (R : relation3 F) : relation4 (fun _ A B C => F A B C)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_1 {F} (R : relation F) : relation5 (fun A _ _ _ _ => F A)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ _ _ _ => R _ _ R0.
Definition lift5_2 {F} (R : relation F) : relation5 (fun _ A _ _ _ => F A)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ _ _ _ => R _ _ R0.
Definition lift5_3 {F} (R : relation F) : relation5 (fun _ _ A _ _ => F A)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ _ _ _ => R _ _ R0.
Definition lift5_4 {F} (R : relation F) : relation5 (fun _ _ _ A _ => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 _ _ _ => R _ _ R0.
Definition lift5_5 {F} (R : relation F) : relation5 (fun _ _ _ _ A => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ R0 => R _ _ R0.
Definition lift5_12 {F} (R : relation2 F) : relation5 (fun A B _ _ _ => F A B)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift5_13 {F} (R : relation2 F) : relation5 (fun A _ B _ _ => F A B)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift5_14 {F} (R : relation2 F) : relation5 (fun A _ _ B _ => F A B)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift5_15 {F} (R : relation2 F) : relation5 (fun A _ _ _ B => F A B)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift5_23 {F} (R : relation2 F) : relation5 (fun _ A B _ _ => F A B)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift5_24 {F} (R : relation2 F) : relation5 (fun _ A _ B _ => F A B)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift5_25 {F} (R : relation2 F) : relation5 (fun _ A _ _ B => F A B)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift5_34 {F} (R : relation2 F) : relation5 (fun _ _ A B _ => F A B)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift5_35 {F} (R : relation2 F) : relation5 (fun _ _ A _ B => F A B)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift5_45 {F} (R : relation2 F) : relation5 (fun _ _ _ A B => F A B)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 _ _ R1 => R _ _ R0 _ _ R1.
Definition lift5_123 {F} (R : relation3 F) : relation5 (fun A B C _ _ => F A B C)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_124 {F} (R : relation3 F) : relation5 (fun A B _ C _ => F A B C)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_125 {F} (R : relation3 F) : relation5 (fun A B _ _ C => F A B C)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_134 {F} (R : relation3 F) : relation5 (fun A _ B C _ => F A B C)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_135 {F} (R : relation3 F) : relation5 (fun A _ B _ C => F A B C)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_145 {F} (R : relation3 F) : relation5 (fun A _ _ B C => F A B C)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_234 {F} (R : relation3 F) : relation5 (fun _ A B C _ => F A B C)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_235 {F} (R : relation3 F) : relation5 (fun _ A B _ C => F A B C)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_245 {F} (R : relation3 F) : relation5 (fun _ A _ B C => F A B C)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_345 {F} (R : relation3 F) : relation5 (fun _ _ A B C => F A B C)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift5_1234 {F} (R : relation4 F) : relation5 (fun A B C D _ => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift5_1235 {F} (R : relation4 F) : relation5 (fun A B C _ D => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift5_1245 {F} (R : relation4 F) : relation5 (fun A B _ C D => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift5_1345 {F} (R : relation4 F) : relation5 (fun A _ B C D => F A B C D)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift5_2345 {F} (R : relation4 F) : relation5 (fun _ A B C D => F A B C D)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1 {F} (R : relation F) : relation6 (fun A _ _ _ _ _ => F A)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => R _ _ R0.
Definition lift6_2 {F} (R : relation F) : relation6 (fun _ A _ _ _ _ => F A)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ _ _ _ _ _ _ => R _ _ R0.
Definition lift6_3 {F} (R : relation F) : relation6 (fun _ _ A _ _ _ => F A)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ _ _ _ _ _ _ => R _ _ R0.
Definition lift6_4 {F} (R : relation F) : relation6 (fun _ _ _ A _ _ => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 _ _ _ _ _ _ => R _ _ R0.
Definition lift6_5 {F} (R : relation F) : relation6 (fun _ _ _ _ A _ => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ R0 _ _ _ => R _ _ R0.
Definition lift6_6 {F} (R : relation F) : relation6 (fun _ _ _ _ _ A => F A)
  := fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ R0 => R _ _ R0.
Definition lift6_12 {F} (R : relation2 F) : relation6 (fun A B _ _ _ _ => F A B)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_13 {F} (R : relation2 F) : relation6 (fun A _ B _ _ _ => F A B)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_14 {F} (R : relation2 F) : relation6 (fun A _ _ B _ _ => F A B)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_15 {F} (R : relation2 F) : relation6 (fun A _ _ _ B _ => F A B)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_16 {F} (R : relation2 F) : relation6 (fun A _ _ _ _ B => F A B)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift6_23 {F} (R : relation2 F) : relation6 (fun _ A B _ _ _ => F A B)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_24 {F} (R : relation2 F) : relation6 (fun _ A _ B _ _ => F A B)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_25 {F} (R : relation2 F) : relation6 (fun _ A _ _ B _ => F A B)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_26 {F} (R : relation2 F) : relation6 (fun _ A _ _ _ B => F A B)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift6_34 {F} (R : relation2 F) : relation6 (fun _ _ A B _ _ => F A B)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_35 {F} (R : relation2 F) : relation6 (fun _ _ A _ B _ => F A B)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_36 {F} (R : relation2 F) : relation6 (fun _ _ A _ _ B => F A B)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift6_45 {F} (R : relation2 F) : relation6 (fun _ _ _ A B _ => F A B)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ _ => R _ _ R0 _ _ R1.
Definition lift6_46 {F} (R : relation2 F) : relation6 (fun _ _ _ A _ B => F A B)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 _ _ _ _ _ R1 => R _ _ R0 _ _ R1.
Definition lift6_56 {F} (R : relation2 F) : relation6 (fun _ _ _ _ A B => F A B)
  := fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ R0 _ _ R1 => R _ _ R0 _ _ R1.
Definition lift6_123 {F} (R : relation3 F) : relation6 (fun A B C _ _ _ => F A B C)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ _ _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_124 {F} (R : relation3 F) : relation6 (fun A B _ C _ _ => F A B C)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ _ _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_125 {F} (R : relation3 F) : relation6 (fun A B _ _ C _ => F A B C)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_126 {F} (R : relation3 F) : relation6 (fun A B _ _ _ C => F A B C)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_134 {F} (R : relation3 F) : relation6 (fun A _ B C _ _ => F A B C)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ _ _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_135 {F} (R : relation3 F) : relation6 (fun A _ B _ C _ => F A B C)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_136 {F} (R : relation3 F) : relation6 (fun A _ B _ _ C => F A B C)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_145 {F} (R : relation3 F) : relation6 (fun A _ _ B C _ => F A B C)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_146 {F} (R : relation3 F) : relation6 (fun A _ _ B _ C => F A B C)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_156 {F} (R : relation3 F) : relation6 (fun A _ _ _ B C => F A B C)
  := fun _ _ R0 _ _ _ _ _ _ _ _ _ _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_234 {F} (R : relation3 F) : relation6 (fun _ A B C _ _ => F A B C)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_235 {F} (R : relation3 F) : relation6 (fun _ A B _ C _ => F A B C)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_236 {F} (R : relation3 F) : relation6 (fun _ A B _ _ C => F A B C)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_245 {F} (R : relation3 F) : relation6 (fun _ A _ B C _ => F A B C)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_246 {F} (R : relation3 F) : relation6 (fun _ A _ B _ C => F A B C)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_256 {F} (R : relation3 F) : relation6 (fun _ A _ _ B C => F A B C)
  := fun _ _ _ _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_345 {F} (R : relation3 F) : relation6 (fun _ _ A B C _ => F A B C)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ _ => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_346 {F} (R : relation3 F) : relation6 (fun _ _ A B _ C => F A B C)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_356 {F} (R : relation3 F) : relation6 (fun _ _ A _ B C => F A B C)
  := fun _ _ _ _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_456 {F} (R : relation3 F) : relation6 (fun _ _ _ A B C => F A B C)
  := fun _ _ _ _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ R2 => R _ _ R0 _ _ R1 _ _ R2.
Definition lift6_1234 {F} (R : relation4 F) : relation6 (fun A B C D _ _ => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ _ _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1235 {F} (R : relation4 F) : relation6 (fun A B C _ D _ => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ R3 _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1236 {F} (R : relation4 F) : relation6 (fun A B C _ _ D => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ _ _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1245 {F} (R : relation4 F) : relation6 (fun A B _ C D _ => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ R3 _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1246 {F} (R : relation4 F) : relation6 (fun A B _ C _ D => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ _ _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1256 {F} (R : relation4 F) : relation6 (fun A B _ _ C D => F A B C D)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ _ _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1345 {F} (R : relation4 F) : relation6 (fun A _ B C D _ => F A B C D)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ R3 _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1346 {F} (R : relation4 F) : relation6 (fun A _ B C _ D => F A B C D)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ _ _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1356 {F} (R : relation4 F) : relation6 (fun A _ B _ C D => F A B C D)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ _ _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_1456 {F} (R : relation4 F) : relation6 (fun A _ _ B C D => F A B C D)
  := fun _ _ R0 _ _ _ _ _ _ _ _ R1 _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_2345 {F} (R : relation4 F) : relation6 (fun _ A B C D _ => F A B C D)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_2346 {F} (R : relation4 F) : relation6 (fun _ A B C _ D => F A B C D)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_2356 {F} (R : relation4 F) : relation6 (fun _ A B _ C D => F A B C D)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_2456 {F} (R : relation4 F) : relation6 (fun _ A _ B C D => F A B C D)
  := fun _ _ _ _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_3456 {F} (R : relation4 F) : relation6 (fun _ _ A B C D => F A B C D)
  := fun _ _ _ _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ R3 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3.
Definition lift6_12345 {F} (R : relation5 F) : relation6 (fun A B C D E _ => F A B C D E)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4 _ _ _ => R _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4.
Definition lift6_12346 {F} (R : relation5 F) : relation6 (fun A B C D _ E => F A B C D E)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ _ _ _ R4 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4.
Definition lift6_12356 {F} (R : relation5 F) : relation6 (fun A B C _ D E => F A B C D E)
  := fun _ _ R0 _ _ R1 _ _ R2 _ _ _ _ _ R3 _ _ R4 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4.
Definition lift6_12456 {F} (R : relation5 F) : relation6 (fun A B _ C D E => F A B C D E)
  := fun _ _ R0 _ _ R1 _ _ _ _ _ R2 _ _ R3 _ _ R4 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4.
Definition lift6_13456 {F} (R : relation5 F) : relation6 (fun A _ B C D E => F A B C D E)
  := fun _ _ R0 _ _ _ _ _ R1 _ _ R2 _ _ R3 _ _ R4 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4.
Definition lift6_23456 {F} (R : relation5 F) : relation6 (fun _ A B C D E => F A B C D E)
  := fun _ _ _ _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4 => R _ _ R0 _ _ R1 _ _ R2 _ _ R3 _ _ R4.
(*Definition liftn_m {n m} (args : list nat) {F} (R : relationn F)
  : relationn (@lift_specialize n m (map argn args) F).
*)
Section Relation_Definition.

  Variable F : type_function.


  (*
  Variable R : relation.

  Section General_Properties_of_Relations.

    Definition reflexive : Prop := forall A RA, @reflexive A RA -> forall x, R RA x x.
    Definition transitive : Prop := forall A RA, @transitive A RA -> forall x, R RA x x.
    Definition transitive : Prop := forall (Ri:Relation_Definitions.relation I) i j k (x:A i) (y:A j) (z:A k), (Ri i j -> Ri j k -> Ri i k) -> R Ri x y -> R Ri y z -> R Ri x z.
    Definition symmetric : Prop := forall (Ri:Relation_Definitions.relation I) i j (x:A i) (y:A j), (Ri i j -> Ri j i) -> R Ri x y -> R Ri y x.
    Definition antisymmetric : Prop := forall (Ri:Relation_Definitions.relation I) i (x y:A i), R Ri x y -> R Ri y x -> x = y.

    (* for compatibility with Equivalence in  ../PROGRAMS/ALG/  *)
    Definition equiv := reflexive /\ transitive /\ symmetric.

  End General_Properties_of_Relations.



  Section Sets_of_Relations.

    Record preorder : Prop :=
      { preord_refl : reflexive; preord_trans : transitive}.

    Record order : Prop :=
      { ord_refl : reflexive;
	ord_trans : transitive;
	ord_antisym : antisymmetric}.

    Record equivalence : Prop :=
      { equiv_refl : reflexive;
	equiv_trans : transitive;
	equiv_sym : symmetric}.

    Record PER : Prop :=  {per_sym : symmetric; per_trans : transitive}.

  End Sets_of_Relations.


  Section Relations_of_Relations.

    Definition inclusion (R1 R2:relation) : Prop :=
      forall Ri i j x y, R1 Ri i j x y -> R2 Ri i j x y.

    Definition same_relation (R1 R2:relation) : Prop :=
      inclusion R1 R2 /\ inclusion R2 R1.
(*
    Definition commut (R1 R2:relation) : Prop :=
      forall i j (x:A i) (y:A j),
	R1 _ _ y x -> forall k (z:A k), R2 _ _ z y ->  exists2 y' : A, R2 y' x & R1 z y'.
*)
  End Relations_of_Relations.

*)
End Relation_Definition.
(*
#[export]
Hint Unfold reflexive transitive antisymmetric symmetric: sets.

#[export]
Hint Resolve Build_preorder Build_order Build_equivalence Build_PER
  preord_refl preord_trans ord_refl ord_trans ord_antisym equiv_refl
  equiv_trans equiv_sym per_sym per_trans: sets.

#[export]
Hint Unfold inclusion same_relation commut: sets.
*)
