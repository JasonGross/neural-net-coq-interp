From Coq Require Import Uint63 QArith Lia List PArray.
From NeuralNetInterp.Util Require Import Default Pointed PArray.
From NeuralNetInterp Require Import max_parameters.
Local Open Scope Q_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
(* Should use IEEE 754 floats from flocq, but let's use rationals for now for ease of linearity, proving, etc *)
(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

(** Coq infra *)
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.
#[local] Coercion N.of_nat : nat >-> N.

Definition Size := list int.

Fixpoint tensor_gen_of_shape (list_type : Type -> Type) (A : Type) (shape : Size) : Type
  := match shape with
     | [] => A
     | l :: shape => list_type (* len l *) (tensor_gen_of_shape list_type A shape)
     end.
Fixpoint empty_tensor_gen_of_shape {list_type A} {shape : Size} {default : pointed A} {default_list_type : forall X, pointed X -> pointed (list_type X)} : pointed (tensor_gen_of_shape list_type A shape)
  := match shape with
     | [] => _
     | l :: shape => _
     end.
#[export] Existing Instance empty_tensor_gen_of_shape.
Definition tensor_of_shape := tensor_gen_of_shape (fun A => array A).
Definition tensor_list_of_shape := tensor_gen_of_shape (fun A => list A).
Ltac get_shape val :=
  lazymatch type of val with
  | tensor_gen_of_shape _ _ ?shape => shape
  | tensor_of_shape _ ?shape => shape
  | tensor_list_of_shape _ ?shape => shape
  | list ?x
    => let len := (eval cbv in (Uint63.of_Z (Z.of_N (N.of_nat (List.length val))))) in
       let rest := lazymatch (eval hnf in val) with
                   | ?val :: _ => get_shape val
                   | ?val => fail 1 "Could not find cons in" val
                   end in
       constr:(len :: rest)
  | array ?x
    => let len := (eval cbv in (PArray.length val)) in
       let rest := let val := (eval cbv in (PArray.get val 0)) in
                   get_shape val in
       constr:(len :: rest)
  | _ => constr:(@nil int)
  end.
Notation shape_of x := (match x return _ with y => ltac:(let s := get_shape y in exact s) end) (only parsing).
Class compute_shape_of {A} (x : A) := get_shape_of : Size.
#[global] Hint Extern 0 (compute_shape_of ?x) => let s := get_shape x in exact s : typeclass_instances.

(*
Structure ndtype := { ndshape : Size ; ty :> Type }.
Definition ndtype_raw
Canonical ndlist_type {A}
Structure Ndlist A := { ndshape : Size ; ndval :> tensor_list_of_shape A ndshape }.
Canonical wrap_list {A} (vals : list (Ndlist A)) := {| ndshape := (List.length vals : int) :: match vals return Size with [] => nil | v :: _ => v.(ndshape) end ; ndval := vals |}.
Structure Tensor {A} := { shape : Size ; numpy :> tensor_array_of_shape
*)
Fixpoint tensor_of_list_ {A} {default : pointed A} {s : Size} : tensor_list_of_shape A s -> tensor_of_shape A s
  := match s with
     | [] => fun t => t
     | _ :: s => fun ls => array_of_list (List.map (@tensor_of_list_ A _ s) ls)
     end.

Notation tensor_of_list ls := (@tensor_of_list_ _ _ (shape_of ls) ls) (only parsing).

(** Hyperparameters *)
Definition N_LAYERS : nat := 1.
Definition N_HEADS : nat := 1.
Definition D_MODEL : nat := 32.
Definition D_HEAD : nat := 32.
(*Definition D_MLP = None*)

Definition D_VOCAB : nat := 64.

Definition W_E : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.W_E.
Definition W_pos : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.W_pos.

Fixpoint broadcast_map {A B} {s1 s2 : Size} (f : A -> tensor_of_shape B s2) {struct s1} : tensor_of_shape A s1 -> tensor_of_shape B (s1 ++ s2)
  := match s1 with
     | [] => f
     | s :: s1
       => PArray.map (broadcast_map f)
     end.

Definition embed {s : Size} (vec : tensor_of_shape int s) : tensor_of_shape Q (s ++ tl (shape_of W_E))
  := broadcast_map (s2:=tl (shape_of W_E)) (fun i => W_E.[i]) vec.
(*
Eval cbv in embed (tensor_of_list [0; 1]%uint63).
  cbn.

Fixpoint map_at_bottom


(*Record Transformer :=
  { W_E *)
*)
