From Coq Require Import Uint63 QArith Lia List PArray.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances.
From NeuralNetInterp Require Import max_parameters.
Local Open Scope Q_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
(* Should use IEEE 754 floats from flocq, but let's use rationals for now for ease of linearity, proving, etc *)
(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

(** Coq infra *)
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.

Inductive Size := snil | snoc (_ : Size) (_ : int).
Fixpoint app (xs ys : Size) : Size
  := match ys with
     | snil => xs
     | snoc ys y => snoc (app xs ys) y
     end.
Definition scons x (xs : Size) : Size := app (snoc snil x) xs.
Declare Scope size_scope.
Delimit Scope size_scope with size.
Bind Scope size_scope with Size.
Notation "x :: xs" := (scons x xs) : size_scope.
Notation "xs ::' x" := (snoc xs x) : size_scope.
Notation "[ ]" := snil : size_scope.
Notation "[ x ]" := (snoc snil x) : size_scope.
Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc snil x) y) .. z) : size_scope.
Notation "s1 ++ s2" := (app s1 s2) : size_scope.
Notation "s1 ++' s2" := (app s1 s2) : size_scope.
Undelimit Scope size_scope.
Local Open Scope size_scope.

Fixpoint tensor_gen_of_shape (list_type : Type -> Type) (A : Type) (shape : Size) : Type
  := match shape with
     | [] => A
     | shape ::' l => tensor_gen_of_shape list_type (list_type A (* len:=l *)) shape
     end.
Fixpoint empty_tensor_gen_of_shape {list_type A} {shape : Size} {default : pointed A} {default_list_type : forall X, pointed X -> pointed (list_type X)} {struct shape} : pointed (tensor_gen_of_shape list_type A shape)
  := match shape with
     | [] => _
     | shape ::' l => empty_tensor_gen_of_shape (shape:=shape)
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
    => let len := uconstr:(Uint63.of_Z (Z.of_N (N.of_nat (List.length val)))) in
       let rest := lazymatch (eval hnf in val) with
                   | cons ?val _ => get_shape val
                   | ?val => fail 1 "Could not find cons in" val
                   end in
       (eval cbv in (scons len rest))
  | array ?x
    => let len := uconstr:(PArray.length val) in
       let rest := let val := (eval cbv in (PArray.get val 0)) in
                   get_shape val in
       (eval cbv in (scons len rest))
  | _ => constr:(snil)
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
Fixpoint tensor_of_list_map_ {A B} {default : pointed B} {s : Size} (f : A -> B) {struct s} : tensor_list_of_shape A s -> tensor_of_shape B s
  := match s return tensor_list_of_shape A s -> tensor_of_shape B s with
     | [] => f
     | s ::' _ => tensor_of_list_map_ (s:=s) (array_of_list_map f)
     end.
Definition tensor_of_list_ {A} {default : pointed A} {s : Size} : tensor_list_of_shape A s -> tensor_of_shape A s
  := tensor_of_list_map_ (fun x => x).
Notation tensor_of_list ls := (@tensor_of_list_ _ _ (shape_of ls) ls) (only parsing).
Notation tensor_of_list_map f ls := (@tensor_of_list_map_ _ _ _ (shape_of ls) f ls) (only parsing).

(** Hyperparameters *)
Definition N_LAYERS : nat := 1.
Definition N_HEADS : nat := 1.
Definition D_MODEL : nat := 32.
Definition D_HEAD : nat := 32.
(*Definition D_MLP = None*)

Definition D_VOCAB : nat := 64.

Definition W_E : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.W_E.
Definition W_pos : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.W_pos.

Declare Scope tensor_scope.
Delimit Scope tensor_scope with tensor.
Local Open Scope tensor_scope.

Fixpoint tensor_map {A B} {s : Size} (f : A -> B) {struct s} : tensor_of_shape A s -> tensor_of_shape B s
  := match s with
     | [] => f
     | s ::' _
       => tensor_map (s:=s) (PArray.map f)
     end.

Fixpoint tensor_map2 {A B C} {s : Size} (f : A -> B -> C) {struct s} : tensor_of_shape A s -> tensor_of_shape B s -> tensor_of_shape C s
  := match s with
     | [] => f
     | s ::' _
       => tensor_map2 (PArray.broadcast_map2 f)
     end.

#[export] Instance add {s A} {addA : has_add A} : has_add (tensor_of_shape A s) := tensor_map2 add.
#[export] Instance sub {s A} {subA : has_sub A} : has_sub (tensor_of_shape A s) := tensor_map2 sub.
#[export] Instance mul {s A} {mulA : has_mul A} : has_mul (tensor_of_shape A s) := tensor_map2 mul.
#[export] Instance div_by {s A B} {div_byAB : has_div_by A B} : has_div_by (tensor_of_shape A s) (tensor_of_shape B s) := tensor_map2 div.

Fixpoint broadcast_map {A B} {s1 s2 : Size} {keepdim : with_default bool false} (f : A -> tensor_of_shape B s2) {struct s1} : tensor_of_shape A s1 -> tensor_of_shape B (s1 ++' (if keepdim then [1] else []) ++' s2)
  := match s1, keepdim return tensor_of_shape A s1 -> tensor_of_shape B (s1 ++' (if keepdim then [1] else []) ++' s2) with
     | [], true => fun x => PArray.make 1 (f x)
     | [], false => f
     | s1 ::' _, keepdim
       => broadcast_map (s1:=s1) (PArray.map f)
     end.

Fixpoint extended_broadcast_map {A B} {s1 s1' s2 : Size} (f : tensor_of_shape A s1' -> tensor_of_shape B s2) {struct s1} : tensor_of_shape A (s1 ++ s1') -> tensor_of_shape B (s1 ++ s2)
  := match s1 with
     | [] => f
     | s :: s1
       => PArray.map (extended_broadcast_map f)
     end.

Definition slice_none_m1 {A s} : tensor_of_shape A s -> tensor_of_shape A (s ::' 1)
  := broadcast_map (s2:=[1]) (PArray.make 1).

Definition keepdim_gen {A B} {s : with_default Size nil} (f : A -> tensor_of_shape B s) : A -> tensor_of_shape B (1 :: s)
  := fun a => PArray.make 1 (f a).
Definition keepdim {A B} (f : A -> B) : A -> tensor_of_shape B [1] := keepdim_gen f.

Fixpoint reduce_axis_m1 {A B s1 s2} (reduction : array A -> B) : tensor_of_shape A (s1 ::' s2) -> tensor_of_shape B s1
  := match s1 with
     | [] => reduction
     | _ :: s1 => PArray.map (reduce_axis_m1 (s1:=s1) reduction)
     end.

Definition embed {s : Size} (tokens : tensor_of_shape int s) : tensor_of_shape Q (s ++' tl (shape_of W_E))
  := broadcast_map (s2:=tl (shape_of W_E)) (fun i => W_E.[i]) tokens.

Definition pos_embed {s : Size} (tokens : tensor_of_shape int s)
  (tokens_length := (hd 1 (rev s))%uint63)
  (batch := (hd 1 s)%uint63)
  (d_model := tl (shape_of W_pos))
  : tensor_of_shape Q ([batch;tokens_length] ++' d_model)
  := PArray.repeat (W_pos.[[:tokens_length]]) batch.

Definition layernorm {A s} {d_model} (x : tensor_of_shape A (s ::' d_model)) (eps : A) (w b : tensor_of_shape A [d_model]) : tensor_of_shape A (s ::' d_model)
  := (let x (* s ::' d_model *) := x - reduce_axis_m1 (keepdim mean) x in
      let scale := reduce_axis_m1 (keepdim mean) (x ²) + eps in
      I)%core.





Eval cbv in embed (tensor_of_list [0; 1]%uint63).
Eval cbv in pos_embed (tensor_of_list [[0; 1]]%uint63).
  cbn.

Fixpoint map_at_bottom


(*Record Transformer :=
  { W_E *)
*)
