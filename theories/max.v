From Coq Require Import Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances.
From NeuralNetInterp.Util Require Nat.
From NeuralNetInterp Require Import max_parameters.
Import Util.Nat.Notations.
Local Open Scope Q_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
(* Should use IEEE 754 floats from flocq, but let's use rationals for now for ease of linearity, proving, etc *)
(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

(** Coq infra *)
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.

Definition Rank := nat.
Bind Scope nat_scope with Rank.
Polymorphic Definition tensor_gen_index_of_rank I : Rank -> Type
  := fix tensor_gen_index_of_rank (s : Rank) : Type
    := match s with
       | O => unit
       | S s => tensor_gen_index_of_rank s * I
       end%type.
Inductive Size : Rank -> Set := snil : Size 0 | snoc {r} (xs : Size r) (x : int) : Size (S r).
(*
Fixpoint Size (r : Rank) : Set
    := match r with
       | O => unit
       | S r' => Size r' * int
       end%type.
*)
Declare Scope size_scope.
Delimit Scope size_scope with size.
Bind Scope size_scope with Size.
(*Definition snil : Size 0 := tt.
Definition snoc {r} (s : Size r) x : Size (S r) := (s, x).*)
Notation "xs ::' x" := (snoc xs x) : size_scope.
Notation "[ ]" := snil : size_scope.
Notation "[ x ]" := (snoc snil x) : size_scope.
Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc snil x) y) .. z) : size_scope.
Definition shd {r : Rank} (s : Size r) : Size (Nat.pred r)
  := match s with
     | snil => snil
     | snoc xs _ => xs
     end.
Definition stl {r : Rank} (s : Size (S r)) : int
  := match s with
     | snoc _ x => x
     end.
Polymorphic Definition tensor_gen_index_of_shape I : forall {r : Rank}, Size r -> Type
  := fix tensor_gen_index_of_shape (r : Rank) : Size r -> Type
    := match r return Size r -> Type with
       | O => fun _ => unit
       | S r' => fun sz => @tensor_gen_index_of_shape r' (shd sz) * I
       end%type.
Fixpoint tensor_index_of_shape {r : Rank} : Size r -> Set
    := match r with
       | O => fun _ => unit
       | S r' => fun sz => @tensor_index_of_shape r' (shd sz) * int
       end%type.
Fixpoint tensor_list_index_of_shape {r : Rank} : Size r -> Set
    := match r with
       | O => fun _ => unit
       | S r' => fun sz => @tensor_list_index_of_shape r' (shd sz) * nat
       end%type.
#[global] Strategy 100 [Rank].
Fixpoint app {r1 r2 : Rank} {struct r2} : Size r1 -> Size r2 -> Size (r1 +' r2)
  := match r2 with
     | 0%nat => fun sz _tt => sz
     | S r2 => fun sz1 sz2 => @app r1 r2 sz1 (shd sz2) ::' stl sz2
     end%size.
Definition scons {r : Rank} x (xs : Size r) : Size _ := app (snoc snil x) xs.
Notation "x :: xs" := (scons x xs) : size_scope.
Notation "s1 ++ s2" := (app s1 s2) : size_scope.
Notation "s1 ++' s2" := (app s1 s2) : size_scope.
Local Open Scope size_scope.

Fixpoint size_map2 {r} (f : int -> int -> int) : Size r -> Size r -> Size r
  := match r with
     | 0%nat => fun _ _ => []
     | S r => fun xs ys => size_map2 f (shd xs) (shd ys) ::' f (stl xs) (stl ys)
     end.

Fixpoint sdroplastn {r : Rank} (n : Rank) : Size r -> Size (r -' n)
  := match n, r with
     | 0%nat, _ => fun xs => xs
     | _, 0%nat => fun _tt => []
     | S n, S r => fun xs => @sdroplastn r n (shd xs)
     end.

Fixpoint slastn {r : Rank} (n : Rank) (xs : Size r) : Size (Nat.min n r)
  := match n, xs with
     | 0%nat, _ => []
     | _, [] => []
     | S n, xs ::' x => slastn n xs ::' x
     end.

Fixpoint shape_ones {r : Rank} : Size r
  := match r with
     | O => []
     | S r => shape_ones ::' 1
     end.
#[global] Arguments shape_ones {r}, r.

Fixpoint tensor_gen_of_rank (list_type : Type -> Type) (A : Type) (r : Rank) : Type
  := match r with
     | 0%nat => A
     | S r => tensor_gen_of_rank list_type (list_type A (* len:=l *)) r
     end.
Definition tensor_gen_of_shape {r} (list_type : Type -> Type) (A : Type) (shape : Size r) : Type
  := tensor_gen_of_rank list_type A r.
Fixpoint empty_tensor_gen_of_rank {list_type A} {r : Rank} {default : pointed A} {default_list_type : forall X, pointed X -> pointed (list_type X)} {struct r} : pointed (tensor_gen_of_rank list_type A r)
  := match r with
     | 0%nat => _
     | S r => empty_tensor_gen_of_rank (r:=r)
     end.
Definition empty_tensor_gen_of_shape {list_type A r} {shape : Size r} {default : pointed A} {default_list_type : forall X, pointed X -> pointed (list_type X)} : pointed (tensor_gen_of_shape list_type A shape)
  := empty_tensor_gen_of_rank.
#[export] Existing Instance empty_tensor_gen_of_rank.
#[export] Existing Instance empty_tensor_gen_of_shape.
Definition tensor_of_rank := @tensor_gen_of_rank (fun A => array A).
Definition tensor_list_of_rank := @tensor_gen_of_rank (fun A => list A).
Definition tensor_of_shape {r} A (s : Size r) := tensor_of_rank A r.
Definition tensor_list_of_shape {r} A (s : Size r) := tensor_list_of_rank A r.
Polymorphic Definition tensor_fun_of_rank I := tensor_gen_of_rank (fun A => I -> A).
Polymorphic Definition tensor_fun_of_shape {r} I A (s : Size r) := tensor_fun_of_rank I A r.
Polymorphic Fixpoint tensor_gen_index_of_shape_make_cps {r B I} {s : Size r} : (tensor_gen_index_of_shape I s -> B) -> tensor_fun_of_shape I B s
  := match s with
     | [] => fun f => f tt
     | s ::' _ => fun k => tensor_gen_index_of_shape_make_cps (s:=s) (fun idxs idx => k (idxs, idx))
     end.

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
Class compute_shape_of {A r} (x : A) := get_shape_of : Size r.
#[global] Hint Extern 0 (compute_shape_of ?x) => let s := get_shape x in exact s : typeclass_instances.

Fixpoint init_map {r A B} (f : A -> B) (s : Size r) : tensor_fun_of_shape int A s -> tensor_of_shape B s
  := match s return tensor_fun_of_shape int A s -> tensor_of_shape B s with
     | [] => f
     | s ::' l => @init_map _ _ (array B) (fun F => PArray.init l (fun i => f (F i))) s
     end.

Definition init {r A} (s : Size r) : tensor_fun_of_shape int A s -> tensor_of_shape A s
  := init_map (fun x => x) s.
Fixpoint with_shape {r A} (s : Size r) : tensor_fun_of_shape int A s -> A
  := match s with
     | [] => fun x => x
     | s ::' l => fun ls => with_shape s ls l
     end.

(*
Structure ndtype := { ndshape : Size ; ty :> Type }.
Definition ndtype_raw
Canonical ndlist_type {A}
Structure Ndlist A := { ndshape : Size ; ndval :> tensor_list_of_shape A ndshape }.
Canonical wrap_list {A} (vals : list (Ndlist A)) := {| ndshape := (List.length vals : int) :: match vals return Size with [] => nil | v :: _ => v.(ndshape) end ; ndval := vals |}.
Structure Tensor {A} := { shape : Size ; numpy :> tensor_array_of_shape
*)
Fixpoint tensor_of_list_map_ {r} {A B} {default : pointed B} {s : Size r} (f : A -> B) {struct s} : tensor_list_of_shape A s -> tensor_of_shape B s
  := match s return tensor_list_of_shape A s -> tensor_of_shape B s with
     | [] => f
     | s ::' _ => tensor_of_list_map_ (s:=s) (array_of_list_map f)
     end.
Definition tensor_of_list_ {r} {A} {default : pointed A} {s : Size r} : tensor_list_of_shape A s -> tensor_of_shape A s
  := tensor_of_list_map_ (fun x => x).
Notation tensor_of_list ls := (@tensor_of_list_ _ _ _ (shape_of ls%list) ls%list) (only parsing).
Notation tensor_of_list_map f ls := (@tensor_of_list_map_ _ _ _ _ (shape_of ls%list) f ls%list) (only parsing).

Fixpoint tensor_make {r} {A} (s : Size r) (x : A) : tensor_of_shape A s
  := match s with
     | [] => x
     | s ::' len => tensor_make s (PArray.make len x)
     end.

Definition ones {r} {A} {one : has_one A} (s : Size r) : tensor_of_shape A s
  := tensor_make s one.
Definition zeros {r} {A} {zero : has_zero A} (s : Size r) : tensor_of_shape A s
  := tensor_make s zero.

Polymorphic Fixpoint tensor_gen_get {r I list_type A s} (getA : forall r' (s' : Size r'), let A := tensor_gen_of_shape list_type A s' in I -> list_type A -> A) {struct s} : tensor_gen_index_of_shape I s -> tensor_gen_of_shape list_type A s -> A
  := match s return tensor_gen_index_of_shape I s -> tensor_gen_of_shape list_type A s -> A with
     | [] => fun dummy x => x
     | s ::' _ => fun idxs t => getA _ [] (snd idxs) (tensor_gen_get (s:=s) (fun r' s' => getA _ (s' ::' 1)) (fst idxs) t)
     end.
Definition tensor_get {r A} {s : Size r} : tensor_index_of_shape s -> tensor_of_shape A s -> A
  := tensor_gen_get (fun _ _ i xs => xs.[i]).
Definition tensor_list_get {r A} {s : Size r} {default : pointed A} : tensor_list_index_of_shape s -> tensor_list_of_shape A s -> A
  := tensor_gen_get (fun _ _ i xs => nth_default point xs i).

Declare Scope tensor_scope.
Delimit Scope tensor_scope with tensor.
Bind Scope tensor_scope with tensor_of_rank.
Bind Scope tensor_scope with tensor_of_shape.
Bind Scope tensor_scope with tensor_list_of_rank.
Bind Scope tensor_scope with tensor_list_of_shape.
Local Open Scope tensor_scope.

Fixpoint tensor_map {r A B} {s : Size r} (f : A -> B) {struct s} : tensor_of_shape A s -> tensor_of_shape B s
  := match s with
     | [] => f
     | s ::' _
       => tensor_map (s:=s) (PArray.map f)
     end.

Definition broadcast_size2 {r} : Size r -> Size r -> Size r := size_map2 max.
Fixpoint tensor_map2 {r} {A B C} (f : A -> B -> C) {struct r} : forall {sA : Size r} {sB : Size r}, tensor_of_shape A sA -> tensor_of_shape B sB -> tensor_of_shape C (broadcast_size2 sA sB)
  := match r with
     | 0%nat => fun _ _ => f
     | S r
       => fun sA sB => tensor_map2 (PArray.broadcast_map2 f) (sA:=shd sA) (sB:=shd sB)
     end.

#[export] Instance tensor_add {r} {sA sB : Size r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A sA) (tensor_of_shape B sB) (tensor_of_shape C (broadcast_size2 sA sB)) := tensor_map2 add.
#[export] Instance tensor_sub {r} {sA sB : Size r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A sA) (tensor_of_shape B sB) (tensor_of_shape C (broadcast_size2 sA sB)) := tensor_map2 sub.
#[export] Instance tensor_mul {r} {sA sB : Size r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A sA) (tensor_of_shape B sB) (tensor_of_shape C (broadcast_size2 sA sB)) := tensor_map2 mul.
#[export] Instance tensor_div_by {r} {sA sB : Size r} {A B C} {div_byAB : has_div_by A B C} : has_div_by (tensor_of_shape A sA) (tensor_of_shape B sB) (tensor_of_shape C (broadcast_size2 sA sB)) := tensor_map2 div.
#[export] Instance tensor_sqrt {r} {s : Size r} {A} {sqrtA : has_sqrt A} : has_sqrt (tensor_of_shape A s) := tensor_map sqrt.
#[export] Instance tensor_opp {r} {s : Size r} {A} {oppA : has_opp A} : has_opp (tensor_of_shape A s) := tensor_map opp.
#[export] Instance add'1 {r} {s : Size r} {a b} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A (s ::' a)) (tensor_of_shape B (s ::' b)) (tensor_of_shape C (s ::' max a b)) | 10 := tensor_add.
#[export] Instance sub'1 {r} {s : Size r} {a b} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A (s ::' a)) (tensor_of_shape B (s ::' b)) (tensor_of_shape C (s ::' max a b)) | 10 := tensor_sub.
#[export] Instance mul'1 {r} {s : Size r} {a b} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A (s ::' a)) (tensor_of_shape B (s ::' b)) (tensor_of_shape C (s ::' max a b)) | 10 := tensor_mul.
#[export] Instance div_by'1 {r} {s : Size r} {a b} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A (s ::' a)) (tensor_of_shape B (s ::' b)) (tensor_of_shape C (s ::' max a b)) | 10 := tensor_div_by.
#[export] Instance add'1s_r {r} {s : Size r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A s) (tensor_of_shape B (shape_ones r)) (tensor_of_shape C s) | 10 := tensor_add.
#[export] Instance add'1s_l {r} {s : Size r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A (shape_ones r)) (tensor_of_shape B s) (tensor_of_shape C s) | 10 := tensor_add.
#[export] Instance sub'1s_r {r} {s : Size r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A s) (tensor_of_shape B (shape_ones r)) (tensor_of_shape C s) | 10 := tensor_sub.
#[export] Instance sub'1s_l {r} {s : Size r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A (shape_ones r)) (tensor_of_shape B s) (tensor_of_shape C s) | 10 := tensor_sub.
#[export] Instance mul'1s_r {r} {s : Size r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A s) (tensor_of_shape B (shape_ones r)) (tensor_of_shape C s) | 10 := tensor_mul.
#[export] Instance mul'1s_l {r} {s : Size r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A (shape_ones r)) (tensor_of_shape B s) (tensor_of_shape C s) | 10 := tensor_mul.
#[export] Instance div_by'1s_r {r} {s : Size r} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A s) (tensor_of_shape B (shape_ones r)) (tensor_of_shape C s) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l {r} {s : Size r} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A (shape_ones r)) (tensor_of_shape B s) (tensor_of_shape C s) | 10 := tensor_div_by.
#[export] Instance add'1s_r'1_same {r} {s : Size r} {a} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A (s ::' a)) (tensor_of_shape B (shape_ones r ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_add.
#[export] Instance add'1s_l'1_same {r} {s : Size r} {a} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A (shape_ones r ::' a)) (tensor_of_shape B (s ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_add.
#[export] Instance sub'1s_r'1_same {r} {s : Size r} {a} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A (s ::' a)) (tensor_of_shape B (shape_ones r ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_sub.
#[export] Instance sub'1s_l'1_same {r} {s : Size r} {a} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A (shape_ones r ::' a)) (tensor_of_shape B (s ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_sub.
#[export] Instance mul'1s_r'1_same {r} {s : Size r} {a} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A (s ::' a)) (tensor_of_shape B (shape_ones r ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_mul.
#[export] Instance mul'1s_l'1_same {r} {s : Size r} {a} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A (shape_ones r ::' a)) (tensor_of_shape B (s ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_mul.
#[export] Instance div_by'1s_r'1_same {r} {s : Size r} {a} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A (s ::' a)) (tensor_of_shape B (shape_ones r ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l'1_same {r} {s : Size r} {a} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A (shape_ones r ::' a)) (tensor_of_shape B (s ::' a)) (tensor_of_shape C (s ::' a)) | 10 := tensor_div_by.
#[export] Instance add'1s_r'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A (s ++' s')) (tensor_of_shape B (shape_ones r ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_add.
#[export] Instance add'1s_l'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {addA : has_add_with A B C} : has_add_with (tensor_of_shape A (shape_ones r ++' s')) (tensor_of_shape B (s ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_add.
#[export] Instance sub'1s_r'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A (s ++' s')) (tensor_of_shape B (shape_ones r ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_sub.
#[export] Instance sub'1s_l'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor_of_shape A (shape_ones r ++' s')) (tensor_of_shape B (s ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_sub.
#[export] Instance mul'1s_r'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A (s ++' s')) (tensor_of_shape B (shape_ones r ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_mul.
#[export] Instance mul'1s_l'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor_of_shape A (shape_ones r ++' s')) (tensor_of_shape B (s ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_mul.
#[export] Instance div_by'1s_r'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A (s ++' s')) (tensor_of_shape B (shape_ones r ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l'1_same_app {r r'} {s : Size r} {s' : Size r'} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor_of_shape A (shape_ones r ++' s')) (tensor_of_shape B (s ++' s')) (tensor_of_shape C (s ++' s')) | 10 := tensor_div_by.

(*
Fixpoint extend_app_nil_l {P : Size -> Type} {s : Size} : P s -> P ([] ++' s)
  := match s with
     | [] => fun x => x
     | s ::' _ => @extend_app_nil_l (fun s => P (s ::' _)) s
     end.
Fixpoint contract_app_nil_l {P : Size -> Type} {s : Size} : P ([] ++' s) -> P s
  := match s with
     | [] => fun x => x
     | s ::' _ => @contract_app_nil_l (fun s => P (s ::' _)) s
     end.
 *)

Fixpoint reshape_app_split_gen {list_type A r1 r2} : tensor_gen_of_rank list_type A (r1 +' r2) -> tensor_gen_of_rank list_type (tensor_gen_of_rank list_type A r2) r1
  := match r2 with
     | 0%nat => fun x => x
     | S r2 => reshape_app_split_gen (r2:=r2)
     end.
Fixpoint reshape_app_combine_gen {list_type A r1 r2} : tensor_gen_of_rank list_type (tensor_gen_of_rank list_type A r2) r1 -> tensor_gen_of_rank list_type A (r1 +' r2)
  := match r2 with
     | 0%nat => fun x => x
     | S r2 => reshape_app_combine_gen (r2:=r2)
     end.
(* infer r1 r2 from the conclusion *)
#[global] Arguments reshape_app_combine_gen list_type A & r1 r2 _.
#[global] Arguments reshape_app_split_gen list_type A & r1 r2 _.
Definition reshape_app_split {A r1 r2 s1 s2} : @tensor_of_shape (r1 +' r2) A (s1 ++' s2) -> tensor_of_shape (tensor_of_shape A s2) s1
  := reshape_app_split_gen.
Definition reshape_app_combine {A r1 r2 s1 s2} : tensor_of_shape (tensor_of_shape A s2) s1 -> @tensor_of_shape (r1 +' r2) A (s1 ++' s2)
  := reshape_app_combine_gen.
(* infer s1 s2 from the conclusion *)
#[global] Arguments reshape_app_combine A & r1 r2 s1 s2 _.
#[global] Arguments reshape_app_split A & r1 r2 s1 s2 _.
(*
Require Import Program . Obligation Tactic := cbn; intros.
Fixpoint broadcast_map_ {A B} {s1 s2 : Size} {keepdim : with_default bool false} (f : A -> tensor_of_shape B s2) {struct s1} : tensor_of_shape A s1 -> tensor_of_shape (tensor_of_shape B (s1 ++' (if keepdim then [1] else []) ++' s2) s1.
refine match s1, keepdim return tensor_of_shape A s1 -> tensor_of_shape B (s1 ++' (if keepdim then [1] else []) ++' s2) with
     | [], true => fun x => reshape_app_combine (s1:=[1]) (PArray.make 1 (f x))
     | [], false => fun x => reshape_app_combine (s1:=[]) (f x)
     | s1 ::' _, keepdim
       => fun x => _ (*(broadcast_map (keepdim:=keepdim) (s1:=s1)) (* _(*PArray.map f*))*)*)
       end; cbn in *.
epose (@broadcast_map _ _ s1 _ keepdim _ x).
epose (@broadcast_map _ _ s1 _ keepdim (fun a => reshape_app_combine (s1:=[1])).
Next Obligation.
  pose (
 pose (broa

Fixpoint extended_broadcast_map {A B} {s1 s1' s2 : Size} (f : tensor_of_shape A s1' -> tensor_of_shape B s2) {struct s1} : tensor_of_shape A (s1 ++ s1') -> tensor_of_shape B (s1 ++ s2)
  := match s1 with
     | [] => f
     | s :: s1
       => PArray.map (extended_broadcast_map f)
     end.
 *)

(*
Definition broadcast_m1 {A s} n : tensor_of_shape A s -> tensor_of_shape A (s ::' n)
  := tensor_map (PArray.make n).
Definition broadcast_0 {A s} n : tensor_of_shape A s -> tensor_of_shape A ([n] ++' s)
  := fun x => reshape_app_combine (PArray.make n x).
#[global] Arguments broadcast_m1 A & s n _.
#[global] Arguments broadcast_0 A & s n _.
Definition slice_none_m1 {A s} : tensor_of_shape A s -> tensor_of_shape A (s ::' 1)
  := broadcast_m1 1.
Definition slice_none_0 {A s} : tensor_of_shape A s -> tensor_of_shape A ([1] ++' s)
  := broadcast_0 1.
*)
Fixpoint repeat {r A} (x : A) (s : Size r) : tensor_of_shape A s
  := match s with
     | [] => x
     | ss ::' s => repeat (PArray.repeat x s) ss
     end.

Fixpoint broadcast' {A} (x : A) {r : Rank} : tensor_of_shape A (shape_ones r)
  := match r with
     | O => x
     | S r => broadcast' (PArray.repeat x 1) (r:=r)
     end.
Definition broadcast {r A} {s : Size r} (x : tensor_of_shape A s) {r' : Rank} : tensor_of_shape A (shape_ones r' ++' s)
  := reshape_app_combine (broadcast' x).

Definition keepdim_gen {r} {s : Size r} {A B} (f : A -> tensor_of_shape B s) : A -> tensor_of_shape B ([1] ++' s)
  := fun a => reshape_app_combine (PArray.make 1 (f a)).
Definition keepdim {A B} (f : A -> B) : A -> tensor_of_shape B [1] := keepdim_gen (s:=[]) f.
#[local] Notation keepdimf := keepdim (only parsing).

Definition reduce_axis_m1' {r A B} {s1 : Size r} {s2} (reduction : array A -> B) : tensor_of_shape A (s1 ::' s2) -> tensor_of_shape B s1
  := tensor_map reduction.

Definition reduce_axis_m1 {r A B} {s1 : Size r} {s2} {keepdim : with_default "keepdim" bool false} (reduction : array A -> B)
  : tensor_of_shape A (s1 ::' s2) -> tensor_of_shape B (s1 ++' if keepdim return Size (if keepdim then _ else _) then [1] else [])
  := fun t
     => let keepdimf :=
          if keepdim return
               (array A -> tensor_of_shape B (if keepdim return Size (if keepdim then _ else _) then [1] else []))
          then keepdimf reduction
          else reduction in
        reshape_app_combine (reduce_axis_m1' keepdimf t).

Definition to_bool {A} {zero : has_zero A} {eqb : has_eqb A} {r} {s : Size r} (xs : tensor_of_shape A s) : tensor_of_shape bool s
  := tensor_map (fun x => x ≠? 0)%core xs.

Definition tril {A} {zero : has_zero A} {rnk} {s : Size rnk} {r c}
  {diagonal : with_default "diagonal" int 0%int63} (input : tensor_of_shape A (s ++' [r; c]))
  : tensor_of_shape A (s ++' [r; c])
  := reshape_app_combine (tensor_map (PArray.tril (diagonal:=diagonal)) input).
#[global] Arguments tril {A%type_scope zero rnk%nat s%size} {r c}%uint63 {diagonal}%sint63 input%tensor.
Definition triu {A} {zero : has_zero A} {rnk} {s : Size rnk} {r c}
  {diagonal : with_default "diagonal" int 0%int63} (input : tensor_of_shape A (s ++' [r; c]))
  : tensor_of_shape A (s ++' [r; c])
  := reshape_app_combine (tensor_map (PArray.triu (diagonal:=diagonal)) input).
#[global] Arguments triu {A%type_scope zero rnk%nat s%size} {r c}%uint63 {diagonal}%sint63 input%tensor.

(** Hyperparameters *)
Definition N_LAYERS : nat := 1.
Definition N_HEADS : nat := 1.
Definition D_MODEL : nat := 32.
Definition D_HEAD : nat := 32.
(*Definition D_MLP = None*)

Definition D_VOCAB : nat := 64.

Definition W_E : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.W_E.
Definition W_pos : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.W_pos.
Definition L0_attn_W_Q : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_Q.
Definition L0_attn_W_K : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_K.
Definition L0_attn_W_V : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_V.
Definition L0_attn_W_O : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_O.
Definition L0_attn_b_Q : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_Q.
Definition L0_attn_b_K : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_K.
Definition L0_attn_b_V : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_V.
Definition L0_attn_b_O : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_O.
Definition L0_ln1_b : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_ln1_b.
Definition L0_ln1_w : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.L0_ln1_w.
Definition ln_final_b : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.ln_final_b.
Definition ln_final_w : tensor_of_shape _ _ := Eval cbv in tensor_of_list max_parameters.ln_final_w.


Definition embed {r} {s : Size r} (tokens : tensor_of_shape int s) : tensor_of_shape Q (s ::' stl (shape_of W_E))
  := tensor_map (fun i => W_E.[i]) tokens.

Definition pos_embed {r} {s : Size (S r)} (tokens : tensor_of_shape int s)
  (tokens_length := stl s) (* s[-1] *)
  (batch := sdroplastn 1 s) (* s[:-1] *)
  (d_model := stl (shape_of W_pos)) (* s[-1] *)
  : tensor_of_shape Q (batch ++' [tokens_length] ::' d_model)
  := repeat (W_pos.[[:tokens_length]]) batch.

Section layernorm.
  Context {r A} {s : Size r} {d_model}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A}
    (eps : A) (w b : tensor_of_shape A [d_model]).

  Definition layernorm_linpart (x : tensor_of_shape A (s ::' d_model))
    : tensor_of_shape A (s ::' d_model)
    := (x - reduce_axis_m1 (keepdim:=true) mean x)%core.

  Definition layernorm_scale (x : tensor_of_shape A (s ::' d_model))
    : tensor_of_shape A (s ::' 1)
    := (√(reduce_axis_m1 (keepdim:=true) mean (x ²) + broadcast' eps))%core.

  Definition layernorm_rescale (x : tensor_of_shape A (s ::' d_model))
                               (scale : tensor_of_shape A (s ::' 1))
    : tensor_of_shape A (s ::' d_model)
    := (x / scale)%core.

  Definition layernorm_postrescale (x : tensor_of_shape A (s ::' d_model))
    : tensor_of_shape A (s ::' d_model)
    := (x * broadcast w + broadcast b)%core.

  Definition layernorm (x : tensor_of_shape A (s ::' d_model))
    : tensor_of_shape A (s ::' d_model)
    := let x := layernorm_linpart x in
       let scale := layernorm_scale x in
       let x := layernorm_rescale x scale in
       layernorm_postrescale x.
End layernorm.

Section ln.
  Context {r} {s : Size r}
    (d_model := Uint63.of_Z cfg.d_model)
    (eps := cfg.eps).

  Section ln1.
    Context (w := L0_ln1_w) (b := L0_ln1_b).

    Definition ln1_linpart (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_linpart x.
    Definition ln1_scale (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln1_rescale (x : tensor_of_shape Q (s ::' d_model)) (scale : tensor_of_shape Q (s ::' 1)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln1_postrescale (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln1 (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm eps w b x.
  End ln1.

  Section ln_final.
    Context (w := ln_final_w) (b := ln_final_b).

    Definition ln_final_linpart (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_linpart x.
    Definition ln_final_scale (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln_final_rescale (x : tensor_of_shape Q (s ::' d_model)) (scale : tensor_of_shape Q (s ::' 1)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln_final_postrescale (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln_final (x : tensor_of_shape Q (s ::' d_model)) : tensor_of_shape Q (s ::' d_model)
      := layernorm eps w b x.
  End ln_final.
End ln.

Section Attention.
  Context {A r} {batch : Size r}
    {sqrtA : has_sqrt A} {coerZ : has_coer Z A} {addA : has_add A} {zeroA : has_zero A} {mulA : has_mul A} {divA : has_div A}
    {pos n_heads d_model d_head} {n_ctx:N}
    {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
    (W_Q W_K W_V W_O : tensor_of_shape A [n_heads; d_model; d_head])
    (b_Q b_K b_V : tensor_of_shape A [n_heads; d_head])
    (b_O : tensor_of_shape A [d_model])
    (IGNORE : A)
    (n_ctx' : int := Uint63.of_Z n_ctx)
    (attn_scale : A := √(coer (Uint63.to_Z d_head)))
    (maybe_n_heads := fun b : bool => if b return Size (if b then _ else _) then [n_heads] else [])
    (query_input key_input value_input : tensor_of_shape A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
    (mask : tensor_of_shape bool [n_ctx'; n_ctx'] := to_bool (tril (A:=bool) (ones [n_ctx'; n_ctx']))).

  (*         if self.cfg.use_split_qkv_input:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        q = self.hook_q(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]*)
  Definition einsum_input
    (input : tensor_of_shape A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
    (W : tensor_of_shape A [n_heads; d_model; d_head])
    : tensor_of_shape A ((batch ::' pos) ++' [n_heads; d_head])
    := (let input : tensor_of_shape (tensor_of_shape A _) (batch ::' pos) := reshape_app_split input in
        let W : tensor_of_shape (tensor_of_shape (tensor_of_shape A [d_head]) [d_model]) [n_heads] := W in
        let f (input : tensor_of_shape A [d_model]) (W : tensor_of_shape (tensor_of_shape A [d_head]) [d_model]) : tensor_of_shape A [d_head]
          := let input_W : tensor_of_shape (tensor_of_shape A [d_head]) [d_model] := tensor_map2 (fun i w => broadcast' i * w)%core input W in
             reduce_axis_m1 (keepdim:=false) PArray.sum (input_W:tensor_of_shape A [d_model; d_head]) in
        tensor_map
          (if use_split_qkv_input return tensor_of_shape (tensor_of_shape A [d_model]) (maybe_n_heads use_split_qkv_input) -> tensor_of_shape (tensor_of_shape A [d_head]) [n_heads]
           then fun input
                => tensor_map2 f input W
           else fun input : tensor_of_shape A [d_model]
                => tensor_map (f input) W)
          input).

  Definition q : tensor_of_shape A ((batch ::' pos) ++' [n_heads; d_head])
    := (einsum_input query_input W_Q + broadcast b_Q)%core.
  Definition k : tensor_of_shape A ((batch ::' pos) ++' [n_heads; d_head])
    := (einsum_input query_input W_K + broadcast b_K)%core.
  Definition v : tensor_of_shape A ((batch ::' pos) ++' [n_heads; d_head])
    := (einsum_input query_input W_V + broadcast b_V)%core.

  Definition attn_scores : tensor_of_shape A (batch ::' n_heads ::' pos ::' pos).
    refine (let q : tensor_of_shape (tensor_of_shape A [pos; n_heads; d_head]) batch := q in
            let k : tensor_of_shape (tensor_of_shape A [pos; n_heads; d_head]) batch := k in
            let qk : tensor_of_shape (tensor_of_shape A [n_heads; pos; pos]) batch
              := tensor_map2
                   _
                   q
                   k in
            let qk : tensor_of_shape A (batch ::' n_heads ::' pos ::' pos) := qk in
            qk / broadcast' attn_scale)%core.
    (*
    From Ltac2 Require Ltac2.
    Notation "'weaksauce_einsum' {{ i1 .. in , j1 .. jn -> k1 .. kn }} t1 t2"
      := (match _ as s1, _ as s2, _ as s3 return tensor_of_shape _ s3 with
          | s1, s2, s3
            => match t1 : tensor_of_shape _ s1, t2 : tensor_of_shape _ s2 return tensor_of_shape _ s3 with
               | t1', t2'
                 =>
               end
          end)
           (only parsing).
    refine match _ as s1, _ as s2, _ as s3 return tensor_of_shape _ s1 -> tensor_of_shape _ s2 -> tensor_of_shape _ s3 with
           | s1, s2, s3 => _
           end.
    epose (init
             [_; _; _]
             (fun head_index query_pos key_pos
              => _)).
   attn_scores = (
            einsum(
                "batch query_pos head_index d_head, \
                    batch key_pos head_index d_head \
                    -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale

    refine (_).

Definition attn
             {r} {batch : Size r} {pos
     *)
  Abort.
End Attention.
Eval cbv in embed (tensor_of_list [0; 1]%uint63).
Eval cbv in pos_embed (tensor_of_list [[0; 1]]%uint63).
