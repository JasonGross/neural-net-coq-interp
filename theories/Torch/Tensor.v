From Coq.Structures Require Import Equalities.
From Coq Require Import ZArith Sint63 Uint63 List PArray Lia.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool (*PrimitiveProd*).
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Util Require Import PArray.Proofs List.Proofs.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Local Open Scope list_scope.
Set Implicit Arguments.
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Import ListNotations.
(*Import PrimitiveProd.Primitive.*)

Definition Rank := nat.
#[global] Bind Scope nat_scope with Rank.
#[local] Coercion is_true : bool >-> Sortclass.

Module Type IndexType.
  Parameter t : Type.
  Notation IndexType := t.
End IndexType.

Module Type ExtendedIndexType.
  Include IndexType.
  Parameter one : has_one t.
  Parameter leb : has_leb t.
  Parameter ltb : has_ltb t.
  Parameter eqb : has_eqb t.
  #[export] Existing Instances eqb.
  #[export] Existing Instances one ltb leb.
End ExtendedIndexType.

Module IndexGen.
  Module Make (IndexType : IndexType).
    Import (hints) IndexType.
    Notation IndexType := IndexType.t.

    Fixpoint t (r : Rank) : Type
      := match r with
         | O => unit
         | S r => t r * IndexType.t
         end.
    Notation Index := t.

    Definition nil : t 0 := tt.
    Definition snoc {r} (s : t r) x : t (S r) := (s, x).
    Module Import IndexNotations0.
      Declare Scope index_scope.
      Delimit Scope index_scope with index.
      Bind Scope index_scope with Index.
      Notation "xs ::' x" := (snoc xs x) : index_scope.
      Notation "[ ]" := nil : index_scope.
      Notation "[ x ]" := (snoc nil x) : index_scope.
      Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc nil x) y) .. z) : index_scope.
    End IndexNotations0.
    Module IndexPatternNotations.
      Declare Scope index_pattern_scope.
      Delimit Scope index_pattern_scope with index_pattern.
      Notation "xs ::' x" := (pair xs x) : index_pattern_scope.
      Notation "[ ]" := tt : index_pattern_scope.
      Notation "[ x ]" := (pair tt x) : index_pattern_scope.
      Notation "[ x ; y ; .. ; z ]" :=  (pair .. (pair (pair tt x) y) .. z) : index_pattern_scope.
    End IndexPatternNotations.
    #[local] Open Scope index_scope.
    Definition hd {r : Rank} : Index (S r) -> Index r := @fst _ _.
    Definition tl {r : Rank} : Index (S r) -> IndexType := @snd _ _.
    Fixpoint app {r1 r2 : Rank} {struct r2} : Index r1 -> Index r2 -> Index (r1 +' r2)
      := match r2 with
         | 0%nat => fun sz _tt => sz
         | S r2 => fun sz1 sz2 => @app r1 r2 sz1 (hd sz2) ::' tl sz2
         end%index.
    Definition cons {r : Rank} x (xs : Index r) : Index _ := app [x] xs.
    Module Export IndexNotations1.
      Include IndexNotations0.
      Notation "x :: xs" := (cons x xs) : index_scope.
      Notation "s1 ++ s2" := (app s1 s2) : index_scope.
      Notation "s1 ++' s2" := (app s1 s2) : index_scope.
    End IndexNotations1.

    Section repeat.
      Context (x : IndexType).
      Fixpoint repeat (r : Rank) : Index r
        := match r with
           | O => []
           | S r => repeat r ::' x
           end.

      Lemma hd_repeat {r} : hd (repeat (S r)) = repeat r.
      Proof using Type. reflexivity. Qed.

      Lemma tl_repeat {r} : tl (repeat (S r)) = x.
      Proof using Type. reflexivity. Qed.
    End repeat.

    Definition item : Index 1 -> IndexType := tl.

    Fixpoint map {r} (f : IndexType -> IndexType) : Index r -> Index r
      := match r with
         | 0%nat => fun _ => []
         | S r => fun xs => map f (hd xs) ::' f (tl xs)
         end.

    Fixpoint map2 {r} (f : IndexType -> IndexType -> IndexType) : Index r -> Index r -> Index r
      := match r with
         | 0%nat => fun _ _ => []
         | S r => fun xs ys => map2 f (hd xs) (hd ys) ::' f (tl xs) (tl ys)
         end.

    (* TODO: nary *)
    Fixpoint map3 {r} (f : IndexType -> IndexType -> IndexType -> IndexType) : Index r -> Index r -> Index r -> Index r
      := match r with
         | 0%nat => fun _ _ _ => []
         | S r => fun xs ys zs => map3 f (hd xs) (hd ys) (hd zs) ::' f (tl xs) (tl ys) (tl zs)
         end.

    Fixpoint fold_map {A B r} (f : IndexType -> A) (accum : B -> A -> B) (init : B) : Index r -> B
      := match r with
         | 0%nat => fun _ => init
         | S r => fun xs => fold_map f accum (accum init (f (tl xs))) (hd xs)
         end.

    Fixpoint fold_map2 {A B r} (f : IndexType -> IndexType -> A) (accum : B -> A -> B) (init : B) : Index r -> Index r -> B
      := match r with
         | 0%nat => fun _ _ => init
         | S r => fun xs ys => fold_map2 f accum (accum init (f (tl xs) (tl ys))) (hd xs) (hd ys)
         end.

    Fixpoint curriedT_dep {r : Rank} : (Index r -> Type) -> Type
      := match r with
         | O => fun f => f []
         | S r => fun f => curriedT_dep (fun init => forall i, f (init ::' i))
         end.
    Definition curriedT {r} (T : Type) : Type := @curriedT_dep r (fun _ => T).

    Fixpoint curry_dep {r} : forall {T}, (forall i : Index r, T i) -> @curriedT_dep r T
      := match r return forall {T}, (forall i : Index r, T i) -> @curriedT_dep r T with
         | O => fun T f => f []
         | S r => fun T f => @curry_dep r _ (fun rest i => f (rest ::' i))
         end.
    Definition curry {r T} : (Index r -> T) -> @curriedT r T
      := @curry_dep r (fun _ => T).
    Fixpoint uncurry_map_dep {r} : forall {A B}, (forall i, A i -> B i) -> @curriedT_dep r A -> (forall i : Index r, B i)
      := match r return forall {A B}, (forall i, A i -> B i) -> @curriedT_dep r A -> (forall i : Index r, B i) with
         | O => fun A B F f 'tt => F _ f
         | S r => fun A B F f '(rest, i)
                  => @uncurry_map_dep
                       r (fun rest => forall i, A (rest ::' i)) (fun rest => B (rest ::' i))
                       (fun rest f => F _ (f _))
                       f rest
         end.
    Definition uncurry_dep {r} {T} : @curriedT_dep r T -> (forall i : Index r, T i)
      := @uncurry_map_dep r T T (fun _ x => x).
    Definition uncurry {r T} : @curriedT r T -> (Index r -> T)
      := uncurry_dep.

    Fixpoint split_radd {r1 r2} {struct r2} : Index (r1 +' r2) -> Index r1 * Index r2
      := match r2 with
         | 0%nat => fun idx => (idx, tt)
         | S r2
           => fun '(idx1, idx2)
              => let '(idx11, idx12) := @split_radd r1 r2 idx1 in
                 (idx11, (idx12, idx2))
         end.
    Fixpoint combine_radd {r1 r2} {struct r2} : Index r1 * Index r2 -> Index (r1 +' r2)
      := match r2 return Index r1 * Index r2 -> Index (r1 +' r2) with
         | 0%nat => fun '(idx, tt) => idx
         | S r2
           => fun '(idx1, (idx2, idx3))
              => (@combine_radd r1 r2 (idx1, idx2), idx3)
         end.

    Definition curry_radd {A r1 r2} : (Index (r1 +' r2) -> A) -> (Index r1 -> Index r2 -> A)
      := fun f i1 i2 => f (combine_radd (i1, i2)).
    Definition uncurry_radd {A r1 r2} : (Index r1 -> Index r2 -> A) -> (Index (r1 +' r2) -> A)
      := fun f i => let '(i1, i2) := split_radd i in f i1 i2.
    Definition curry_S {A r} : (Index (1 +' r) -> A) -> (Index 1 -> Index r -> A)
      := curry_radd.
    Definition uncurry_S {A r} : (Index 1 -> Index r -> A) -> (Index (1 +' r) -> A)
      := uncurry_radd.

    Module UncurryNotation.
      Notation "'uncurry_fun' x1 .. xn => body"
        := (match _ return _ with
            | ty => uncurry_S (fun x1 => .. (uncurry_S (fun xn => match body return Index 0 -> ty with v => fun 'tt => v end)) .. )
            end)
             (only parsing, at level 200, x1 binder, xn binder, body at level 200).
    End UncurryNotation.

    Module Export IndexNotations.
      Include IndexNotations1.
      (*Include UncurryNotation.*)
    End IndexNotations.

    Module UncurryCoercions.
      Coercion uncurry_dep : curriedT_dep >-> Funclass.
      Coercion uncurry : curriedT >-> Funclass.
    End UncurryCoercions.

    Fixpoint droplastn {r : Rank} (n : Rank) : Index r -> Index (r -' n)
      := match n, r with
         | 0%nat, _ => fun xs => xs
         | _, 0%nat => fun _tt => []
         | S n, S r => fun xs => @droplastn r n (hd xs)
         end.

    Fixpoint lastn {r : Rank} (n : Rank) : Index r -> Index (Nat.min n r)
      := match n, r return Index r -> Index (Nat.min n r) with
         | 0%nat, _ => fun _ => []
         | _, 0%nat => fun _ => []
         | S n, S r => fun xs => lastn n (hd xs) ::' (tl xs)
         end.
  End Make.

  Module Type MakeSig (IndexType : IndexType) := Nop <+ Make IndexType.

  Module ExtendedMake (IndexType : ExtendedIndexType).
    Import (hints) IndexType.
    Include Make IndexType.

    Definition ones {r} := repeat 1%core r.

    #[export] Instance eqb {r : Rank} : has_eqb (Index r)
      := fold_map2 IndexType.eqb andb true.
    #[export] Instance leb {r : Rank} : has_leb (Index r)
      := fold_map2 IndexType.leb andb true.
    #[export] Instance ltb {r : Rank} : has_ltb (Index r)
      := match r with O => fun _ _ => false | _ => fold_map2 IndexType.ltb andb true end.
    Lemma expand_eqb {r} (xs ys : Index (S r)) : ((xs =? ys) = ((hd xs =? hd ys) && (@Classes.eqb _ IndexType.eqb (tl xs) (tl ys))))%core%bool.
    Proof.
      cbv [Classes.eqb]; cbn.
      set (b := IndexType.eqb _ _); clearbody b.
      revert b; induction r as [|r IH]; cbn; try reflexivity; intros.
      rewrite !IH; destruct b, IndexType.eqb, eqb; reflexivity.
    Qed.
    Lemma expand_leb {r} (xs ys : Index (S r)) : ((xs <=? ys) = ((hd xs <=? hd ys) && (tl xs <=? tl ys)))%core%bool.
    Proof.
      cbv [Classes.leb]; cbn.
      set (b := IndexType.leb _ _); clearbody b.
      revert b; induction r as [|r IH]; cbn; try reflexivity; intros.
      rewrite !IH; destruct b, IndexType.leb, leb; reflexivity.
    Qed.
    Lemma expand_ltb {r} (xs ys : Index (S r)) : ((xs <? ys) = (match r with O => true | _ => hd xs <? hd ys end && (@Classes.ltb _ IndexType.ltb (tl xs) (tl ys))))%core%bool.
    Proof.
      cbv [Classes.ltb ltb]; cbn.
      set (b := IndexType.ltb _ _); clearbody b.
      revert b; induction r as [|r IH]; cbn; try reflexivity; intros.
      rewrite !IH; destruct b, IndexType.ltb, r; try reflexivity.
      all: destruct fold_map2; reflexivity.
    Qed.
  End ExtendedMake.

  Module Type ExtendedMakeSig (IndexType : ExtendedIndexType) := Nop <+ ExtendedMake IndexType.
End IndexGen.

Module Shape.
  Module ShapeType <: ExtendedIndexType.
    Definition t : Type := int.
    #[global] Strategy 100 [t].
    #[global] Bind Scope uint63_scope with t.
    Definition one : has_one t := _.
    Definition eqb : has_eqb t := _.
    Definition leb : has_leb t := Uint63.leb.
    Definition ltb : has_ltb t := Uint63.ltb.
  End ShapeType.

  Include IndexGen.ExtendedMake ShapeType.

  Module Export ShapeNotations.
    Declare Scope shape_scope.
    Delimit Scope shape_scope with shape.
    Bind Scope shape_scope with t.
    Bind Scope uint63_scope with IndexType.
    Notation "xs ::' x" := (snoc xs x) : shape_scope.
    Notation "[ ]" := nil : shape_scope.
    Notation "[ x ]" := (snoc nil x) : shape_scope.
    Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc nil x) y) .. z) : shape_scope.
    Notation "x :: xs" := (cons x xs) : shape_scope.
    Notation "s1 ++ s2" := (app s1 s2) : shape_scope.
    Notation "s1 ++' s2" := (app s1 s2) : shape_scope.
  End ShapeNotations.

  Definition broadcast2 {r} : Index r -> Index r -> Index r
    := map2 max.
  Definition broadcast3 {r} : Index r -> Index r -> Index r -> Index r
    := map3 (fun a b c => max (max a b) c).
End Shape.
Notation ShapeType := Shape.IndexType.
Notation Shape := Shape.Index.
Export Shape.ShapeNotations.
Export (hints) Shape.

Module RawIndex.
  Module RawIndexType <: ExtendedIndexType.
    Definition t : Type := int.
    #[global] Strategy 100 [t].
    #[global] Bind Scope uint63_scope with t.
    Definition one : has_one t := _.
    Definition eqb : has_eqb t := _.
    Definition leb : has_leb t := Uint63.leb.
    Definition ltb : has_ltb t := Uint63.ltb.
  End RawIndexType.

  Include IndexGen.ExtendedMake RawIndexType.

  Module Export RawIndexNotations.
    Declare Scope raw_index_scope.
    Delimit Scope raw_index_scope with raw_index.
    Bind Scope raw_index_scope with t.
    Bind Scope uint63_scope with IndexType.
    Notation "xs ::' x" := (snoc xs x) : raw_index_scope.
    Notation "[ ]" := nil : raw_index_scope.
    Notation "[ x ]" := (snoc nil x) : raw_index_scope.
    Notation "[ x ; y ; .. ; z ]" :=  (snoc .. (snoc (snoc nil x) y) .. z) : raw_index_scope.
    Notation "x :: xs" := (cons x xs) : raw_index_scope.
    Notation "s1 ++ s2" := (app s1 s2) : raw_index_scope.
    Notation "s1 ++' s2" := (app s1 s2) : raw_index_scope.
  End RawIndexNotations.
End RawIndex.
Notation RawIndexType := RawIndex.IndexType.
Notation RawIndex := RawIndex.Index.
Export RawIndex.RawIndexNotations.
Export (hints) RawIndex.

Module Index.
  Module IndexType <: ExtendedIndexType.
    Definition t : Type := int.
    #[global] Strategy 100 [t].
    #[global] Bind Scope sint63_scope with t.
    Definition one : has_one t := _.
    Definition eqb : has_eqb t := _.
    Definition leb : has_leb t := Sint63.leb.
    Definition ltb : has_ltb t := Sint63.ltb.
  End IndexType.

  Include IndexGen.ExtendedMake IndexType.
  Export IndexNotations.
  Bind Scope sint63_scope with IndexType.
End Index.
Notation IndexType := Index.IndexType.
Notation Index := Index.Index.
Export Index.IndexNotations.
Export (hints) Index.
Bind Scope sint63_scope with Index.IndexType.

Definition tensor_of_rank (A : Type) (r : Rank) : Type
  := RawIndex r -> A.
Definition tensor {r : Rank} (A : Type) (s : Shape r) : Type
  := tensor_of_rank A r.
Declare Scope tensor_scope.
Delimit Scope tensor_scope with tensor.
Declare Scope raw_tensor_scope.
Delimit Scope raw_tensor_scope with tensor.
Bind Scope tensor_scope with tensor_of_rank.
Bind Scope tensor_scope with tensor.
Local Open Scope tensor_scope.

#[export] Instance empty_of_rank {A r} {default : pointed A} : pointed (tensor_of_rank A r)
  := fun _ => default.
#[export] Instance empty {r A} {default : pointed A} {s : Shape r} : pointed (tensor A s)
  := empty_of_rank.

#[export] Typeclasses Opaque Index.

Ltac get_shape val :=
  lazymatch type of val with
  | tensor _ ?shape => shape
  | list ?x
    => let len := (eval cbv in (Uint63.of_Z (Z.of_N (N.of_nat (List.length val))))) in
       let rest := lazymatch (eval hnf in val) with
                   | cons ?val _ => get_shape val
                   | ?val => fail 1 "Could not find cons in" val
                   end in
       constr:(Shape.cons len rest)
  | array ?x
    => let len := (eval cbv in (PArray.length val)) in
       let rest := let val := (eval cbv in (PArray.get val 0)) in
                   get_shape val in
       constr:(Shape.cons len rest)
  | _ => constr:(Shape.nil)
  end.
Notation shape_of x := (match x return _ with y => ltac:(let s := get_shape y in exact s) end) (only parsing).
Class compute_shape_of {A r} (x : A) := get_shape_of : Shape r.
#[global] Hint Extern 0 (compute_shape_of ?x) => let s := get_shape x in exact s : typeclass_instances.

Module PArray.
  Fixpoint concrete_tensor_of_rank (A : Type) (r : Rank) : Type
    := match r with
       | O => A
       | S r => concrete_tensor_of_rank (array A) r
       end.
  Definition concrete_tensor {r : Rank} (A : Type) (s : Shape r) : Type
    := concrete_tensor_of_rank A r.
  #[global] Strategy 100 [tensor_of_rank tensor concrete_tensor concrete_tensor_of_rank].

  Fixpoint concretize {r : Rank} {A : Type} {default : pointed A} {struct r} : forall {s : Shape r} (t : tensor A s), concrete_tensor A s
    := match r with
       | 0%nat => fun _tt f => f tt
       | S r
         => fun '(s, len) f
            => concretize (r:=r) (A:=array A) (s:=s) (fun idxs => PArray.init_default len (fun idx => f (idxs, idx)))
       end.
  Fixpoint abstract_of_rank {r : Rank} {A : Type} {struct r}
    : concrete_tensor_of_rank A r -> tensor_of_rank A r
    := match r with
       | O => fun v _tt => v
       | S r => fun t '(idxs, idx) => PArray.get (@abstract_of_rank r (array A) t idxs) idx
       end.
  Definition abstract {r : Rank} {A : Type} {s : Shape r} : concrete_tensor A s -> tensor A s
    := abstract_of_rank.

  Notation to_tensor t := (@abstract _ _ (shape_of t%array) t%array) (only parsing).

  Lemma abstract_concretize {r A default} {s : Shape r} {t} {idxs : RawIndex r}
    (in_bounds : is_true (match r with O => true | _ => idxs <? s end)%core)
    (in_max_bounds : is_true (match r with O => true | _ => idxs <? RawIndex.repeat PArray.max_length r end)%core)
    : abstract (@concretize r A default s t) idxs = t idxs.
  Proof.
    cbv [abstract].
    revert A default idxs t in_bounds in_max_bounds; induction r as [|r IH]; cbn [abstract_of_rank concretize]; intros.
    { destruct idxs; reflexivity. }
    { cbv [is_true] in *.
      rewrite RawIndex.expand_ltb, Bool.andb_true_iff in in_bounds, in_max_bounds.
      destruct idxs, s.
      rewrite IH by first [ apply in_bounds | apply in_max_bounds ].
      rewrite PArray.get_init_default.
      rewrite RawIndex.tl_repeat in *.
      cbv [Classes.ltb Classes.leb RawIndex.RawIndexType.ltb RawIndex.tl] in *.
      cbn [RawIndex.hd RawIndex.tl RawIndex.repeat] in *.
      cbn in *.
      do 2 destruct PrimInt63.ltb; destruct in_bounds, in_max_bounds; try congruence; cbn.
      reflexivity. }
  Qed.
End PArray.

Module List.
  Fixpoint concrete_tensor_of_rank (A : Type) (r : Rank) : Type
    := match r with
       | O => A
       | S r => concrete_tensor_of_rank (list A) r
       end.
  Definition concrete_tensor {r : Rank} (A : Type) (s : Shape r) : Type
    := concrete_tensor_of_rank A r.
  #[global] Strategy 100 [tensor_of_rank tensor concrete_tensor concrete_tensor_of_rank].

  Fixpoint concretize {r : Rank} {A : Type} {struct r} : forall {s : Shape r} (t : tensor A s), concrete_tensor A s
    := match r return forall {s : Shape r} (t : tensor A s), concrete_tensor A s with
       | 0%nat => fun _tt f => f tt
       | S r
         => fun '(s, len) f
            => concretize (r:=r) (A:=list A) (s:=s) (fun idxs => List.map (fun idx => f (idxs, Uint63.of_Z (Z.of_nat idx))) (List.seq 0 (Z.to_nat (Uint63.to_Z len))))
       end.
  Fixpoint abstract_of_rank {r : Rank} {A : Type} {default : pointed A} {struct r}
    : concrete_tensor_of_rank A r -> tensor_of_rank A r
    := match r return concrete_tensor_of_rank A r -> tensor_of_rank A r with
       | O => fun v _tt => v
       | S r => fun t '(idxs, idx) => nth_default default (@abstract_of_rank r (list A) _ t idxs) (Z.to_nat (Uint63.to_Z idx))
       end.
  Definition abstract {r : Rank} {A : Type} {default : pointed A} {s : Shape r} : concrete_tensor A s -> tensor A s
    := abstract_of_rank.

  Notation to_tensor t := (@abstract _ _ _ (shape_of t%list) t%list) (only parsing).

  Lemma abstract_concretize {r A} {default : pointed A} {s : Shape r} {t} {idxs : RawIndex r}
    (in_bounds : is_true (match r with O => true | _ => idxs <? s end)%core)
    : abstract (@concretize r A s t) idxs = t idxs.
  Proof.
    cbv [abstract].
    revert A default idxs t in_bounds; induction r as [|r IH]; cbn [abstract_of_rank concretize]; intros.
    { destruct idxs; reflexivity. }
    { cbv [is_true] in *.
      rewrite RawIndex.expand_ltb, Bool.andb_true_iff in in_bounds.
      destruct idxs as [idxs idx], s as [ss s].
      rewrite IH by first [ apply in_bounds | apply in_max_bounds ].
      cbv [nth_default].
      rewrite nth_error_map.
      rewrite List.nth_error_seq.
      cbv [Classes.ltb Classes.leb RawIndex.RawIndexType.ltb RawIndex.tl] in *.
      cbn in in_bounds.
      rewrite Uint63.ltb_spec in in_bounds.
      destruct (Uint63.to_Z_bounded idx).
      destruct (Uint63.to_Z_bounded s).
      destruct Nat.ltb eqn:H'; cbn [option_map].
      1:rewrite Nat.ltb_lt, <- Z2Nat.inj_lt in H' by assumption.
      2:rewrite Nat.ltb_ge, <- Z2Nat.inj_le in H' by assumption.
      1: rewrite Nat.add_0_l, Z2Nat.id, of_to_Z by assumption.
      all: first [ reflexivity | lia ]. }
  Qed.
End List.

Definition adjust_indices_for {r} (s : Shape r) : Index r -> RawIndex r
  := Index.map2 (fun s i => i mod s) s.

Definition with_shape {r A} (s : Shape r) : @Shape.curriedT r A -> A
  := fun f => Shape.uncurry f s.

Notation of_array ls := (PArray.to_tensor ls) (only parsing).
Notation of_list ls := (List.to_tensor ls) (only parsing).

Definition repeat {r A} (x : A) {s : Shape r} : tensor A s
  := fun _ => x.
Definition ones {r} {A} {one : has_one A} (s : Shape r) : tensor A s
  := repeat one.
Definition zeros {r} {A} {zero : has_zero A} (s : Shape r) : tensor A s
  := repeat zero.

Definition raw_get {r A} {s : Shape r} (t : tensor A s) (idxs : RawIndex r) : A
  := t idxs.
Definition get {r A} {s : Shape r} (t : tensor A s) (idxs : Index r) : A
  := raw_get t (adjust_indices_for s idxs).

Notation "x .[ y ]" := (get x y) : tensor_scope.
Notation "x .[ y ]" := (raw_get x y) : raw_tensor_scope.

Definition curried_raw_get {r A} {s : Shape r} (t : tensor A s) : @RawIndex.curriedT r A
  := RawIndex.curry (fun idxs => raw_get t idxs).
Definition curried_get {r A} {s : Shape r} (t : tensor A s) : @Index.curriedT r A
  := Index.curry (fun idxs => get t idxs).

Definition map {r A B} {s : Shape r} (f : A -> B) (t : tensor A s) : tensor B s
  := fun i => f (t i).
Definition map2 {r A B C} {sA sB : Shape r} (f : A -> B -> C) (tA : tensor A sA) (tB : tensor B sB) : tensor C (Shape.broadcast2 sA sB)
  := fun i => f (tA i) (tB i).
Definition map3 {r A B C D} {sA sB sC : Shape r} (f : A -> B -> C -> D) (tA : tensor A sA) (tB : tensor B sB) (tC : tensor C sC) : tensor D (Shape.broadcast3 sA sB sC)
  := fun i => f (tA i) (tB i) (tC i).

Definition where_ {r A} {sA : Shape r} {sB : Shape r} {sC : Shape r} (condition : tensor bool sA) (input : tensor A sB) (other : tensor A sC) : tensor A (Shape.broadcast3 sA sB sC)
  := map3 Bool.where_ condition input other.

#[export] Instance tensor_add {r} {sA sB : Shape r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A sA) (tensor B sB) (tensor C (Shape.broadcast2 sA sB)) := map2 add.
#[export] Instance tensor_sub {r} {sA sB : Shape r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A sA) (tensor B sB) (tensor C (Shape.broadcast2 sA sB)) := map2 sub.
#[export] Instance tensor_mul {r} {sA sB : Shape r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A sA) (tensor B sB) (tensor C (Shape.broadcast2 sA sB)) := map2 mul.
#[export] Instance tensor_div_by {r} {sA sB : Shape r} {A B C} {div_byAB : has_div_by A B C} : has_div_by (tensor A sA) (tensor B sB) (tensor C (Shape.broadcast2 sA sB)) := map2 div.
#[export] Instance tensor_sqrt {r} {s : Shape r} {A} {sqrtA : has_sqrt A} : has_sqrt (tensor A s) := map sqrt.
#[export] Instance tensor_opp {r} {s : Shape r} {A} {oppA : has_opp A} : has_opp (tensor A s) := map opp.
#[export] Instance add'1 {r} {s : Shape r} {a b} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A (s ::' a)) (tensor B (s ::' b)) (tensor C (s ::' max a b)) | 10 := tensor_add.
#[export] Instance sub'1 {r} {s : Shape r} {a b} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A (s ::' a)) (tensor B (s ::' b)) (tensor C (s ::' max a b)) | 10 := tensor_sub.
#[export] Instance mul'1 {r} {s : Shape r} {a b} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A (s ::' a)) (tensor B (s ::' b)) (tensor C (s ::' max a b)) | 10 := tensor_mul.
#[export] Instance div_by'1 {r} {s : Shape r} {a b} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A (s ::' a)) (tensor B (s ::' b)) (tensor C (s ::' max a b)) | 10 := tensor_div_by.
#[export] Instance add'1s_r {r} {s : Shape r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A s) (tensor B (@Shape.ones r)) (tensor C s) | 10 := tensor_add.
#[export] Instance add'1s_l {r} {s : Shape r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A (@Shape.ones r)) (tensor B s) (tensor C s) | 10 := tensor_add.
#[export] Instance sub'1s_r {r} {s : Shape r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A s) (tensor B (@Shape.ones r)) (tensor C s) | 10 := tensor_sub.
#[export] Instance sub'1s_l {r} {s : Shape r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A (@Shape.ones r)) (tensor B s) (tensor C s) | 10 := tensor_sub.
#[export] Instance mul'1s_r {r} {s : Shape r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A s) (tensor B (@Shape.ones r)) (tensor C s) | 10 := tensor_mul.
#[export] Instance mul'1s_l {r} {s : Shape r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A (@Shape.ones r)) (tensor B s) (tensor C s) | 10 := tensor_mul.
#[export] Instance div_by'1s_r {r} {s : Shape r} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A s) (tensor B (@Shape.ones r)) (tensor C s) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l {r} {s : Shape r} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A (@Shape.ones r)) (tensor B s) (tensor C s) | 10 := tensor_div_by.
#[export] Instance add'1s_r'1_same {r} {s : Shape r} {a} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A (s ::' a)) (tensor B (@Shape.ones r ::' a)) (tensor C (s ::' a)) | 10 := tensor_add.
#[export] Instance add'1s_l'1_same {r} {s : Shape r} {a} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A (@Shape.ones r ::' a)) (tensor B (s ::' a)) (tensor C (s ::' a)) | 10 := tensor_add.
#[export] Instance sub'1s_r'1_same {r} {s : Shape r} {a} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A (s ::' a)) (tensor B (@Shape.ones r ::' a)) (tensor C (s ::' a)) | 10 := tensor_sub.
#[export] Instance sub'1s_l'1_same {r} {s : Shape r} {a} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A (@Shape.ones r ::' a)) (tensor B (s ::' a)) (tensor C (s ::' a)) | 10 := tensor_sub.
#[export] Instance mul'1s_r'1_same {r} {s : Shape r} {a} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A (s ::' a)) (tensor B (@Shape.ones r ::' a)) (tensor C (s ::' a)) | 10 := tensor_mul.
#[export] Instance mul'1s_l'1_same {r} {s : Shape r} {a} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A (@Shape.ones r ::' a)) (tensor B (s ::' a)) (tensor C (s ::' a)) | 10 := tensor_mul.
#[export] Instance div_by'1s_r'1_same {r} {s : Shape r} {a} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A (s ::' a)) (tensor B (@Shape.ones r ::' a)) (tensor C (s ::' a)) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l'1_same {r} {s : Shape r} {a} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A (@Shape.ones r ::' a)) (tensor B (s ::' a)) (tensor C (s ::' a)) | 10 := tensor_div_by.
#[export] Instance add'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A (s ++' s')) (tensor B (@Shape.ones r ++' s')) (tensor C (s ++' s')) | 10 := tensor_add.
#[export] Instance add'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {addA : has_add_with A B C} : has_add_with (tensor A (@Shape.ones r ++' s')) (tensor B (s ++' s')) (tensor C (s ++' s')) | 10 := tensor_add.
#[export] Instance sub'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A (s ++' s')) (tensor B (@Shape.ones r ++' s')) (tensor C (s ++' s')) | 10 := tensor_sub.
#[export] Instance sub'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor A (@Shape.ones r ++' s')) (tensor B (s ++' s')) (tensor C (s ++' s')) | 10 := tensor_sub.
#[export] Instance mul'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A (s ++' s')) (tensor B (@Shape.ones r ++' s')) (tensor C (s ++' s')) | 10 := tensor_mul.
#[export] Instance mul'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor A (@Shape.ones r ++' s')) (tensor B (s ++' s')) (tensor C (s ++' s')) | 10 := tensor_mul.
#[export] Instance div_by'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A (s ++' s')) (tensor B (@Shape.ones r ++' s')) (tensor C (s ++' s')) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor A (@Shape.ones r ++' s')) (tensor B (s ++' s')) (tensor C (s ++' s')) | 10 := tensor_div_by.

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

Definition reshape_app_split {A r1 r2 s1 s2} : @tensor (r1 +' r2) A (s1 ++' s2) -> tensor (tensor A s2) s1
  := RawIndex.curry_radd.
Definition reshape_app_combine {A r1 r2 s1 s2} : tensor (tensor A s2) s1 -> @tensor (r1 +' r2) A (s1 ++' s2)
  := RawIndex.uncurry_radd.
(* infer s1 s2 from the conclusion *)
#[global] Arguments reshape_app_combine A & r1 r2 s1 s2 _.
#[global] Arguments reshape_app_split A & r1 r2 s1 s2 _.
Definition reshape_snoc_split {A r s1 s2} : @tensor (r +' 1) A (s1 ::' s2) -> tensor (tensor A [s2]) s1
  := RawIndex.curry_radd.
Definition reshape_snoc_combine {A r s1 s2} : tensor (tensor A [s2]) s1 -> @tensor (r +' 1) A (s1 ::' s2)
  := RawIndex.uncurry_radd.
Definition uncurry {r A} {s : Shape r} : @RawIndex.curriedT r A -> tensor A s
  := RawIndex.uncurry.
Definition curry {r A} {s : Shape r} : tensor A s -> @RawIndex.curriedT r A
  := RawIndex.curry.
(*
Definition reshape_S_fun_combine {I A} {r : Rank} : (I -> tensor_fun_of_rank I A r) -> tensor_fun_of_rank I A (1 +' r)
  := match reshape_app_combine_gen (r1:=1) (r2:=r) with x => x end.
Definition reshape_S_fun_split {I A} {r : Rank} : tensor_fun_of_rank I A (1 +' r) -> (I -> tensor_fun_of_rank I A r)
  := match reshape_app_split_gen (r1:=1) (r2:=r) with x => x end.
*)
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

Definition broadcast' {A} (x : A) {r : Rank} : tensor A (@Shape.ones r)
  := repeat x.
Definition broadcast {r A} {s : Shape r} (x : tensor A s) {r' : Rank} : tensor A (@Shape.ones r' ++' s)
  := reshape_app_combine (broadcast' x).

Definition keepdim_gen {r} {s : Shape r} {A B} (f : A -> tensor B s) : A -> tensor B ([1] ++' s)
  := fun a => broadcast (f a).
Definition keepdim {A B} (f : A -> B) : A -> tensor B [1] := keepdim_gen (s:=[]) (fun a 'tt => f a).
#[local] Notation keepdimf := keepdim (only parsing).

Definition reduce_axis_m1' {r A B} {s1 : Shape r} {s2}
  (reduction : forall (start stop step : RawIndexType), (RawIndexType -> A) -> B)
  (t : tensor A (s1 ::' s2))
  : tensor B s1
  := map (fun v => reduction 0 s2 1 (fun i => raw_get v [i])) (reshape_snoc_split t).

Definition reduce_axis_m1 {r A B} {s1 : Shape r} {s2} {keepdim : with_default "keepdim" bool false}
  (reduction : forall (start stop step : RawIndexType), (RawIndexType -> A) -> B)
  : tensor A (s1 ::' s2) -> tensor B (s1 ++' if keepdim return Shape (if keepdim then _ else _) then [1] else [])
  := if keepdim
          return tensor A (s1 ::' s2) -> tensor B (s1 ++' if keepdim return Shape (if keepdim then _ else _) then [1] else [])
     then fun t idxs => reduce_axis_m1' reduction t (RawIndex.hd idxs)
     else reduce_axis_m1' reduction.

Definition to_bool {A} {zero : has_zero A} {eqb : has_eqb A} {r} {s : Shape r} (xs : tensor A s) : tensor bool s
  := map (fun x => x ≠? 0)%core xs.

(** Quoting https://pytorch.org/docs/stable/generated/torch.tril.html

torch.tril(input, diagonal=0, *, out=None) → Tensor

Returns the lower triangular part of the matrix (2-D tensor) or batch
of matrices [input], the other elements of the result tensor [out] are
set to 0.

The lower triangular part of the matrix is defined as the elements on
and below the diagonal.

The argument [diagonal] controls which diagonal to consider. If
[diagonal = 0], all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the
main diagonal, and similarly a negative value excludes just as many
diagonals below the main diagonal. The main diagonal are the set of
indices {(i,i)} for i ∈ [0,min{d₁,d₂}−1] where d₁,d₂ are the
dimensions of the matrix. *)
Definition tril {A} {zero : has_zero A} {rnk} {s : Shape rnk} {r c}
  {diagonal : with_default "diagonal" int 0%int63} (input : tensor A (s ++' [r; c]))
  : tensor A (s ++' [r; c])
  := fun '(((_, i), j) as idxs)
     => if ((0 ≤? i) && (i <? r) && (Sint63.max 0 (1 + i + diagonal) ≤? j) && (j <? c))%bool
        then 0%core
        else input idxs.
#[global] Arguments tril {A%type_scope zero rnk%nat s%shape} {r c}%uint63 {diagonal}%sint63 input%tensor.
(** Quoting https://pytorch.org/docs/stable/generated/torch.triu.html

torch.triu(input, diagonal=0, *, out=None) → Tensor

Returns the upper triangular part of the matrix (2-D tensor) or batch
of matrices [input], the other elements of the result tensor [out] are
set to 0.

The upper triangular part of the matrix is defined as the elements on
and above the diagonal.

The argument [diagonal] controls which diagonal to consider. If
[diagonal = 0], all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the
main diagonal, and similarly a negative value includes just as many
diagonals below the main diagonal. The main diagonal are the set of
indices {(i,i)} for i ∈ [0,min{d₁,d₂}−1] where d₁,d₂ are the
dimensions of the matrix. *)
Definition triu {A} {zero : has_zero A} {rnk} {s : Shape rnk} {r c}
  {diagonal : with_default "diagonal" int 0%int63} (input : tensor A (s ++' [r; c]))
  : tensor A (s ++' [r; c])
  := fun '(((_, i), j) as idxs)
     => if ((0 ≤? i) && (i <? r) && (0 ≤? j) && (j <? Sint63.max 0 (i + diagonal)))%bool
        then 0%core
        else input idxs.
#[global] Arguments triu {A%type_scope zero rnk%nat s%shape} {r c}%uint63 {diagonal}%sint63 input%tensor.
