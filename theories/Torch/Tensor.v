From Coq.Structures Require Import Equalities.
From Coq Require Import ZArith Sint63 Uint63 List PArray Lia.
From NeuralNetInterp.Util Require Nat.
From NeuralNetInterp.Util.Tactics Require Import ClearAll ClearbodyAll.
From NeuralNetInterp.Util Require Import Wf_Uint63 PArray.Proofs List.Proofs Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool (*PrimitiveProd*).
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.Reduction.
Import Arith.Classes.
Import Instances.Uint63.
Local Open Scope list_scope.
Set Implicit Arguments.
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
  Parameter zero : has_zero t.
  Parameter one : has_one t.
  Parameter leb : has_leb t.
  Parameter ltb : has_ltb t.
  Parameter eqb : has_eqb t.
  Parameter mul : has_mul t.
  Parameter add : has_add t.
  Parameter int_div : has_int_div t.
  Parameter modulo : has_mod t.
  #[export] Existing Instances eqb one zero ltb leb mul add int_div modulo.
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

    Polymorphic Fixpoint fold_map {A B r} (f : IndexType -> A) (accum : B -> A -> B) (init : B) : Index r -> B
      := match r with
         | 0%nat => fun _ => init
         | S r => fun xs => fold_map f accum (accum init (f (tl xs))) (hd xs)
         end.

    Polymorphic Fixpoint fold_map2 {A B r} (f : IndexType -> IndexType -> A) (accum : B -> A -> B) (init : B) : Index r -> Index r -> B
      := match r with
         | 0%nat => fun _ _ => init
         | S r => fun xs ys => fold_map2 f accum (accum init (f (tl xs) (tl ys))) (hd xs) (hd ys)
         end.

    Polymorphic Fixpoint map_fold_map
      {r}
      {A} {f : IndexType -> A} {accum : Type -> A -> Type} {init : Type}
      {A'} {f' : IndexType -> A'} {accum' : Type -> A' -> Type} {init' : Type}
      (Finit : init -> init')
      (F : forall b i b', (b -> b') -> accum b (f i) -> accum' b' (f' i))
      {struct r}
      : forall idx : Index r, @fold_map A Type r f accum init idx -> @fold_map A' Type r f' accum' init' idx
      := match r return forall idx : Index r, @fold_map A Type r f accum init idx -> @fold_map A' Type r f' accum' init' idx with
         | 0%nat => fun _ => Finit
         | S r => fun idx => @map_fold_map r A f accum _ A' f' accum' _ (F _ _ _ Finit) F (hd idx)
         end.

    Polymorphic Fixpoint map2_fold_map
      {r}
      {A} {f : IndexType -> A} {accum : Type -> A -> Type} {init : Type}
      {A'} {f' : IndexType -> A'} {accum' : Type -> A' -> Type} {init' : Type}
      {A''} {f'' : IndexType -> A''} {accum'' : Type -> A'' -> Type} {init'' : Type}
      (Finit : init -> init' -> init'')
      (F : forall b b' b'' i, (b -> b' -> b'') -> accum b (f i) -> accum' b' (f' i) -> accum'' b'' (f'' i))
      {struct r}
      : forall idx : Index r, @fold_map A Type r f accum init idx -> @fold_map A' Type r f' accum' init' idx -> @fold_map A'' Type r f'' accum'' init'' idx
      := match r return forall idx : Index r, @fold_map A Type r f accum init idx -> @fold_map A' Type r f' accum' init' idx -> @fold_map A'' Type r f'' accum'' init'' idx with
         | 0%nat => fun _ => Finit
         | S r => fun idx => map2_fold_map (r:=r) (F _ _ _ _ Finit) F (hd idx)
         end.

    Definition tuple {r} (A : IndexType -> Type) (s : Index r) : Type
      := fold_map A (fun x y => Datatypes.prod y x) unit s.

    Module Tuple.
      Fixpoint init' {r} : forall {A B} {s : Index r},
          (forall i, A i)
          -> B
          -> fold_map A (fun x y => Datatypes.prod y x) B s
        := match r with
           | O => fun A B s f b => b
           | S r => fun A B s f b => @init' r A (A (tl s) * B)%type (hd s) f (f (tl s), b)
           end.

      Definition init {r A} {s : Index r} (f : forall i, A i) : @tuple r A s
        := init' f tt.

      Definition map {r A A' s} (f : forall idx, A idx -> A' idx) : @tuple r A s -> @tuple r A' s
        := map_fold_map (fun tt => tt) (fun _ _ _ f_snd xy => (f _ (fst xy), f_snd (snd xy))) _.

      Definition map2 {r A A' A'' s} (f : forall idx, A idx -> A' idx -> A'' idx) : @tuple r A s -> @tuple r A' s -> @tuple r A'' s
        := map2_fold_map (fun _ _ => tt) (fun _ _ _ _ f_snd xy x'y' => (f _ (fst xy) (fst x'y'), f_snd (snd xy) (snd x'y'))) _.

      Fixpoint to_list' {r} : forall {A B C} {s : Index r},
          (forall i, A i -> C)
          -> (B -> list C)
          -> fold_map A (fun x y => Datatypes.prod y x) B s
          -> list C
        := match r with
           | O => fun A B C s g f ts => f ts
           | S r
             => fun A B C s g f ts
                => let f := fun ba => (g _ (fst ba) :: f (snd ba))%list in
                   @to_list' r A (A (tl s) * B)%type C (hd s) g f ts
           end.
      Definition to_list {r A s} (ts : @tuple r (fun _ => A) s) : list A
        := to_list' (fun _ x => x) (fun _ => []%list) ts.

      Definition nth_error {r A s} (ts : @tuple r (fun _ => A) s) n : option A
        := nth_error (to_list ts) n.
      Definition nth_default {r A s} default (ts : @tuple r (fun _ => A) s) n : A
        := nth_default default (to_list ts) n.
    End Tuple.

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

    Ltac curry_let f :=
      let f' := fresh f in
      let f'' := fresh f in
      rename f into f';
      pose (curry f') as f;
      pose (uncurry f) as f'';
      cbv [uncurry curry uncurry_dep curry_dep uncurry_map_dep snoc nil] in *;
      repeat match goal with
        | [ |- context C[f'] ]
          => let C' := context C[f''] in
             cut C'; [ clear_all; clearbody_all_has_evar; abstract (subst f f' f''; cbv beta iota; exact (fun x => x)) | ]
        | [ H := context C[f'] |- _ ]
          => let C' := context C[f''] in
             let H' := fresh H in
             rename H into H';
             pose C' as H;
             assert (H = H') by (clear_all; clearbody_all_has_evar; abstract (subst H H' f f' f''; cbv iota beta; reflexivity));
             clearbody H'; subst H'
        | [ H : context C [f'] |- _ ] => let C' := context C[f''] in change C' in H
        end;
      cbv [f'] in f; clear f'; hnf in f, f''; subst f''; cbn [fst snd] in *.
    Ltac curry_let_step _ :=
      match goal with
      | [ f := _ |- _ ] => curry_let f
      end.
    Ltac curry_lets _ := repeat curry_let_step ().

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

    Fixpoint reduce {A} (reduction : A -> IndexType -> A) (init : A) {r} : Index r -> A
      := match r with
         | 0%nat => fun _ => init
         | S r => fun idxs => reduction (reduce reduction init (hd idxs)) (tl idxs)
         end.
  End Make.

  Module Type MakeSig (IndexType : IndexType) := Nop <+ Make IndexType.

  Module ExtendedMake (IndexType : ExtendedIndexType).
    Import (hints) IndexType.
    Include Make IndexType.
    Import IndexNotations.
    #[local] Open Scope index_scope.

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

    Module Rank.
      Fixpoint filter {r : Rank} (f : IndexType -> bool) : Index r -> Rank
        := match r with
           | 0%nat => fun _ => 0%nat
           | S r => fun idx => @filter r f (hd idx) +' if f (tl idx) then 1 else 0
           end.
      Definition squeeze {r : Rank} (i : Index r) : Rank
        := filter (fun i => (i != 1)%core) i.
    End Rank.
    Fixpoint filter {r : Rank} (f : IndexType -> bool) : forall i : Index r, Index (Rank.filter f i)
      := match r return forall i : Index r, Index (Rank.filter f i) with
         | 0%nat => fun _ => []
         | S r => fun idx => @filter r f (hd idx) ++' if f (tl idx) as ftlidx return Index (if ftlidx then _ else _) then [tl idx] else []
         end.

    Definition squeeze {r : Rank} (i : Index r) : Index (Rank.squeeze i)
      := filter (fun i => (i != 1)%core) i.

    Fixpoint unfilter {r : Rank} (f : IndexType -> bool) : forall {i : Index r}, Index (Rank.filter f i) -> Index r
      := match r return forall i : Index r, Index (Rank.filter f i) -> Index r with
         | 0%nat => fun _ idxs => idxs
         | S r
           => fun idx
              => if f (tl idx) as ftlidx return Index (_ +' if ftlidx then 1 else 0) -> Index (S r)
                 then fun idxs => @unfilter r f (hd idx) (hd idxs) ::' tl idxs
                 else fun idxs => @unfilter r f (hd idx) idxs ::' 0
         end%core.

    Definition unsqueeze {r : Rank} {i : Index r} : Index (Rank.squeeze i) -> Index r
      := unfilter (fun i => (i != 1)%core).

    Definition prod {r} : Index r -> IndexType
      := reduce mul 1%core.
  End ExtendedMake.

  Module Type ExtendedMakeSig (IndexType : ExtendedIndexType) := Nop <+ ExtendedMake IndexType.
End IndexGen.

Module Shape.
  Module ShapeType <: ExtendedIndexType.
    Definition t : Type := int.
    #[global] Strategy 100 [t].
    #[global] Bind Scope uint63_scope with t.
    Definition one : has_one t := _.
    Definition zero : has_zero t := _.
    Definition eqb : has_eqb t := _.
    Definition mul : has_mul t := _.
    Definition add : has_add t := _.
    Definition int_div : has_int_div t := _.
    Definition modulo : has_mod t := _.
    (* eta expand to get around COQBUG(https://github.com/coq/coq/issues/17663) *)
    Definition leb : has_leb t := fun x y => Uint63.leb x y.
    Definition ltb : has_ltb t := fun x y => Uint63.ltb x y.
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

  Definition reshape' {r} : Index r -> Z
    := Shape.reduce (fun z x => z * Uint63.to_Z x)%Z 1%Z.
  Definition reshape {r} (s : Index r) : Index 1
    := [Uint63.of_Z (reshape' s)].

  Definition keepdim {keepdim : with_default "keepdim" bool false} : t _
    := if keepdim return t (if keepdim then _ else _) then [1] else [].
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
    Definition zero : has_zero t := _.
    Definition eqb : has_eqb t := _.
    Definition mul : has_mul t := _.
    Definition add : has_add t := _.
    Definition int_div : has_int_div t := _.
    Definition modulo : has_mod t := _.
    (* eta expand to get around COQBUG(https://github.com/coq/coq/issues/17663) *)
    Definition leb : has_leb t := fun x y => Uint63.leb x y.
    Definition ltb : has_ltb t := fun x y => Uint63.ltb x y.
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

  Fixpoint reshape' {r} : Shape r -> Index r -> Z
    := match r with
       | 0%nat => fun s idx => 0
       | S r
         => fun s idx
            => Uint63.to_Z (tl idx) + @reshape' r (Shape.hd s) (hd idx) * Uint63.to_Z (Shape.tl s)
       end%Z%core%raw_index.

  Fixpoint unreshape' {r} : Shape r -> Z -> Index r
    := match r with
       | 0%nat => fun _ _ => []
       | S r
         => fun s idx
            => let tl_idx := idx mod (Uint63.to_Z (Shape.tl s)) in
               let hd_idx := idx // (Uint63.to_Z (Shape.tl s)) in
               @unreshape' r (Shape.hd s) hd_idx ::' Uint63.of_Z tl_idx
       end%Z%core%raw_index.

  Definition reshape {r} (s : Shape r) (idx : Index r) : Index 1 := [Uint63.of_Z (reshape' s idx)].
  Definition unreshape {r} (s : Shape r) (idx : Index 1) : Index r := unreshape' s (Uint63.to_Z (item idx)).

  Lemma unrereshape' {r} s idx : (match r with 0%nat => true | _ => idx <? s end)%core -> @unreshape' r s (@reshape' r s idx) = idx /\ (0 <= @reshape' r s idx < Shape.reshape' s)%Z.
  Proof.
    induction r; [ | rewrite expand_ltb ];
      cbv [Shape.reshape'] in *;
      cbn [Shape Index unreshape' reshape' Shape.reduce] in *;
      cbv [is_true Shape.hd Shape.tl snoc nil hd tl fst snd Classes.int_div Classes.add Classes.mul Classes.modulo Z_has_int_div] in *;
      repeat match goal with H : unit |- _ => destruct H | H : _ * _ |- _ => destruct H end; try (split; try reflexivity; lia);
      cbv [Classes.ltb RawIndexType.ltb Uint63.ltb] in *.
    all: rewrite Bool.andb_true_iff, Z_mod_plus_full, Uint63.ltb_spec.
    all: intros [H0 H1].
    repeat match goal with
           | [ |- context[to_Z ?x] ]
             => let lem := constr:(to_Z_bounded x) in
                let ty := type of lem in
                lazymatch goal with
                | [ _ : ty |- _ ] => fail
                | _ => idtac
                end;
                pose proof lem
           end.
    rewrite Z.mod_small, Z_div_plus_full, Z.div_small, Z.add_0_l, Uint63.of_to_Z by lia.
    destruct r;
      [ cbn [Shape Index unreshape' reshape' Shape.reduce] in *;
        cbv [is_true Shape.hd Shape.tl snoc nil hd tl fst snd Classes.int_div Classes.add Classes.mul Classes.modulo Z_has_int_div] in *;
        repeat match goal with H : unit |- _ => destruct H | H : _ * _ |- _ => destruct H end; split; try reflexivity; lia
      | ].
    specialize (IHr _ _ ltac:(eassumption)).
    destruct IHr as [IHr1 IHr2].
    rewrite IHr1; split; try reflexivity.
    nia.
  Qed.

  Lemma reunreshape' {r} s idx : (0 <= idx < Shape.reshape' s)%Z -> @reshape' r s (@unreshape' r s idx) = idx /\ (match r with 0%nat => true | _ => @unreshape' r s idx <? s end)%core.
  Proof.
    revert idx; induction r; intro idx; [ | rewrite expand_ltb ];
      cbv [Shape.reshape'] in *;
      cbn [Shape Index unreshape' reshape' Shape.reduce] in *;
      cbv [is_true Shape.hd Shape.tl snoc nil hd tl fst snd Classes.int_div Classes.add Classes.mul Classes.modulo Z_has_int_div] in *;
      repeat match goal with H : unit |- _ => destruct H | H : _ * _ |- _ => destruct H end; try (split; try reflexivity; lia);
      cbv [Classes.ltb RawIndexType.ltb Uint63.ltb] in *.
    all: rewrite Bool.andb_true_iff, Uint63.ltb_spec, !Uint63.of_Z_spec.
    intro H.
    repeat match goal with
           | [ |- context[to_Z ?x] ]
             => let lem := constr:(to_Z_bounded x) in
                let ty := type of lem in
                lazymatch goal with
                | [ _ : ty |- _ ] => fail
                | _ => idtac
                end;
                pose proof lem
           | [ |- context[(?x mod ?y)%Z] ]
             => let lem := constr:(Z.mod_pos_bound x y ltac:(lia)) in
                let ty := type of lem in
                lazymatch goal with
                | [ _ : ty |- _ ] => fail
                | _ => idtac
                end;
                pose proof lem
           | [ H : (0 <= ?idx < ?x * ?y)%Z |- _ ]
             => lazymatch goal with
                | [ _ : (0 < y)%Z |- _ ] => fail
                | _ => idtac
                end;
                assert (0 < x)%Z by nia;
                assert (0 < y)%Z by nia
           end.
    rewrite ?Z.mod_small by lia.
    match goal with
    | [ |- context[reshape' ?s (unreshape' ?s ?idx)] ]
      => specialize (IHr s idx)
    end.
    specialize (IHr ltac:(Z.to_euclidean_division_equations; nia)).
    destruct IHr as [IHr1 IHr2].
    rewrite IHr1; repeat split; try (now destruct r); try lia; [].
    Z.to_euclidean_division_equations; nia.
  Qed.
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
    Definition zero : has_zero t := _.
    Definition eqb : has_eqb t := _.
    Definition mul : has_mul t := _.
    Definition add : has_add t := _.
    Definition int_div : has_int_div t := _.
    Definition modulo : has_mod t := _.
    (* eta expand to get around COQBUG(https://github.com/coq/coq/issues/17663) *)
    Definition leb : has_leb t := fun x y => Sint63.leb x y.
    Definition ltb : has_ltb t := fun x y => Sint63.ltb x y.
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

(*
Definition tensor_of_rank@{a r} (A : Type@{a}) (r : Rank)
  := RawIndex@{r} r -> A.
(* we could have a separate universe for the shape, but since the shape argument is a phantom one anyway, we don't bother *)
Definition tensor@{a r} {r : Rank} (A : Type@{a}) (s : Shape@{r} r)
  := tensor_of_rank@{a r} A r.
*)
Monomorphic Definition tensor_of_rank (r : Rank) (A : Type) : Type
  := RawIndex r -> A.
Monomorphic Definition tensor {r : Rank} (s : Shape r) (A : Type) : Type
  := tensor_of_rank r A.

Monomorphic Definition tensor_dep {r s A} (P : A -> Type) (x : @tensor r s A)
  := forall i : RawIndex r, P (x i).

Definition tensor_undep {r s A P x} (t : @tensor_dep r s A (fun _ => P) x) : @tensor r s P
  := t.

Declare Scope tensor_scope.
Delimit Scope tensor_scope with tensor.
Declare Scope raw_tensor_scope.
Delimit Scope raw_tensor_scope with raw_tensor.
Bind Scope tensor_scope with tensor_of_rank.
Bind Scope tensor_scope with tensor.
Local Open Scope tensor_scope.

#[export] Instance empty_of_rank {r A} {default : pointed A} : pointed (tensor_of_rank r A)
  := fun _ => default.
#[export] Instance empty {r A} {default : pointed A} {s : Shape r} : pointed (tensor s A)
  := empty_of_rank.

#[export] Typeclasses Opaque Index.

Ltac get_shape val :=
  lazymatch type of val with
  | tensor ?shape _ => shape
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
Class compute_shape_of {r A} (x : A) := get_shape_of : Shape r.
#[global] Hint Extern 0 (compute_shape_of ?x) => let s := get_shape x in exact s : typeclass_instances.

Module PArray.
  Fixpoint concrete_tensor_of_rank (r : Rank) (A : Type) : Type
    := match r with
       | O => A
       | S r => concrete_tensor_of_rank r (array A)
       end.
  Definition concrete_tensor {r : Rank} (s : Shape r) (A : Type) : Type
    := concrete_tensor_of_rank r A.
  #[global] Strategy 100 [tensor_of_rank tensor concrete_tensor concrete_tensor_of_rank].

  Module Tensor.
    Fixpoint map {r s A B} (f : A -> B) : @concrete_tensor r s A -> @concrete_tensor r s B
      := match r return forall {s}, @concrete_tensor r s A -> @concrete_tensor r s B with
         | O => fun _ => f
         | S r => fun s t => @map r (Shape.hd s) _ _ (PArray.map f) t
         end s.
    Definition copy {r s A} t := @map r s A A (fun x => x) t.
  End Tensor.

  Fixpoint concretize {r : Rank} {s : Shape r} {A : Type} {default : pointed A} {struct r} : forall (t : tensor s A), concrete_tensor s A
    := match r return forall (s : Shape r) (t : tensor s A), concrete_tensor s A with
       | 0%nat => fun _tt f => f tt
       | S r
         => fun '(s, len) f
            => concretize (r:=r) (A:=array A) (s:=s) (fun idxs => PArray.init_default len (fun idx => f (idxs, idx)))
       end s.
  Fixpoint abstract_of_rank {r : Rank} {A : Type} {struct r}
    : concrete_tensor_of_rank r A -> tensor_of_rank r A
    := match r with
       | O => fun v _tt => v
       | S r => fun t '(idxs, idx) => PArray.get (@abstract_of_rank r (array A) t idxs) idx
       end.
  Definition abstract {r : Rank} {s : Shape r} {A : Type} : concrete_tensor s A -> tensor s A
    := abstract_of_rank.

  Notation to_tensor t := (@abstract _ (shape_of t%array) _ t%array) (only parsing).

  Lemma abstract_concretize {r} {s : Shape r} {A default} {t} {idxs : RawIndex r}
    (in_bounds : is_true (match r with O => true | _ => idxs <? s end)%core)
    (in_max_bounds : is_true (match r with O => true | _ => idxs <? RawIndex.repeat PArray.max_length r end)%core)
    : abstract (@concretize r s A default t) idxs = t idxs.
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

  Definition reabstract {r s A} (t_ : unit -> @tensor r s A) (t : @concrete_tensor r s A) : @tensor r s A
    := let t := abstract t in
       fun idxs
       => if ((idxs <? s) && (idxs <? RawIndex.repeat PArray.max_length r))%core%bool
          then t idxs
          else t_ tt idxs.

  Lemma reabstract_correct {r A} {s : Shape r} {t_} {t} {idxs : RawIndex r}
    : (forall
          (in_bounds : is_true (match r with O => true | _ => idxs <? s end)%core)
          (in_max_bounds : is_true (match r with O => true | _ => idxs <? RawIndex.repeat PArray.max_length r end)%core),
          abstract t idxs = t_ tt idxs)
      -> @reabstract r s A t_ t idxs = t_ tt idxs.
  Proof.
    cbv [reabstract].
    cbv [andb].
    repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end.
    all: repeat match goal with H : context[match ?x with _ => _ end] |- _ => destruct x eqn:? end.
    all: auto.
    all: discriminate.
  Qed.

  Lemma reabstract_ext_correct {r s A default} {t_ t}
    : t = @concretize r s A default (t_ tt) -> forall idxs, @reabstract r s A t_ t idxs = t_ tt idxs.
  Proof. intros; subst; apply reabstract_correct, abstract_concretize. Qed.

  Definition checkpoint {r s A default} t : @tensor r s A
    := let t_ _ := t in
       let t := @concretize r s A default t in
       reabstract t_ t.

  Lemma checkpoint_correct {r s A default t} {idxs : RawIndex r}
    : @checkpoint r s A default t idxs = t idxs.
  Proof. cbv [checkpoint]; erewrite reabstract_ext_correct; reflexivity. Qed.

  Definition maybe_checkpoint {r s A default} {use_checkpoint : with_default "use_checkpoint" bool true} t : @tensor r s A
    := if use_checkpoint then @checkpoint r s A default t else t.

  Lemma maybe_checkpoint_correct {r s A default use_checkpoint t} {idxs : RawIndex r}
    : @maybe_checkpoint r s A default use_checkpoint t idxs = t idxs.
  Proof. cbv [maybe_checkpoint]; destruct use_checkpoint; rewrite ?checkpoint_correct; reflexivity. Qed.
End PArray.

Module List.
  Fixpoint concrete_tensor_of_rank (r : Rank) (A : Type) : Type
    := match r with
       | O => A
       | S r => concrete_tensor_of_rank r (list A)
       end.
  Definition concrete_tensor {r : Rank} (s : Shape r) (A : Type) : Type
    := concrete_tensor_of_rank r A.
  #[global] Strategy 100 [tensor_of_rank tensor concrete_tensor concrete_tensor_of_rank].

  Module Tensor.
    Fixpoint map {r s A B} (f : A -> B) : @concrete_tensor r s A -> @concrete_tensor r s B
      := match r return forall {s : Shape r}, @concrete_tensor r s A -> @concrete_tensor r s B with
         | O => fun _ => f
         | S r => fun s t => @map r (Shape.hd s) _ _ (List.map f) t
         end s.
    Definition copy {r s A} t := @map r s A A (fun x => x) t.
  End Tensor.

  Fixpoint concretize {r : Rank} {s : Shape r} {A : Type} {struct r} : forall (t : tensor s A), concrete_tensor s A
    := match r return forall {s : Shape r} (t : tensor s A), concrete_tensor s A with
       | 0%nat => fun _tt f => f tt
       | S r
         => fun '(s, len) f
            => concretize (r:=r) (A:=list A) (s:=s) (fun idxs => List.map (fun idx => f (idxs, Uint63.of_Z (Z.of_nat idx))) (List.seq 0 (Z.to_nat (Uint63.to_Z len))))
       end s.
  Fixpoint abstract_of_rank {r : Rank} {A : Type} {default : pointed A} {struct r}
    : concrete_tensor_of_rank r A -> tensor_of_rank r A
    := match r return concrete_tensor_of_rank r A -> tensor_of_rank r A with
       | O => fun v _tt => v
       | S r => fun t '(idxs, idx) => nth_default default (@abstract_of_rank r (list A) _ t idxs) (Z.to_nat (Uint63.to_Z idx))
       end.
  Definition abstract {r : Rank} {s : Shape r} {A : Type} {default : pointed A} : concrete_tensor s A -> tensor s A
    := abstract_of_rank.

  Notation to_tensor t := (@abstract _ (shape_of t%list) _ _ t%list) (only parsing).

  Lemma abstract_concretize {r s A} {default : pointed A} {t} {idxs : RawIndex r}
    (in_bounds : is_true (match r with O => true | _ => idxs <? s end)%core)
    : abstract (@concretize r s A t) idxs = t idxs.
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

  Definition reabstract {r s A default} (t_ : unit -> @tensor r s A) (t : @concrete_tensor r s A) : @tensor r s A
    := let t := @abstract r s A default t in
       fun idxs
       => if (idxs <? s)%core
          then t idxs
          else t_ tt idxs.

  Lemma reabstract_correct {r s A default} {t_} {t} {idxs : RawIndex r}
    : (forall
          (in_bounds : is_true (match r with O => true | _ => idxs <? s end)%core),
          abstract t idxs = t_ tt idxs)
      -> @reabstract r s A default t_ t idxs = t_ tt idxs.
  Proof.
    cbv [reabstract].
    repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end.
    all: auto.
  Qed.

  Lemma reabstract_ext_correct {r A default} {s : Shape r} {t_ t}
    : t = @concretize r s A (t_ tt) -> forall idxs, @reabstract r s A default t_ t idxs = t_ tt idxs.
  Proof. intros; subst; apply reabstract_correct, abstract_concretize. Qed.

  Definition checkpoint {r : Rank} {s A default} t : @tensor r s A
    := let t_ _ := t in
       let t := @concretize r s A t in
       @reabstract r s A default t_ t.

  Lemma checkpoint_correct {r s A default} {t} {idxs : RawIndex r}
    : @checkpoint r s A default t idxs = t idxs.
  Proof. cbv [checkpoint]; erewrite reabstract_ext_correct; reflexivity. Qed.

  Definition maybe_checkpoint {r s A default} {use_checkpoint : with_default "use_checkpoint" bool true} t : @tensor r s A
    := if use_checkpoint then @checkpoint r s A default t else t.

  Lemma maybe_checkpoint_correct {r s A default use_checkpoint t} {idxs : RawIndex r}
    : @maybe_checkpoint r s A default use_checkpoint t idxs = t idxs.
  Proof. cbv [maybe_checkpoint]; destruct use_checkpoint; rewrite ?checkpoint_correct; reflexivity. Qed.
End List.

Definition adjust_index_for (s : ShapeType) : Index.IndexType -> RawIndex.IndexType
  := fun i => i mod s.

Definition adjust_indices_for {r} (s : Shape r) : Index r -> RawIndex r
  := Index.map2 adjust_index_for s.

Definition with_shape {r} (s : Shape r) {A} : @Shape.curriedT r A -> A
  := fun f => Shape.uncurry f s.

Notation of_array ls := (PArray.to_tensor ls) (only parsing).
Notation of_list ls := (List.to_tensor ls) (only parsing).

Definition repeat' {r} {s : Shape r} {A} (x : A) : tensor s A
  := fun _ => x.
Definition ones {r} (s : Shape r) {A} {one : has_one A} : tensor s A
  := repeat' one.
Definition zeros {r} (s : Shape r) {A} {zero : has_zero A} : tensor s A
  := repeat' zero.

Definition raw_get {r} {s : Shape r} {A} (t : tensor s A) (idxs : RawIndex r) : A
  := t idxs.
Definition get {r} {s : Shape r} {A} (t : tensor s A) (idxs : Index r) : A
  := raw_get t (adjust_indices_for s idxs).
Definition item {A} (t : tensor [] A) : A := raw_get t tt.

Notation "x .[ y ]" := (get x y) : tensor_scope.
Notation "x .[ y ]" := (raw_get x y) : raw_tensor_scope.

Definition curried_raw_get {r} {s : Shape r} {A} (t : tensor s A) : @RawIndex.curriedT r A
  := RawIndex.curry (fun idxs => raw_get t idxs).
Definition curried_get {r} {s : Shape r} {A} (t : tensor s A) : @Index.curriedT r A
  := Index.curry (fun idxs => get t idxs).

Definition map {r} {s : Shape r} {A B} (f : A -> B) (t : tensor s A) : tensor s B
  := fun i => f (t i).
Definition map2 {r} {sA sB : Shape r} {A B C} (f : A -> B -> C) (tA : tensor sA A) (tB : tensor sB B) : tensor (Shape.broadcast2 sA sB) C
  := fun i => f (tA i) (tB i).
Definition map3 {r} {sA sB sC : Shape r} {A B C D} (f : A -> B -> C -> D) (tA : tensor sA A) (tB : tensor sB B) (tC : tensor sC C) : tensor (Shape.broadcast3 sA sB sC) D
  := fun i => f (tA i) (tB i) (tC i).

Definition map_dep {r} {s : Shape r} {A B} (f : forall a : A, B a) (t : tensor s A) : tensor_dep B t
  := fun i => f (t i).


Definition where_ {r} {sA sB sC : Shape r} {A} (condition : tensor sA bool) (input : tensor sB A) (other : tensor sC A) : tensor (Shape.broadcast3 sA sB sC) A
  := map3 Bool.where_ condition input other.

(* TODO: autobroadcast initial *)
#[export] Instance tensor_add {r} {sA sB : Shape r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor sA A) (tensor sB B) (tensor (Shape.broadcast2 sA sB) C) := map2 add.
#[export] Instance tensor_sub {r} {sA sB : Shape r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor sA A) (tensor sB B) (tensor (Shape.broadcast2 sA sB) C) := map2 sub.
#[export] Instance tensor_mul {r} {sA sB : Shape r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor sA A) (tensor sB B) (tensor (Shape.broadcast2 sA sB) C) := map2 mul.
#[export] Instance tensor_div_by {r} {sA sB : Shape r} {A B C} {div_byAB : has_div_by A B C} : has_div_by (tensor sA A) (tensor sB B) (tensor (Shape.broadcast2 sA sB) C) := map2 div.
#[export] Instance tensor_sqrt {r} {s : Shape r} {A} {sqrtA : has_sqrt A} : has_sqrt (tensor s A) := map sqrt.
#[export] Instance tensor_opp {r} {s : Shape r} {A} {oppA : has_opp A} : has_opp (tensor s A) := map opp.
#[export] Instance add'1 {r} {s : Shape r} {a b} {A B C} {addA : has_add_with A B C} : has_add_with (tensor (s ::' a) A) (tensor (s ::' b) B) (tensor (s ::' max a b) C) | 10 := tensor_add.
#[export] Instance sub'1 {r} {s : Shape r} {a b} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor (s ::' a) A) (tensor (s ::' b) B) (tensor (s ::' max a b) C) | 10 := tensor_sub.
#[export] Instance mul'1 {r} {s : Shape r} {a b} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor (s ::' a) A) (tensor (s ::' b) B) (tensor (s ::' max a b) C) | 10 := tensor_mul.
#[export] Instance div_by'1 {r} {s : Shape r} {a b} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor (s ::' a) A) (tensor (s ::' b) B) (tensor (s ::' max a b) C) | 10 := tensor_div_by.
#[export] Instance add'1s_r {r} {s : Shape r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor s A) (tensor (@Shape.ones r) B) (tensor s C) | 10 := tensor_add.
#[export] Instance add'1s_l {r} {s : Shape r} {A B C} {addA : has_add_with A B C} : has_add_with (tensor (@Shape.ones r) A) (tensor s B) (tensor s C) | 10 := tensor_add.
#[export] Instance sub'1s_r {r} {s : Shape r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor s A) (tensor (@Shape.ones r) B) (tensor s C) | 10 := tensor_sub.
#[export] Instance sub'1s_l {r} {s : Shape r} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor (@Shape.ones r) A) (tensor s B) (tensor s C) | 10 := tensor_sub.
#[export] Instance mul'1s_r {r} {s : Shape r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor s A) (tensor (@Shape.ones r) B) (tensor s C) | 10 := tensor_mul.
#[export] Instance mul'1s_l {r} {s : Shape r} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor (@Shape.ones r) A) (tensor s B) (tensor s C) | 10 := tensor_mul.
#[export] Instance div_by'1s_r {r} {s : Shape r} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor s A) (tensor (@Shape.ones r) B) (tensor s C) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l {r} {s : Shape r} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor (@Shape.ones r) A) (tensor s B) (tensor s C) | 10 := tensor_div_by.
#[export] Instance add'1s_r'1_same {r} {s : Shape r} {a} {A B C} {addA : has_add_with A B C} : has_add_with (tensor (s ::' a) A) (tensor (@Shape.ones r ::' a) B) (tensor (s ::' a) C) | 10 := tensor_add.
#[export] Instance add'1s_l'1_same {r} {s : Shape r} {a} {A B C} {addA : has_add_with A B C} : has_add_with (tensor (@Shape.ones r ::' a) A) (tensor (s ::' a) B) (tensor (s ::' a) C) | 10 := tensor_add.
#[export] Instance sub'1s_r'1_same {r} {s : Shape r} {a} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor (s ::' a) A) (tensor (@Shape.ones r ::' a) B) (tensor (s ::' a) C) | 10 := tensor_sub.
#[export] Instance sub'1s_l'1_same {r} {s : Shape r} {a} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor (@Shape.ones r ::' a) A) (tensor (s ::' a) B) (tensor (s ::' a) C) | 10 := tensor_sub.
#[export] Instance mul'1s_r'1_same {r} {s : Shape r} {a} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor (s ::' a) A) (tensor (@Shape.ones r ::' a) B) (tensor (s ::' a) C) | 10 := tensor_mul.
#[export] Instance mul'1s_l'1_same {r} {s : Shape r} {a} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor (@Shape.ones r ::' a) A) (tensor (s ::' a) B) (tensor (s ::' a) C) | 10 := tensor_mul.
#[export] Instance div_by'1s_r'1_same {r} {s : Shape r} {a} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor (s ::' a) A) (tensor (@Shape.ones r ::' a) B) (tensor (s ::' a) C) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l'1_same {r} {s : Shape r} {a} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor (@Shape.ones r ::' a) A) (tensor (s ::' a) B) (tensor (s ::' a) C) | 10 := tensor_div_by.
#[export] Instance add'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {addA : has_add_with A B C} : has_add_with (tensor (s ++' s') A) (tensor (@Shape.ones r ++' s') B) (tensor (s ++' s') C) | 10 := tensor_add.
#[export] Instance add'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {addA : has_add_with A B C} : has_add_with (tensor (@Shape.ones r ++' s') A) (tensor (s ++' s') B) (tensor (s ++' s') C) | 10 := tensor_add.
#[export] Instance sub'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor (s ++' s') A) (tensor (@Shape.ones r ++' s') B) (tensor (s ++' s') C) | 10 := tensor_sub.
#[export] Instance sub'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {subA : has_sub_with A B C} : has_sub_with (tensor (@Shape.ones r ++' s') A) (tensor (s ++' s') B) (tensor (s ++' s') C) | 10 := tensor_sub.
#[export] Instance mul'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor (s ++' s') A) (tensor (@Shape.ones r ++' s') B) (tensor (s ++' s') C) | 10 := tensor_mul.
#[export] Instance mul'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {mulA : has_mul_with A B C} : has_mul_with (tensor (@Shape.ones r ++' s') A) (tensor (s ++' s') B) (tensor (s ++' s') C) | 10 := tensor_mul.
#[export] Instance div_by'1s_r'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor (s ++' s') A) (tensor (@Shape.ones r ++' s') B) (tensor (s ++' s') C) | 10 := tensor_div_by.
#[export] Instance div_by'1s_l'1_same_app {r r'} {s : Shape r} {s' : Shape r'} {A B C} {div_byA : has_div_by A B C} : has_div_by (tensor (@Shape.ones r ++' s') A) (tensor (s ++' s') B) (tensor (s ++' s') C) | 10 := tensor_div_by.

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

Definition reshape_app_split' {r1 r2 s1 s2 A} : @tensor (r1 +' r2) (s1 ++' s2) A -> tensor s1 (tensor s2 A)
  := RawIndex.curry_radd.
Definition reshape_app_combine' {r1 r2 s1 s2 A} : tensor s1 (tensor s2 A) -> @tensor (r1 +' r2) (s1 ++' s2) A
  := RawIndex.uncurry_radd.
(* infer s1 s2 from the conclusion *)
#[global] Arguments reshape_app_combine' & r1 r2 s1 s2 A _.
#[global] Arguments reshape_app_split' & r1 r2 s1 s2 A _.
Definition reshape_app_split {r1 r2 s1 s2 A} : @tensor (r1 +' r2) (s1 ++' s2) A -> tensor s1 (tensor s2 A)
  := reshape_app_split'.
Definition reshape_app_combine {r1 r2 s1 s2 A} : tensor s1 (tensor s2 A) -> @tensor (r1 +' r2) (s1 ++' s2) A
  := reshape_app_combine'.
Definition reshape_snoc_split {r s1 s2 A} : @tensor (r +' 1) (s1 ::' s2) A -> tensor s1 (tensor [s2] A)
  := RawIndex.curry_radd.
Definition reshape_snoc_combine {r s1 s2 A} : tensor s1 (tensor [s2] A) -> @tensor (r +' 1) (s1 ::' s2) A
  := RawIndex.uncurry_radd.
Definition uncurry {r} {s : Shape r} {A} : @RawIndex.curriedT r A -> tensor s A
  := RawIndex.uncurry.
Definition curry {r} {s : Shape r} {A} : tensor s A -> @RawIndex.curriedT r A
  := RawIndex.curry.

Definition map' {ra1 ra2 rb} {sa1 : Shape ra1} {sa2 : Shape ra2} {sb : Shape rb} {A B} (f : tensor sa2 A -> tensor sb B) (t : tensor (sa1 ++' sa2) A) : tensor (sa1 ++' sb) B
  := reshape_app_combine (map f (reshape_app_split t)).
Definition map2' {ri1 ri2 ro} {sA1 sB1 : Shape ri1} {sA2 sB2 : Shape ri2} {so : Shape ro} {A B C} (f : tensor sA2 A -> tensor sB2 B -> tensor so C) (tA : tensor (sA1 ++' sA2) A) (tB : tensor (sB1 ++' sB2) B) : tensor (Shape.broadcast2 sA1 sB1 ++' so) C
  := reshape_app_combine (map2 f (reshape_app_split tA) (reshape_app_split tB)).

Definition broadcast' {r : Rank} {A} (x : A) : tensor (@Shape.ones r) A
  := repeat' x.
Definition broadcast {r r'} {s : Shape r} {A} (x : tensor s A) : tensor (@Shape.ones r' ++' s) A
  := reshape_app_combine (broadcast' x).
Definition repeat {r r'} {s : Shape r} (s' : Shape r') {A} (x : tensor s A) : tensor (s' ++' s) A
  := reshape_app_combine (repeat' x (s:=s')).

Definition keepdim_gen {r} {s : Shape r} {A B} (f : A -> tensor s B) : A -> tensor ([1] ++' s) B
  := fun a => broadcast (f a).
Definition keepdim {A B} (f : A -> B) : A -> tensor [1] B := keepdim_gen (s:=[]) (fun a 'tt => f a).
#[local] Notation keepdimf := keepdim (only parsing).

Definition reduce_axis_m1' {r} {s1 : Shape r} {s2} {A B}
  (reduction : forall (start stop step : RawIndexType), (RawIndexType -> A) -> B)
  (t : tensor (s1 ::' s2) A)
  : tensor s1 B
  := map (fun v => reduction 0 s2 1 (fun i => raw_get v [i])) (reshape_snoc_split t).

Definition reduce_axis_m1 {r} {s1 : Shape r} {s2} {keepdim : with_default "keepdim" bool false} {A B}
  (reduction : forall (start stop step : RawIndexType), (RawIndexType -> A) -> B)
  : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) B
  := if keepdim
          return tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) B
     then fun t idxs => reduce_axis_m1' reduction t (RawIndex.hd idxs)
     else reduce_axis_m1' reduction.

Definition unsqueeze_dim_m1 {r} {s : Shape r} {A} (t : tensor s A) : tensor (s ::' 1) A
  := fun idxs => raw_get t (RawIndex.hd idxs).

Definition gather_dim_m1 {r} {ssinput ssindex : Shape r} {sinput' sindex'} {A}
  (sinput := (ssinput ::' sinput')%shape) (sindex := (ssindex ::' sindex')%shape)
  (input : tensor sinput A)
  (index : tensor sindex IndexType)
  : tensor sindex A
  := fun idx => raw_get input (RawIndex.hd idx ::' adjust_index_for sinput' (raw_get index idx))%raw_index.

Definition squeeze {r} {s : Shape r} {A} (t : tensor s A) : tensor (Shape.squeeze s) A
  := fun idx => raw_get t (RawIndex.unsqueeze idx).

Definition reshape_all {r} {s : Shape r} {A} (t : tensor s A) : tensor (Shape.reshape s) A
  := fun idx => raw_get t (RawIndex.unreshape s idx).
Definition unreshape_all {r} {s : Shape r} {A} (t : tensor (Shape.reshape s) A) : tensor s A
  := fun idx => raw_get t (RawIndex.reshape s idx).
(*
Definition reshape {A r1 r2} {s1 : Shape r1} (t : tensor s1 A) (s2 : Shape r2) : tensor s2 A
  := unreshape_m1 (reshape_m1 t : tensor (Shape.reshape s2) A).
 *)

Section reduce_axis_m1.
  Context {r} {s1 : Shape r} {s2 : ShapeType} {keepdim : with_default "keepdim" bool false}
    {A}
    {zeroA : has_zero A} {oneA : has_one A} {coerZ : has_coer Z A}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
    {maxA : has_max A} {minA : has_min A}
    {lebA : has_leb A} {ltbA : has_ltb A}.

  Definition sum_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) A
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.sum.
  Definition prod_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) A
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.prod.
  Definition max_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) A
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.max.
  Definition min_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) A
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.min.
  Definition argmax_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) RawIndexType
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.argmax.
  Definition argmin_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) RawIndexType
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.argmin.
  Definition mean_dim_m1 : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) A
    := reduce_axis_m1 (keepdim:=keepdim) Reduction.mean.
  Definition var_dim_m1 {correction : with_default "correction" Z 1%Z}
    : tensor (s1 ::' s2) A -> tensor (s1 ++' @Shape.keepdim keepdim) A
    := reduce_axis_m1 (keepdim:=keepdim) (Reduction.var (correction:=correction)).
End reduce_axis_m1.

Section reduce.
  Context {r} {s : Shape r}
    {A}
    {zeroA : has_zero A} {oneA : has_one A} {coerZ : has_coer Z A}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
    {maxA : has_max A} {minA : has_min A}
    {lebA : has_leb A} {ltbA : has_ltb A}.

  Definition sum (t : tensor s A) : tensor [] A
    := reduce_axis_m1 Reduction.sum (reshape_all t).
  Definition prod (t : tensor s A) : tensor [] A
    := reduce_axis_m1 Reduction.prod (reshape_all t).
  Definition max (t : tensor s A) : tensor [] A
    := reduce_axis_m1 Reduction.max (reshape_all t).
  Definition min (t : tensor s A) : tensor [] A
    := reduce_axis_m1 Reduction.min (reshape_all t).
  Definition mean (t : tensor s A) : tensor [] A
    := reduce_axis_m1 Reduction.mean (reshape_all t).
  Definition var {correction : with_default "correction" Z 1%Z} (t : tensor s A) : tensor [] A
    := reduce_axis_m1 (Reduction.var (correction:=correction)) (reshape_all t).
End reduce.

Definition softmax_dim_m1 {r} {s0 : Shape r} {s'}
  (s:=(s0 ::' s')%shape)
  {A B C}
  {addB : has_add B} {expA : has_exp_to A B} {zeroB : has_zero B} {divB : has_div_by B B C}
  {use_checkpoint : with_default "use_checkpoint" bool true}
  {defaultB : pointed B}
  (t : tensor s A) : tensor s C
  := (let exp_t : tensor s B := PArray.maybe_checkpoint (map exp t) in
      let sum_exp_t : tensor (s0 ::' 1) B := PArray.maybe_checkpoint (sum_dim_m1 (keepdim:=true) exp_t) in
      exp_t / sum_exp_t)%core.

Definition log_softmax_dim_m1 {r} {s0 : Shape r} {s'} (s:=(s0 ::' s')%shape)
  {A B C D}
  {addB : has_add B} {lnA : has_ln_to B C} {expA : has_exp_to A B} {zeroB : has_zero B} {subB : has_sub_with A C D}
  {use_checkpoint : with_default "use_checkpoint" bool true}
  {defaultB : pointed B}
  {defaultC : pointed C}
  (t : tensor s A) : tensor s D
  := (let exp_t : tensor s B := PArray.maybe_checkpoint (map exp t) in
      let sum_exp_t : tensor (s0 ::' 1) B := sum_dim_m1 (keepdim:=true) exp_t in
      let ln_sum_exp_t : tensor (s0 ::' 1) C := PArray.maybe_checkpoint (map ln sum_exp_t) in
      t - ln_sum_exp_t)%core.

Definition softmax {r} {s : Shape r}
  {A B C}
  {addB : has_add B} {expA : has_exp_to A B} {zeroB : has_zero B} {divB : has_div_by B B C}
  {use_checkpoint : with_default "use_checkpoint" bool true}
  {defaultB : pointed B}
  (t : tensor s A) : tensor s C
  := (let exp_t : tensor s B := PArray.maybe_checkpoint (map exp t) in
      let sum_exp_t : B := item (sum_dim_m1 (keepdim:=false) (reshape_all exp_t)) in
      exp_t / broadcast' sum_exp_t)%core.

Definition log_softmax {r} {s : Shape r}
  {A B C D}
  {addB : has_add B} {lnA : has_ln_to B C} {expA : has_exp_to A B} {zeroB : has_zero B} {subACD : has_sub_with A C D}
  {use_checkpoint : with_default "use_checkpoint" bool true}
  {defaultB : pointed B}
  (t : tensor s A) : tensor s D
  := (let exp_t : tensor s B := PArray.maybe_checkpoint (map exp t) in
      let sum_exp_t : B := item (sum_dim_m1 (keepdim:=false) (reshape_all exp_t)) in
      let ln_sum_exp_t : C := ln sum_exp_t in
      t - broadcast' ln_sum_exp_t)%core.

Definition to_bool {r} {s : Shape r} {A} {zero : has_zero A} {eqb : has_eqb A} (xs : tensor s A) : tensor s bool
  := map (fun x => x ≠? 0)%core xs.

Definition of_bool {r} {s : Shape r} {A} {zero : has_zero A} {one : has_one A} (xs : tensor s bool) : tensor s A
  := map (fun x:bool => if x then 1 else 0)%core xs.

(*
Definition arange {A B} {START STOP STEP IDX} {oneA : has_one A} {zeroStart : has_zero START} {oneStep : has_one STEP} {sub : has_sub_with STOP START A} {subA : has_sub A} {div : has_int_div_by A STEP B} {coerZ : has_coer B Z} {coerIDX : has_coer int IDX} {add : has_add_with START C D} {mul : has_mul_with STEP IDX C}
  {start : with_default "start" START 0%core} (stop : STOP) {step : with_default "step" STEP 1%core}
  : tensor int [(1 + Uint63.of_Z (((stop - start) - 1) // step))%core%uint63]
  := fun idx => let idx := RawIndex.item idx in
                (start + idx * step)%uint63.
*)
Definition arange {start : with_default "start" int 0%uint63} (stop : int) {step : with_default "step" int 1%uint63}
  : tensor [(1 + (stop - start - 1) / step)%uint63] int
  := fun idx => let idx := RawIndex.item idx in
                (start + idx * step)%uint63.

#[global] Arguments arange (_ _ _)%uint63.
#[global] Arguments arange {_} _ {_}, _ _ {_}, _ _ _.

(* TODO: nary *)
Definition tupleify {s1 s2 A B} (t1 : tensor [s1] A) (t2 : tensor [s2] B) : tensor [s1; s2] (A * B)
  := fun '((tt, a), b) => (raw_get t1 [a], raw_get t2 [b]).
Definition cartesian_prod {s1 s2 A} (t1 : tensor [s1] A) (t2 : tensor [s2] A) : tensor [s1 * s2; 2] A
  := fun '((tt, idx), tuple_idx)
     => let '(a, b) := raw_get (reshape_all (tupleify t1 t2)) [idx] in
        nth_default a [a; b] (Z.to_nat (Uint63.to_Z (tuple_idx mod 2))).
Fixpoint ntupleify' {r} : forall {s : Shape r} {A B1 B2},
    (B1 -> B2)
    -> Shape.fold_map (fun s => tensor [s] A) (fun x y => (y * x)%type) B1 s
    -> tensor s (Shape.fold_map (fun _ => A) (fun x y => (y * x)%type) B2 s)
  := match r with
     | O => fun s A B1 B2 f ts i => f ts
     | S r
       => fun s A B1 B2 f ts i
          => let f := (fun tab1 => let '(ta, b1) := (fst tab1, snd tab1) in (raw_get ta [RawIndex.tl i], f b1)) in
             @ntupleify' r (Shape.hd s) A (tensor [Shape.tl s] A * B1)%type (A * B2)%type f ts (RawIndex.hd i)
     end.
Definition ntupleify {r} {s : Shape r} {A} (ts : Shape.tuple (fun s => tensor [s] A) s) : tensor s (Shape.tuple (fun _ => A) s)
  := ntupleify' (fun _tt => tt) ts.
Definition cartesian_nprod {r} {s : Shape r} {A} {defaultA : pointed A} (ts : Shape.tuple (fun s => tensor [s] A) s) : tensor [Shape.fold_map id mul 1 s; Uint63.of_Z r] A
  := fun '((tt, idx), tuple_idx)
     => let ts := raw_get (reshape_all (ntupleify ts)) [idx] in
        Shape.Tuple.nth_default point ts (Z.to_nat (Uint63.to_Z (tuple_idx mod Uint63.of_Z r))).
Definition cartesian_exp {s A} {defaultA : pointed A} (t : tensor [s] A) (n : ShapeType) : tensor [(s^(n:N))%core; n] A
  := @cartesian_nprod n (Shape.repeat s n) A _ (Shape.Tuple.init (fun _ => t)).

Definition dot {s A B C} {mulA : has_mul_with A B C} {addC : has_add C} {zeroC : has_zero C}
  (x : tensor [s] A) (y : tensor [s] B)
  : tensor [] C
  := fun _ => ∑_(0 ≤ i < s) (x.[[i]] * y.[[i]])%raw_tensor.

Definition mm {n m p A B C} {mulA : has_mul_with A B C} {addC : has_add C} {zeroC : has_zero C}
  (x : tensor [n; m] A) (y : tensor [m; p] B)
  : tensor [n; p] C
  := fun '((tt, r), c) => ∑_(0 ≤ i < m) (x.[[r; i]] * y.[[i; c]])%raw_tensor.

(** transpose last two dimensions *)
Definition T {r} {s : Shape r} {n m A} (t : tensor (s ::' n ::' m) A) : tensor (s ::' m ::' n) A
  := fun '((idxs, i), j) => raw_get t ((idxs, j), i).

#[export] Instance dotmatvec {r} {s : Shape r} {n m A B C} {mulA : has_mul_with A B C} {addC : has_add C} {zeroC : has_zero C}
  : has_matmul (tensor (s ::' n ::' m) A) (tensor [m] B) (tensor (s ::' m) C)
  := fun x y '(idx, i) => ∑_(0 ≤ j < n) (x.[idx ::' j ::' i] * y.[[i]])%raw_tensor.
#[export] Instance dotvecmat {r} {s : Shape r} {n m A B C} {mulA : has_mul_with A B C} {addC : has_add C} {zeroC : has_zero C}
  : has_matmul (tensor [n] A) (tensor (s ::' n ::' m) B) (tensor (s ::' m) C)
  := fun x y '(idx, i) => ∑_(0 ≤ j < n) (x.[[j]] * y.[idx ::' j ::' i])%raw_tensor.
#[export] Instance tensor_matmul {r} {sA sB : Shape r} {n m p A B C} {mulA : has_mul_with A B C} {addC : has_add C} {zeroC : has_zero C}
  : has_matmul (tensor (sA ::' n ::' m) A) (tensor (sB ::' m ::' p) B) (tensor (Shape.broadcast2 sA sB ::' n ::' p) C)
  := map2' mm.

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
Definition tril {rnk} {s : Shape rnk} {r c} {A} {zero : has_zero A}
  {diagonal : with_default "diagonal" int 0%int63} (input : tensor (s ++' [r; c]) A)
  : tensor (s ++' [r; c]) A
  := fun '(((_, i), j) as idxs)
     => if ((0 ≤? i) && (i <? r) && (Sint63.max 0 (1 + i + diagonal) ≤? j) && (j <? c))%bool
        then 0%core
        else input idxs.
#[global] Arguments tril {rnk%nat s%shape} {r c}%uint63 {A%type_scope zero} {diagonal}%sint63 input%tensor.
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
Definition triu {rnk} {s : Shape rnk} {r c} {A} {zero : has_zero A}
  {diagonal : with_default "diagonal" int 0%int63} (input : tensor (s ++' [r; c]) A)
  : tensor (s ++' [r; c]) A
  := fun '(((_, i), j) as idxs)
     => if ((0 ≤? i) && (i <? r) && (0 ≤? j) && (j <? Sint63.max 0 (i + diagonal)))%bool
        then 0%core
        else input idxs.
#[global] Arguments triu {rnk%nat s%shape} {r c}%uint63 {A%type_scope zero} {diagonal}%sint63 input%tensor.

(** Quoting
https://pytorch.org/docs/stable/generated/torch.diagonal.html

torch.diagonal(input, offset=0, dim1=0, dim2=1) → Tensor

Returns a partial view of input with the its diagonal elements with
respect to dim1 and dim2 appended as a dimension at the end of the
shape.

The argument offset controls which diagonal to consider:

- If [offset = 0], it is the main diagonal.

- If [offset > 0], it is above the main diagonal.

- If [offset < 0], it is below the main diagonal.

 *)
(** N.B. we compute a batch diagonal (dim1=-2, dim2=-1) *)
Definition diagonal {b} {s : Shape b} {r c} {A}
  {offset : with_default "offset" int 0%int63}
  (input : tensor (s ++' [r; c]) A)
  : tensor (s ::' Uint63.min (r - Uint63.min r (Sint63.abs offset)) (c - Uint63.min c (Sint63.abs offset))) A
  := if (offset <=? 0)%sint63
     then fun '(idxs, i)
          => input ((idxs, i - offset), i)%uint63
     else fun '(idxs, i)
          => input ((idxs, i), i + offset)%uint63.
#[global] Arguments diagonal {b%nat s%shape} {r c}%uint63 {A%type_scope} {offset}%sint63 input%tensor.

Definition coer_tensor {r s A B} {coerAB : has_coer A B} : @tensor r s A -> @tensor r s B
  := Tensor.map coer.
#[export] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
#[local] Set Warnings Append "-unsupported-attributes".
#[export] Coercion coer_tensor : tensor >-> tensor.
#[local] Set Warnings Append "unsupported-attributes".
#[export] Set Warnings Append "uniform-inheritance,ambiguous-paths".
