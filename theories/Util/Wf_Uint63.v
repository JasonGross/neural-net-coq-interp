From Coq Require Import Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Default.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead.
Import Instances.Uint63.
(*
Set Universe Polymorphism.
Set Polymorphic Inductive Cumulativity.
Unset Universe Minimization ToSet.
*)
Local Open Scope uint63_scope.

#[local] Coercion is_true : bool >-> Sortclass.
Definition ltof {A} (f : A -> int) (a b : A) := f a <? f b.

Lemma well_founded_ltof {A f} : well_founded (@ltof A f).
Proof.
  unshelve eapply well_founded_lt_compat with (fun x:A => f x:nat); cbv [is_true ltof coer_int_N' coer coer_int_N coer_int_Z].
  intros *; rewrite Uint63.ltb_spec, Z2Nat.inj_lt, !Z_N_nat by apply to_Z_bounded; trivial.
Qed.

Lemma lt_wf : well_founded Uint63.ltb.
Proof.
  apply @well_founded_ltof with (f:=fun x => x).
Qed.

Lemma well_founded_gtof {A f} {bound} : well_founded (fun x y:A => (f y <? f x) && (f x <=? bound)).
Proof.
  apply @well_founded_lt_compat with (f:=fun x:A => bound - f x); cbv [is_true ltof coer_int_N'].
  intros x y.
  pose proof (to_Z_bounded (f y)).
  pose proof (to_Z_bounded (f x)).
  pose proof (to_Z_bounded bound).
  rewrite andb_true_iff, Uint63.ltb_spec, Uint63.leb_spec, !Z_N_nat, !Uint63.sub_spec by apply to_Z_bounded.
  intros; rewrite !Z.mod_small by lia; lia.
Qed.

Lemma gt_wf {bound} : well_founded (fun x y => (y <? x) && (x <=? bound)).
Proof.
  apply @well_founded_gtof with (f:=fun x => x).
Qed.

#[global] Arguments gt_wf {_}, _.

Inductive LoopBody_ S A : Type :=
| break (v : S) : LoopBody_ S A
| continue (v : S) : LoopBody_ S A
| ret (v : A) (st : S) : LoopBody_ S A
.
#[global] Arguments break {_ _}.
#[global] Arguments continue {_ _}.
#[global] Arguments ret {_ _}.

Definition LoopBody S A := S -> LoopBody_ S A.
Definition bind {S A B} (x : LoopBody S A) (k : A -> LoopBody S B) : LoopBody S B
  := fun st => match x st with
               | break st => break st
               | continue st => continue st
               | ret x st => k x st
               end.
Definition get {S} : LoopBody S S := fun st => ret st st.
Definition update {S} (v : S -> S) : LoopBody S unit := fun st => ret tt (v st).
Definition set {S} (v : S) : LoopBody S unit := update (fun _ => v).
#[export] Instance LoopBody_Monad {S} : Monad (LoopBody S) := { ret := @ret S ; bind := @bind S }.

Definition run_body {S}
  (v : LoopBody S unit)
  {T}
  (breakf : S -> T) (continuef : S -> T)
  (st : S)
  : T
  := match v st with
     | break v => breakf v
     | continue v => continuef v
     | ret tt v => continuef v
     end.

Definition for_loop_lt {A} (i : int) (max : int) (step : int)
  (body : int -> LoopBody A unit)
  (init : A)
  : A.
Proof.
  refine (let step := if (step =? 0) then 1 else step in
          Fix
            (Acc_intro_generator Uint63.size (@gt_wf max))
            (fun _ => _)
            (fun i continue state
             => if Sumbool.sumbool_of_bool (i <? max)
                then
                  let break := fun v => v in
                  let continue := if Sumbool.sumbool_of_bool (step <? (max - i))
                                  then (fun v => continue (i + step) _ v)
                                  else (fun v => v) in
                  run_body (body i) break continue state
                else state)
            i
            init).
  { abstract
      (cbv [is_true];
       pose proof (Uint63.to_Z_bounded max);
       pose proof (Uint63.to_Z_bounded i);
       pose proof (Uint63.to_Z_bounded step);
       assert (0 < Uint63.to_Z step)%Z
         by (repeat match goal with H : context[step] |- _ => revert H end;
             subst step;
             case Uint63.eqbP; rewrite Uint63.to_Z_0, ?Uint63.to_Z_1; intros; lia);
       rewrite andb_true_iff;
       rewrite Uint63.ltb_spec, Uint63.leb_spec, Uint63.sub_spec, Uint63.add_spec in *;
       rewrite !Z.mod_small in * by lia;
       lia). }
Defined.

Definition LoopBody__R {S S' A A'} (RS : S -> S' -> Prop) (RA : A -> A' -> Prop)
  : LoopBody_ S A -> LoopBody_ S' A' -> Prop
  := fun x y
     => match x, y with
        | break stx, break sty
        | continue stx, continue sty
          => RS stx sty
        | ret ax stx, ret ay sty
          => RA ax ay /\ RS stx sty
        | break _, _
        | continue _, _
        | ret _ _, _
          => False
        end.
Definition LoopBody_R {S S' A A'} RS RA : LoopBody S A -> LoopBody S' A' -> Prop
  := respectful_hetero _ _ _ _ RS (fun _ _ => @LoopBody__R S S' A A' RS RA).

Lemma for_loop_lt_Proper_gen_hetero
  {A B R Runit start stop step f g}
  (Hfg : forall x, LoopBody_R R Runit (f x) (g x))
  : (respectful_hetero _ _ _ _ R (fun _ _ => R))
      (@for_loop_lt A start stop step f)
      (@for_loop_lt B start stop step g).
Proof.
  cbv [respectful_hetero for_loop_lt Fix].
  set (wf := Acc_intro_generator _ _ _); clearbody wf.
  revert start wf.
  fix IH 2.
  intros i wf init init' Hinit.
  destruct wf as [wf].
  cbn [Fix_F Acc_inv].
  set (wf' := wf _).
  specialize (fun v => IH _ (wf' v)).
  clearbody wf'.
  cbv [run_body].
  do 2 (destruct Sumbool.sumbool_of_bool; try assumption).
  all: lazymatch goal with
       | [ H : forall x : int, LoopBody_R _ _ (?f x) (?g x) |- context[?f ?i ?init] ]
         => lazymatch goal with
            | [ |- context[g i ?init'] ]
              => specialize (H i init init' ltac:(assumption));
                 cbv [LoopBody__R] in H; destruct (f i init) eqn:?, (g i init') eqn:?
            end
       end.
  all: destruct_head'_unit.
  all: destruct_head'_and.
  all: destruct_head'_False.
  all: try assumption.
  all: now apply IH.
Qed.

#[export] Instance for_loop_lt_Proper_gen {A R Runit} : Proper (eq ==> eq ==> eq ==> (pointwise_relation _ (LoopBody_R R Runit)) ==> R ==> R) (@for_loop_lt A).
Proof. repeat intro; subst; eapply (@for_loop_lt_Proper_gen_hetero A A); eassumption. Qed.

#[export] Instance for_loop_lt_Proper {A} : Proper (eq ==> eq ==> eq ==> (pointwise_relation _ (pointwise_relation _ eq)) ==> eq ==> eq) (@for_loop_lt A).
Proof.
  generalize (@for_loop_lt_Proper_gen A eq eq).
  repeat (let x := fresh "x" in intros H x; specialize (H x); revert H).
  intro H; repeat intro; apply H; clear H; try assumption; [].
  let H := match goal with H : pointwise_relation _ _ _ _ |- _ => H end in
  revert H.
  repeat (let x := fresh "x" in intros H x; specialize (H x); revert H).
  intro H; intros; subst; rewrite H; hnf; break_innermost_match; repeat constructor.
Qed.

Module LoopNotationAlises.
  Notation break := break.
  Notation continue := continue.
  Notation get := get.
  Notation update := update.
  Notation set := set.
  Notation ret := Monad.ret (only parsing).
  Notation bind := Monad.bind (only parsing).
End LoopNotationAlises.

Module Import LoopNotation1.
  Export MonadNotation.
  #[export] Existing Instance LoopBody_Monad.

  Notation "'with_state' state 'for' ( x := init ;; x <? max ;; x += step ) {{ body }}"
    := (for_loop_lt init max step (fun x => body%monad) state)
         (x binder, init at level 49, max at level 49, step at level 49, body at level 200, only printing, format "'with_state'  '/' '[hv ' state ']'  '//' 'for'  ( x  :=  init ;;  x  <?  max ;;  x  +=  step ) '//' '[v  ' {{  '/' body ']' '//' }}").
  Notation "'with_state' state 'for' ( x := init ;; x <? max ;; x ++ ) {{ body }}"
    := (for_loop_lt init max 1 (fun x => body%monad) state)
         (x binder, init at level 49, max at level 49, body at level 200, only printing, format "'with_state'  '/' '[hv ' state ']'  '//' 'for'  ( x  :=  init ;;  x  <?  max ;;  x ++ )  '//' '[v  ' {{  '//' body ']' '//' }}").
  Notation "'with_state' state 'for' ( x := init ;; y <? max ;; z += step ) {{ body }}"
    := (match (fun x : unit => conj (eq_refl : x = y) (eq_refl : x = z)) return _ with
        | _
          => for_loop_lt init max step (fun x => body%monad) state
        end)
         (only parsing, x binder, init at level 49, max at level 49, step at level 49, y at level 49, z at level 49, body at level 200).
  Notation "'with_state' state 'for' ( x := init ;; y <? max ;; z ++ ) {{ body }}"
    := (match (fun x : unit => conj (eq_refl : x = y) (eq_refl : x = z)) return _ with
        | _
          => for_loop_lt init max 1 (fun x => body%monad) state
        end)
         (only parsing, x binder, init at level 49, max at level 49, y at level 49, z at level 49, body at level 200).
  (*
  Check with_state 0 for (x := 0;; x <? 10;; x++) {{ y <- get;; set (y+x) }}.
   *)

End LoopNotation1.

Definition map_reduce {A B} (reduce : B -> A -> B) (init : B) (start : int) (stop : int) (step : int) (f : int -> A) : B
  := (with_state init
        for (i := start;; i <? stop;; i += step) {{
            val <-- get;;
            set (reduce val (f i))
     }})%core.

Definition map_reduce_no_init {A} (reduce : A -> A -> A) (start : int) (stop : int) (step : int) (f : int -> A) : A
  := (let step' := if (step =? 0) then 1 else step in
      with_state (f start)
        for (i := (start + step');; i <? stop;; i += step) {{
            val <-- get;;
            set (reduce val (f i))
     }})%core.

Lemma map_reduce_ext {A B A' B'} (RA RB : _ -> _ -> Prop) {reduce reduce'}
  (Hreduce : forall b b', RB b b' -> forall a a', RA a a' -> RB (reduce b a) (reduce' b' a'))
  {init init'} (Hinit : RB init init')
  {start start'} (Hstart : start = start')
  {stop stop'} (Hstop : stop = stop')
  {step step'} (Hstep : step = step')
  {f f'} (Hfg : forall i, RA (f i) (f' i))
  : RB (@map_reduce A B reduce init start stop step f) (@map_reduce A' B' reduce' init' start' stop' step' f').
Proof.
  cbv [map_reduce]; subst; eapply @for_loop_lt_Proper_gen_hetero with (Runit:=fun _ _ => True); try assumption; [].
  repeat intro; cbv; repeat split; auto.
Qed.

Lemma map_reduce_no_init_ext {A A'} (R : A -> A' -> Prop) {reduce reduce'}
  (Hreduce : forall b b', R b b' -> forall a a', R a a' -> R (reduce b a) (reduce' b' a'))
  {start start'} (Hstart : start = start')
  {stop stop'} (Hstop : stop = stop')
  {step step'} (Hstep : step = step')
  {f f'} (Hfg : forall i, R (f i) (f' i))
  : R (@map_reduce_no_init A reduce start stop step f) (@map_reduce_no_init A' reduce' start' stop' step' f').
Proof.
  cbv [map_reduce_no_init]; subst; eapply @for_loop_lt_Proper_gen_hetero with (Runit:=fun _ _ => True); try assumption; auto; [].
  repeat intro; cbv; repeat split; auto.
Qed.

Definition argmin_ {A B} {lebB : has_leb B} (x y : A * B) : A * B
  := if (snd x <=? snd y)%core then x else y.
Definition argmax_ {A B} {ltbB : has_ltb B} (x y : A * B) : A * B
  := if (snd x <? snd y)%core then y else x.

Module Import Reduction.
  Definition sum {A} {zeroA : has_zero A} {addA : has_add A} (start : int) (stop : int) (step : int) (f : int -> A) : A
    := map_reduce add zero start stop step f.
  Definition prod {A} {oneA : has_one A} {mulA : has_mul A} (start : int) (stop : int) (step : int) (f : int -> A) : A
    := map_reduce mul one start stop step f.
  Definition max {A} {maxA : has_max A} (start : int) (stop : int) (step : int) (f : int -> A) : A
    := map_reduce_no_init Classes.max start stop step f.
  Definition min {A} {minA : has_min A} (start : int) (stop : int) (step : int) (f : int -> A) : A
    := map_reduce_no_init Classes.min start stop step f.
  Definition argmin {A} {lebA : has_leb A} (start : int) (stop : int) (step : int) (f : int -> A) : int
    := fst (map_reduce_no_init argmin_ start stop step (fun i => (i, f i))).
  Definition argmax {A} {ltbA : has_ltb A} (start : int) (stop : int) (step : int) (f : int -> A) : int
    := fst (map_reduce_no_init argmax_ start stop step (fun i => (i, f i))).

  Module Import LoopNotation2.
    Notation "\sum_ ( m <= i < n ) F" := (sum m n 1 (fun i => F%core)).
    Notation "\sum_ ( m ≤ i < n ) F" := (sum m n 1 (fun i => F%core)).
    Notation "∑_ ( m <= i < n ) F" := (sum m n 1 (fun i => F%core)).
    Notation "∑_ ( m ≤ i < n ) F" := (sum m n 1 (fun i => F%core)).
    Notation "\prod_ ( m <= i < n ) F" := (prod m n 1 (fun i => F%core)).
    Notation "\prod_ ( m ≤ i < n ) F" := (prod m n 1 (fun i => F%core)).
    Notation "∏_ ( m <= i < n ) F" := (prod m n 1 (fun i => F%core)).
    Notation "∏_ ( m ≤ i < n ) F" := (prod m n 1 (fun i => F%core)).
  End LoopNotation2.

  Definition mean {A B C} {zeroA : has_zero A} {addA : has_add A} {div_by : has_div_by A B C} {coerB : has_coer Z B} (start : int) (stop : int) (step : int) (f : int -> A) : C
    := (sum start stop step f / coer (Uint63.to_Z (1 + (stop - start - 1) // step)))%core.
  Definition var {A B} {zeroA : has_zero A} {addA : has_add A} {mulA : has_mul A} {subA : has_sub A} {div_by : has_div_by A B A} {coerB : has_coer Z B} {correction : with_default "correction" Z 1%Z}
    (start : int) (stop : int) (step : int) (f : int -> A) : A
    := (let xbar := mean start stop step f in
        let N := Uint63.to_Z (1 + (stop - start - 1) // step) in
        (sum start stop step (fun i => (f i - xbar)²) / (coer (N - correction)))%core).

  Lemma sum_ext
    {A} {zeroA : has_zero A} {addA : has_add A}
    {B} {zeroB : has_zero B} {addB : has_add B}
    {start stop step : int}
    (R : A -> B -> Prop)
    {f f'} (Hf : forall i, R (f i) (f' i))
    (Rzero : R zero zero)
    (Radd : (respectful_hetero _ _ _ _ R (fun _ _ => respectful_hetero _ _ _ _ R (fun _ _ => R)))
              add
              add)
    : R (sum start stop step f) (sum start stop step f').
  Proof.
    cbv [sum]; eapply @map_reduce_ext with (RA:=R) (RB:=R); eauto.
  Qed.

  Lemma sum_equiv
    {A B} (F : A -> B)
    {zeroA : has_zero A} {addA : has_add A}
    {zeroB : has_zero B} {addB : has_add B}
    (Radd : forall x y, F (add x y) = add (F x) (F y))
    (Rzero : F zero = zero)
    {start stop step f}
    : F (sum start stop step f) = sum start stop step (fun x => F (f x)).
  Proof.
    apply @sum_ext with (R:=fun x y => F x = y); eauto; repeat intro; subst; auto.
  Qed.

  Lemma prod_ext
    {A} {oneA : has_one A} {mulA : has_mul A}
    {B} {oneB : has_one B} {mulB : has_mul B}
    {start stop step : int}
    (R : A -> B -> Prop)
    {f f'} (Hf : forall i, R (f i) (f' i))
    (Rone : R one one)
    (Rmul : (respectful_hetero _ _ _ _ R (fun _ _ => respectful_hetero _ _ _ _ R (fun _ _ => R)))
              mul
              mul)
    : R (prod start stop step f) (prod start stop step f').
  Proof.
    cbv [prod]; eapply @map_reduce_ext with (RA:=R) (RB:=R); eauto.
  Qed.

  Lemma prod_equiv
    {A B} (F : A -> B)
    {oneA : has_one A} {mulA : has_mul A}
    {oneB : has_one B} {mulB : has_mul B}
    (Rmul : forall x y, F (mul x y) = mul (F x) (F y))
    (Rone : F one = one)
    {start stop step f}
    : F (prod start stop step f) = prod start stop step (fun x => F (f x)).
  Proof.
    apply @prod_ext with (R:=fun x y => F x = y); eauto; repeat intro; subst; auto.
  Qed.

  Lemma max_ext
    {A} {maxA : has_max A}
    {B} {maxB : has_max B}
    {start stop step : int}
    (R : A -> B -> Prop)
    {f f'} (Hf : forall i, R (f i) (f' i))
    (Rmax : (respectful_hetero _ _ _ _ R (fun _ _ => respectful_hetero _ _ _ _ R (fun _ _ => R)))
              Classes.max
              Classes.max)
    : R (max start stop step f) (max start stop step f').
  Proof.
    cbv [max]; eapply @map_reduce_no_init_ext with (R:=R); eauto.
  Qed.

  Lemma max_equiv
    {A B} (F : A -> B)
    {maxA : has_max A}
    {maxB : has_max B}
    (Rmax : forall x y, F (Classes.max x y) = Classes.max (F x) (F y))
    {start stop step f}
    : F (max start stop step f) = max start stop step (fun x => F (f x)).
  Proof.
    apply @max_ext with (R:=fun x y => F x = y); eauto; repeat intro; subst; auto.
  Qed.

  Lemma min_ext
    {A} {minA : has_min A}
    {B} {minB : has_min B}
    {start stop step : int}
    (R : A -> B -> Prop)
    {f f'} (Hf : forall i, R (f i) (f' i))
    (Rmin : (respectful_hetero _ _ _ _ R (fun _ _ => respectful_hetero _ _ _ _ R (fun _ _ => R)))
              Classes.min
              Classes.min)
    : R (min start stop step f) (min start stop step f').
  Proof.
    cbv [min]; eapply @map_reduce_no_init_ext with (R:=R); eauto.
  Qed.

  Lemma min_equiv
    {A B} (F : A -> B)
    {minA : has_min A}
    {minB : has_min B}
    (Rmin : forall x y, F (Classes.min x y) = Classes.min (F x) (F y))
    {start stop step f}
    : F (min start stop step f) = min start stop step (fun x => F (f x)).
  Proof.
    apply @min_ext with (R:=fun x y => F x = y); eauto; repeat intro; subst; auto.
  Qed.

  Lemma mean_ext
    {A B C} {zeroA : has_zero A} {addA : has_add A} {div_by : has_div_by A B C} {coerB : has_coer Z B}
    {A' B' C'} {zeroA' : has_zero A'} {addA' : has_add A'} {div_by' : has_div_by A' B' C'} {coerB' : has_coer Z B'}
    (RA : A -> A' -> Prop)
    (RB : B -> B' -> Prop)
    (RC : C -> C' -> Prop)
    {start stop step}
    {f f'} (Hf : forall i, RA (f i) (f' i))
    (Rzero : RA zero zero)
    (Radd : (respectful_hetero _ _ _ _ RA (fun _ _ => respectful_hetero _ _ _ _ RA (fun _ _ => RA)))
              add
              add)
    (Rdiv : (respectful_hetero _ _ _ _ RA (fun _ _ => respectful_hetero _ _ _ _ RB (fun _ _ => RC)))
              div
              div)
    (Rcoer : forall x : Z, RB (coer x) (coer x))
    : RC (mean start stop step f) (mean start stop step f').
  Proof. cbv [mean]; apply Rdiv; [ eapply @sum_ext | ]; auto. Qed.

  Lemma mean_equiv
    {A B} (F : A -> B)
    {zeroA : has_zero A} {addA : has_add A} {divA : has_div A} {coerA : has_coer Z A}
    {zeroB : has_zero B} {addB : has_add B} {divB : has_div B} {coerB : has_coer Z B}
    {correction : with_default "correction" Z 1%Z}
    (Radd : forall x y, F (add x y) = add (F x) (F y))
    (Rdiv : forall x y, F (div x y) = div (F x) (F y))
    (Rcoer : forall x : Z, F (coer x) = coer x)
    (Rzero : F zero = zero)
    {start stop step f}
    : F (mean start stop step f) = mean start stop step (fun x => F (f x)).
  Proof.
    apply @mean_ext with (RA:=fun x y => F x = y) (RB:=fun x y => F x = y) (RC:=fun x y => F x = y); eauto; repeat intro; subst; auto.
  Qed.

  Lemma var_ext
    {A B} {zeroA : has_zero A} {addA : has_add A} {mulA : has_mul A} {subA : has_sub A} {div_by : has_div_by A B A} {coerB : has_coer Z B}
    {A' B'} {zeroA' : has_zero A'} {addA' : has_add A'} {mulA' : has_mul A'} {subA' : has_sub A'} {div_by' : has_div_by A' B' A'} {coerB' : has_coer Z B'}
    (RA : A -> A' -> Prop)
    (RB : B -> B' -> Prop)
    {start stop step}
    {f f'} (Hf : forall i, RA (f i) (f' i))
    (Rzero : RA zero zero)
    (Radd : (respectful_hetero _ _ _ _ RA (fun _ _ => respectful_hetero _ _ _ _ RA (fun _ _ => RA)))
              add
              add)
    (Rmul : (respectful_hetero _ _ _ _ RA (fun _ _ => respectful_hetero _ _ _ _ RA (fun _ _ => RA)))
              mul
              mul)
    (Rsub : (respectful_hetero _ _ _ _ RA (fun _ _ => respectful_hetero _ _ _ _ RA (fun _ _ => RA)))
              sub
              sub)
    (Rdiv : (respectful_hetero _ _ _ _ RA (fun _ _ => respectful_hetero _ _ _ _ RB (fun _ _ => RA)))
              div
              div)
    (Rcoer : forall x : Z, RB (coer x) (coer x))
    : RA (var start stop step f) (var start stop step f').
  Proof.
    cbv [var sqr]; apply Rdiv; [ eapply @sum_ext | ]; auto; intros.
    apply Rmul; apply Rsub; auto.
    all: apply @mean_ext with (RA:=RA) (RB:=RB) (RC:=RA); auto.
  Qed.

  Lemma var_equiv
    {A B} (F : A -> B)
    {zeroA : has_zero A} {addA : has_add A} {mulA : has_mul A} {subA : has_sub A} {divA : has_div A} {coerA : has_coer Z A}
    {zeroB : has_zero B} {addB : has_add B} {mulB : has_mul B} {subB : has_sub B} {divB : has_div B} {coerB : has_coer Z B}
    (Radd : forall x y, F (add x y) = add (F x) (F y))
    (Rmul : forall x y, F (mul x y) = mul (F x) (F y))
    (Rsub : forall x y, F (sub x y) = sub (F x) (F y))
    (Rdiv : forall x y, F (div x y) = div (F x) (F y))
    (Rcoer : forall x : Z, F (coer x) = coer x)
    (Rzero : F zero = zero)
    {start stop step f}
    : F (var start stop step f) = var start stop step (fun x => F (f x)).
  Proof.
    apply @var_ext with (RA:=fun x y => F x = y) (RB:=fun x y => F x = y); eauto; repeat intro; subst; auto.
  Qed.

  #[export] Hint Opaque sum prod max min argmin argmax mean var : rewrite.
End Reduction.
Export (hints) Reduction.

Module LoopNotation.
  Include LoopNotationAlises.
  Export LoopNotation1.
  Export LoopNotation2.
End LoopNotation.
