From Coq Require Import Zify ZifyUint63 Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Arith.Classes.Laws Arith.Instances.Laws Arith.Instances.Zify Default.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose.
From NeuralNetInterp.Util Require Import Wf_Uint63.
Import Arith.Classes Arith.Instances.Uint63.
#[local] Open Scope core_scope.
#[local] Coercion is_true : bool >-> Sortclass.

Module Reduction.
  Export Wf_Uint63.Reduction.

  Lemma for_loop_lt_invariant_gen {A i max} {step0:int} {body init}
    (step : int := if (step0 =? 0) then 1 else step0)
    (P : int -> A -> Prop)
    (Q : A -> Prop)
    (Hinit : P i init)
    (Hbody : forall i v continuef, P i v -> (i <? max) = true -> (step <? max - i) = true -> (forall init, P (i + step) init -> Q (continuef init)) -> Q (run_body (body i) (fun v => v) continuef v))
    (Hbody_fin : forall i v, P i v -> (i <? max) = true -> (step <? max - i) = false -> Q (run_body (body i) (fun v => v) (fun v => v) v))
    (Hfin : forall i v, P i v -> (i <? max) = false -> Q v)
    : Q (@for_loop_lt A i max step0 body init).
  Proof.
    cbv [for_loop_lt Fix].
    change ((if step0 =? 0 then 1 else step0)%uint63) with step.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    revert i wf init Hinit.
    fix IH 2; destruct wf as [wf]; intros.
    cbn [Fix_F Acc_inv]; specialize (fun x y => IH _ (wf x y)).
    break_innermost_match; eauto.
  Qed.

  Lemma map_reduce_no_init_spec_count {A reduce start stop step0 f} (P : nat -> int -> _ -> Prop)
    (step : int := if (step0 =? 0) then 1 else step0)
    (Hinit : P 1 (start + step) (f start))
    (Hstep : forall n i v, P (S n) i v -> (start + step <=? i) = true -> (i <? stop) = true -> i = start + step + step * Uint63.of_Z n -> P (S (S n)) (i + step) (reduce v (f i)))
    : P (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0))
        (start + step + step * (if start + step <? stop then 1 + (stop - (start + step) - 1) // step else 0))
        (@map_reduce_no_init A reduce start stop step0 f).
  Proof.
    cbv [map_reduce_no_init for_loop_lt Fix Classes.eqb int_has_eqb Classes.one Classes.zero int_has_one int_has_zero Classes.leb Classes.ltb int_has_leb int_has_leb int_has_ltb nat_has_zero nat_has_one Classes.add int_has_add] in *.
    fold step.
    set (f' := fun (i : int) continue state => _) at 1.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    set (start' := (start + step)%uint63) in *.
    generalize dependent (f start); intro init; intros.
    clearbody start'; clear start; rename start' into start.
    set (start0 := start) in Hstep.
    assert (Hstart : (start0 <=? start) = true) by (cbv; lia).
    set (n0 := O) in Hinit.
    assert (Hn : start = start0 + step * Uint63.of_Z n0) by (cbv; lia).
    change (P ?x) with (P (n0 + x)%nat).
    clearbody start0 n0.
    revert start wf init Hstart n0 Hn Hinit.
    fix IH 2.
    intros ? [wf]; intros; cbn [Fix_F]; cbv [Acc_inv].
    unfold f' at 1.
    set (wf' := wf _); specialize (fun pf y b => IH _ (wf' y) b pf); clearbody wf'; clear wf.
    cbv [Monad.bind run_body LoopNotation.get LoopBody_Monad bind LoopNotation.set LoopNotation.update].
    cbv [Classes.sub Classes.mul Classes.zero Classes.add Classes.ltb Classes.leb Classes.one Classes.int_div int_has_int_div int_has_sub int_has_ltb int_has_add int_has_mul int_has_leb] in *.
    break_match; break_innermost_match_hyps; auto.
    all: try (specialize (IH ltac:(lia))).
    all: lazymatch goal with
         | [ H : (?start <? ?stop)%uint63 = true, H' : (?step <? ?stop - ?start)%uint63 = false |- context[((?stop - ?start - 1) / ?step)%uint63] ]
           => let H'' := fresh in
              let H''' := fresh in
              assert (H'' : (stop - start - 1 <? step)%uint63 = true) by lia;
              assert (H''' : (0 <? step)%uint63 = true) by lia;
              replace ((stop - start - 1) / step)%uint63 with 0%uint63 by (clear -H'' H'''; nia)
         | _ => idtac
         end.
    all: lazymatch goal with
         | [ |- context[(?x * (1 + 0))%uint63] ]
           => replace (x * (1 + 0))%uint63 with x by lia
         | _ => idtac
         end.
    all: rewrite ?Nat.add_0_r.
    all: try match goal with
           | [ H : ?P _ ?x ?v |- ?P _ ?y ?v ] => replace y with x by lia
           end.
    all: try assumption.
    all: try match goal with |- ?P (?n + ?x)%nat _ _ => change x with 1%nat end.
    all: rewrite ?Nat.add_succ_r, ?Nat.add_0_r.
    all: auto.
    all: try (exfalso; lia).
    all: [ > ].
    do 3 (let x := open_constr:(_) in specialize (IH x)).
    lazymatch goal with
    | [ |- ?P ?n ?y ?v ]
      => lazymatch type of IH with
         | _ -> _ -> P ?n' ?x ?v'
           => unify v v';
              replace y with x;
              [ replace n with n'
              | ]
         end
    end.
    { specialize (fun pf => IH pf ltac:(eauto)).
      specialize (IH ltac:(lia)).
      auto. }
    all: match goal with
         | [ |- context[((?stop - (?start + ?step) - 1) / ?step)%uint63] ]
           => replace ((stop - (start + step) - 1))%uint63 with (((stop - start - 1) - step))%uint63 by lia
         end.
    all: match goal with
         | [ |- context[((?x - ?z) / ?z)%uint63] ]
           => let H := fresh in
              let H' := fresh in
              assert (H : z <> 0%uint63) by (subst z; clear; break_innermost_match; lia);
              assert (H' : (z <=? x)%uint63 = true) by lia;
              replace ((x - z) / z)%uint63 with (x / z - 1)%uint63;
              [
              | revert H H'; generalize x z; clear;
                intros;
                zify;
                let wBv := (eval cbv in wB) in
                repeat match goal with
                  | [ H : context[(wBv * ?q)%Z] |- _ ]
                    => lazymatch goal with
                       | [ H' : (q = 0 \/ q = 1)%Z |- _ ] => fail
                       | [ H' : (q = 0 \/ q = -1)%Z |- _ ] => fail
                       | _ => first [ assert (q = 0 \/ q = 1)%Z by nia
                                    | assert (q = 0 \/ q = -1)%Z by nia ]
                       end
                  end ]
         end;
      [ | nia ].
    2: lia.
    { assert ((0 <? (stop - start - 1) / step)%uint63 = true) by lia.
      generalize dependent (((stop - start - 1) / step)%uint63); intros.
      lia. }
  Qed.

  Lemma map_reduce_no_init_spec {A reduce start stop step0 f} (P : int -> _ -> Prop)
    (step : int := if (step0 =? 0) then 1 else step0)
    (Hinit : P (start + step) (f start))
    (Hstep : forall i v, P i v -> (start + step <=? i) = true -> (i <? stop) = true -> P (i + step) (reduce v (f i)))
    : P (start + step + step * (if start + step <? stop then 1 + (stop - (start + step) - 1) // step else 0))
        (@map_reduce_no_init A reduce start stop step0 f).
  Proof.
    eapply map_reduce_no_init_spec_count with (P := fun _ => P); eauto.
  Qed.

  Lemma map_reduce_spec_count {A B reduce init start stop step0 f} (P : nat -> int -> _ -> Prop)
    (step : int := if (step0 =? 0) then 1 else step0)
    (Hinit : P 0 start init)
    (Hstep : forall n i v, P n i v -> (start <=? i) = true -> (i <? stop) = true -> i = start + step * Uint63.of_Z n -> P (S n) (i + step) (reduce v (f i)))
    : P (if start <? stop then S (Z.to_nat ((stop - start - 1) // step : int)) else 0)
        (start + step * if start <? stop then 1 + (stop - start - 1) // step else 0)
        (@map_reduce A B reduce init start stop step0 f).
  Proof.
    cbv [map_reduce for_loop_lt Fix Classes.eqb int_has_eqb Classes.one Classes.zero int_has_one int_has_zero Classes.leb Classes.ltb int_has_leb int_has_leb int_has_ltb nat_has_zero] in *.
    fold step.
    set (f' := fun (i : int) continue state => _) at 1.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    set (start0 := start) in Hstep.
    assert (Hstart : (start0 <=? start) = true) by (cbv; lia).
    set (n0 := O) in Hinit.
    assert (Hn : start = start0 + step * Uint63.of_Z n0) by (cbv; lia).
    change (P ?x) with (P (n0 + x)%nat).
    clearbody start0 n0.
    revert start wf init Hstart n0 Hn Hinit.
    fix IH 2.
    intros ? [wf]; intros; cbn [Fix_F]; cbv [Acc_inv].
    unfold f' at 1.
    set (wf' := wf _); specialize (fun pf y b => IH _ (wf' y) b pf); clearbody wf'; clear wf.
    cbv [Monad.bind run_body LoopNotation.get LoopBody_Monad bind LoopNotation.set LoopNotation.update].
    cbv [Classes.sub Classes.mul Classes.zero Classes.add Classes.ltb Classes.leb Classes.one Classes.int_div int_has_int_div int_has_sub int_has_ltb int_has_add int_has_mul int_has_leb] in *.
    break_match; break_innermost_match_hyps; auto.
    all: try (specialize (IH ltac:(lia))).
    all: lazymatch goal with
         | [ H : (?start <? ?stop)%uint63 = true, H' : (?step <? ?stop - ?start)%uint63 = false |- context[((?stop - ?start - 1) / ?step)%uint63] ]
           => let H'' := fresh in
              let H''' := fresh in
              assert (H'' : (stop - start - 1 <? step)%uint63 = true) by lia;
              assert (H''' : (0 <? step)%uint63 = true) by lia;
              replace ((stop - start - 1) / step)%uint63 with 0%uint63 by (clear -H'' H'''; nia)
         | _ => idtac
         end.
    all: lazymatch goal with
         | [ |- context[(?x * (1 + 0))%uint63] ]
           => replace (x * (1 + 0))%uint63 with x by lia
         | _ => idtac
         end.
    all: rewrite ?Nat.add_0_r.
    all: try match goal with
           | [ H : ?P _ ?x ?v |- ?P _ ?y ?v ] => replace y with x by lia
           end.
    all: try assumption.
    all: try match goal with |- ?P (?n + ?x)%nat _ _ => change x with 1%nat end.
    all: rewrite ?Nat.add_1_r.
    all: auto.
    all: try (exfalso; lia).
    all: [ > ].
    do 3 (let x := open_constr:(_) in specialize (IH x)).
    lazymatch goal with
    | [ |- ?P ?n ?y ?v ]
      => lazymatch type of IH with
         | _ -> _ -> P ?n' ?x ?v'
           => unify v v';
              replace y with x;
              [ replace n with n'
              | ]
         end
    end.
    { specialize (fun pf => IH pf ltac:(eauto)).
      specialize (IH ltac:(lia)).
      auto. }
    all: match goal with
         | [ |- context[((?stop - (?start + ?step) - 1) / ?step)%uint63] ]
           => replace ((stop - (start + step) - 1))%uint63 with (((stop - start - 1) - step))%uint63 by lia
         end.
    all: match goal with
         | [ |- context[((?x - ?z) / ?z)%uint63] ]
           => let H := fresh in
              let H' := fresh in
              assert (H : z <> 0%uint63) by (subst z; clear; break_innermost_match; lia);
              assert (H' : (z <=? x)%uint63 = true) by lia;
              replace ((x - z) / z)%uint63 with (x / z - 1)%uint63;
              [
              | revert H H'; generalize x z; clear;
                intros;
                zify;
                let wBv := (eval cbv in wB) in
                repeat match goal with
                  | [ H : context[(wBv * ?q)%Z] |- _ ]
                    => lazymatch goal with
                       | [ H' : (q = 0 \/ q = 1)%Z |- _ ] => fail
                       | [ H' : (q = 0 \/ q = -1)%Z |- _ ] => fail
                       | _ => first [ assert (q = 0 \/ q = 1)%Z by nia
                                    | assert (q = 0 \/ q = -1)%Z by nia ]
                       end
                  end ]
         end;
      [ | nia ].
    2: lia.
    { assert ((0 <? (stop - start - 1) / step)%uint63 = true) by lia.
      generalize dependent (((stop - start - 1) / step)%uint63); intros.
      lia. }
  Qed.

  Lemma map_reduce_spec {A B reduce init start stop step0 f} (P : int -> _ -> Prop)
    (step : int := if (step0 =? 0) then 1 else step0)
    (Hinit : P start init)
    (Hstep : forall i v, P i v -> (start <=? i) = true -> (i <? stop) = true -> P (i + step) (reduce v (f i)))
    : P (start + step * (if start <? stop then 1 + (stop - start - 1) // step else 0))
        (@map_reduce A B reduce init start stop step0 f).
  Proof.
    eapply map_reduce_spec_count with (P := fun _ => P); eauto.
  Qed.

  Definition in_bounds (start stop step i : int) : bool
    := ((start <=? i) && ((i =? start) || (i <? stop)) && (((i - start) mod step) =? 0)).

  Definition in_bounds_alt (start stop step i : int)
    := (exists n : nat,
           (n <= Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
           /\ i = start + step * Uint63.of_Z n).

  Lemma argmax_gen_spec {A} {ltbA : has_ltb A} {ltbA_trans : @Transitive A Classes.ltb} {ltbA_irref : @Irreflexive A Classes.ltb} {ltbA_total : Antisymmetric A (@eq A) (fun x y => negb (Classes.ltb x y))}
    {start stop step0 f i v}
    (step : int := if (step0 =? 0) then 1 else step0)
    : map_reduce_no_init (@argmax_ int A ltbA) start stop step0 (fun i : int => (i, f i)) = (i, v)
      <-> (v = f i
           /\ in_bounds_alt start stop step i
           /\ forall j,
              in_bounds_alt start stop step j
              -> (((f j <? f i) = true)
                  \/ (f j = f i /\ (i <=? j) = true))).
  Proof.
    revert v i; cbv [in_bounds_alt].
    apply map_reduce_no_init_spec_count with (P:=fun n _ v => forall v' i', v = _ <-> _).
    all: fold step.
    all: intros; cbv [argmax_]; cbn [fst snd]; break_innermost_match; split; intros.
    all: cbv [Classes.one Classes.zero Classes.add Classes.mul nat_has_one int_has_one int_has_add int_has_mul int_has_zero Nat.lt] in *.
    all: repeat first [ progress subst
                      | progress change (of_Z 0%nat) with 0%uint63 in *
                      | progress intros
                      | reflexivity
                      | progress cbn [fst snd Nat.pred] in *
                      | match goal with
                        | [ H : (_, _) = (_, _) |- _ ] => inversion H; clear H
                        | [ H : _ /\ _ |- _ ] => destruct H
                        | [ H : ex _ |- _ ] => destruct H
                        | [ H : ?x <= 0%nat |- _ ] => assert (x = 0%nat) by lia; subst; clear H
                        | [ |- context[(?x * 0)%uint63] ] => replace (x * 0)%uint63 with 0%uint63 by lia
                        | [ |- context[(?x + 0)%uint63] ] => replace (x + 0)%uint63 with x by lia
                        | [ |- ?x = ?x /\ _ ] => split
                        | [ |- 0%nat < 1%nat ] => lia
                        | [ |- _ /\ ?x = ?x ] => split
                        | [ |- (exists n, n <= 0%nat /\ _) /\ _ ] => split; [ exists 0%nat | ]
                        | [ |- (?x <? ?x) = true \/ _ ] => right
                        | [ |- (?x <=? ?x) = true ] => generalize x; cbv; clear; intros; lia
                        | [ H : _ * _ |- _ ] => destruct H
                        | [ H : forall v' i', (?i, ?v) = (i', v') <-> _ |- _ ]
                          => pose proof (proj1 (H _ _) eq_refl);
                             pose proof (fun v' i' => proj2 (H v' i'));
                             clear H
                        | [ H : forall v' i', v' = _ /\ _ -> _ |- _ ] => specialize (fun i' pf => H _ i' (conj eq_refl pf))
                        | [ |- _ /\ _ ] => split
                        | [ |- context[(?start + ?step + ?step * of_Z (Z.of_N (N.of_nat ?n')))%uint63] ]
                          => replace (start + step + step * of_Z (Z.of_N (N.of_nat n')))%uint63
                            with (start + step * of_Z (Z.of_N (N.of_nat (S n'))))%uint63 in *
                            by lia
                        | [ H : context[(?start + ?step + ?step * of_Z (Z.of_N (N.of_nat ?n')))%uint63] |- _ ]
                          => replace (start + step + step * of_Z (Z.of_N (N.of_nat n')))%uint63
                            with (start + step * of_Z (Z.of_N (N.of_nat (S n'))))%uint63 in *
                            by lia
                        | [ |- exists n, _ /\ (?start + ?step * of_Z (Z.of_N (N.of_nat ?n')) = ?start + ?step * of_Z (Z.of_N (N.of_nat n)))%uint63 ]
                          => exists n'
                        | [ H : forall j, ex _ -> (?f j <? ?f ?x) = true \/ _, H' : (?f ?x <? ?f ?y) = true |- (?f ?z <? ?f ?y) = true \/ _ ]
                          => specialize (fun pf => H z (ex_intro _ _ (conj pf eq_refl)))
                        | [ H : ?x <= ?y -> _, H' : ?x <= S ?y |- _ ]
                          => is_var x; destruct (Nat.eq_decidable x (S y)); subst;
                             [ clear H H'
                             | assert (x <= y) by lia;
                               clear H';
                               specialize (H ltac:(assumption)) ]
                        | [ H : ?x <= S ?n |- _ ]
                          => is_var x;
                             destruct (Nat.eq_dec x (S n)); subst;
                             [ | assert (x <= n)%nat by lia ];
                             clear H
                        end
                      | lia
                      | match goal with
                        | [ H : context[?x = true] |- _ ] => change (x = true) with (is_true x) in H
                        | [ |- context[?x = true] ] => change (x = true) with (is_true x)
                        | [ H : is_true (?x <? ?y), H' : is_true (?z <? ?x) \/ (?z = ?x /\ _) |- is_true (?z <? ?y) \/ _ ]
                          => left; destruct H' as [?|[H' ?]]; [ etransitivity; eassumption | rewrite H'; assumption ]
                        | [ H : forall j, (exists n0, n0 <= S ?n /\ @?A j n0 = @?B j n0) -> _ |- _ ]
                          => pose proof (H _ (ex_intro _ (S n) (conj (@Nat.le_refl _) eq_refl)));
                             let pf := fresh in
                             pose proof (fun j (pf : exists n0, n0 <= n /\ A j n0 = B j n0)
                                         => H j ltac:(let n0 := fresh in
                                                      let H1 := fresh in
                                                      let H2 := fresh in
                                                      destruct pf as [n0 [H1 H2]];
                                                      exists n0; split; [ apply le_S, H1 | apply H2 ]));
                             clear H
                        | [ H : forall i, ex _ /\ _ -> _, H' : forall j, ex _ -> _ |- _ ]
                          => let H'' := fresh in
                             pose proof ((fun pf => H _ (conj (ex_intro _ _ (conj pf eq_refl)) H')) ltac:(assumption)) as H'';
                             lazymatch type of H'' with
                             | (?x, _) = (?x, _) => fail
                             | (?x, ?y) = (_, _)
                               => inversion H''; clear H'';
                                  generalize dependent x; intros; subst
                             end
                        | [ HIrr : Irreflexive _, HTrans : Transitive _, H : is_true (?x <? ?y) \/ _, H' : is_true (?y <? ?x) |- _ ]
                          => destruct H as [H|H]; [ exfalso; clear -H H' HTrans HIrr; eapply HIrr, HTrans; eassumption | ]
                        | [ H : (?x <? ?y) = false |- is_true (?y <? ?x) \/ _ ]
                          => destruct (y <? x) eqn:?; [ left | right ]
                        | [ HIrr : Irreflexive _, HTrans : Transitive _, H : is_true (?x <? ?y), H' : is_true (?y <? ?x) |- _ ]
                          => exfalso; clear -H H' HTrans HIrr; eapply HIrr, HTrans; eassumption
                        | [ HAnti : Antisymmetric _ eq _, H : (?x <? ?y) = false, H' : (?y <? ?x) = false |- _ ]
                          => assert (x = y) by (eapply HAnti; rewrite ?H, ?H'; reflexivity);
                             clear H H'
                        end
                      | congruence ].
  Admitted.

  (*
  Lemma in_bounds_alt start stop step i
    : in_bounds start stop step i = true
      <-> (exists n : nat,
              (n <= Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
              /\ i = start + step * Uint63.of_Z n).
  Proof.
    destruct (start + step <? stop) eqn:Hlt.
    all: cbv [in_bounds].
    2: { split; [ intro H; exists 0%nat; split | intros [n H] ].
         lia.
         cbv.
         assert (start
         lia.
    { cbv [in_bounds]; split; [ intro H | intros [n H] ].
    { exists

*)
  Lemma argmax_spec {A} {ltbA : has_ltb A} {start stop step f v}
    : @argmax A ltbA start stop step f = v
      <-> (in_bounds start stop step v = true
           /\ forall j,
              in_bounds start stop step j = true
              -> (((f j <? f v) = true)
                  \/ (f j = f v /\ (v <=? j) = true))).
  Proof.
    cbv [argmax].
    destruct map_reduce_no_init eqn:H.
    (*rewrite argmax_gen_spec in H.
    destruct_head'_and; subst; destruct_head'_ex; destruct_head'_and; subst; cbn [fst snd].*)
    (* XXX FIXME *)
  Admitted.

  Lemma argmax_max_equiv
    {A} {ltbA : has_ltb A}
    {maxA : has_max A}
    {start stop step f}
    (Hltb_max_compat : forall x y : A, Classes.max x y = if x <? y then y else x)
    : @max A maxA start stop step f = f (@argmax A ltbA start stop step f).
  Proof.
    cbv [max argmax].
    cbv [map_reduce_no_init for_loop_lt Fix Classes.eqb int_has_eqb Classes.one Classes.zero int_has_one int_has_zero Classes.leb Classes.ltb int_has_leb int_has_leb int_has_ltb nat_has_zero nat_has_one Classes.add int_has_add] in *.
    set (step' := if _ : bool then _ else step).
    set (f' := fun (i : int) continue state => _) at 1.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    set (start' := (start + step')%uint63) in *.
    set (v := (start, f start)).
    change (f start) with (snd v).
    assert (Hv : snd v = f (fst v)) by reflexivity.
    destruct v as [v fv]; cbn [snd fst] in *; subst.
    replace start with (start' - step')%uint63 by (subst start'; lia).
    clearbody start'; clear start; rename start' into start.
    revert start wf v.
    fix IH 2.
    intros ? [wf]; intros; cbn [Fix_F]; cbv [Acc_inv].
    unfold f' at 1.
    set (wf' := wf _); specialize (fun y => IH _ (wf' y)); clearbody wf'; clear wf.
    cbv [Monad.bind run_body LoopNotation.get LoopBody_Monad bind LoopNotation.set LoopNotation.update].
    cbv [Classes.sub Classes.mul Classes.zero Classes.add Classes.ltb Classes.leb Classes.one Classes.int_div int_has_int_div int_has_sub int_has_ltb int_has_add int_has_mul int_has_leb] in *.
    break_match; break_innermost_match_hyps; auto.
    all: cbv [argmax_]; cbn [fst snd]; rewrite Hltb_max_compat; cbv [Classes.ltb]; break_innermost_match; try reflexivity.
    all: rewrite IH; reflexivity.
  Qed.

  Lemma map_reduce_const {A B reduce init start stop step0 f}
    (step := if step0 =? 0 then 1 else step0)
    : @map_reduce A B reduce init start stop step0 (fun _ => f)
      = N.iter ((if start <? stop then 1 + (((stop - start - 1) // step : int) : N) else 0)%core) (fun v => reduce v f) init.
  Proof.
    rewrite (map_reduce_spec_count (fun (n : nat) (i:int) v => v = N.iter (N.of_nat n) (fun v => reduce v f) init)); fold step.
    all: cbv [Classes.add Classes.sub Classes.div Classes.int_div Classes.one Classes.eqb Classes.zero Classes.mul int_has_add int_has_sub int_has_int_div int_has_one int_has_zero int_has_eqb int_has_mul N_has_add N_has_one coer_int_N'] in *.
    all: try reflexivity.
    { f_equal; break_innermost_match; try reflexivity; lia. }
    { intros; subst.
      rewrite Nnat.Nat2N.inj_succ.
      rewrite N.iter_succ; reflexivity. }
  Qed.

  Lemma map_reduce_distr12 {A B R reduce init1 init2 start stop step} {f : int -> B -> A} {F}
    : R init2 (F init1)
      -> (forall init init' x,
             R init' (F init)
             -> R (reduce init' (F (f x))) (F (fun j => reduce (init j) (f x j))))
      -> R (@map_reduce A A reduce init2 start stop step (fun i => F (f i)))
           (F (fun j => @map_reduce A A reduce (init1 j) start stop step (fun i => f i j))).
  Proof.
    cbv [map_reduce for_loop_lt Fix].
    set (F' := fun (i : int) continue state => _) at 1.
    lazymatch goal with
    | [ |- context G[fun j => Fix_F ?T (@?f'v j) ?wf (@?i j)] ]
      => pose f'v as f';
         let G := context G[fun j => Fix_F T (f' j) wf (i j)] in
         change G; cbv beta
    end.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    revert start wf init1 init2.
    fix IH 2.
    intros ? [wf]; intros; cbn [Fix_F]; cbv [Acc_inv].
    unfold f' at 1.
    unfold F' at 1.
    set (wf' := wf _); specialize (fun y => IH _ (wf' y)); clearbody wf'; clear wf.
    cbv [Monad.bind run_body LoopNotation.get LoopBody_Monad bind LoopNotation.set LoopNotation.update].
    subst; break_match; auto.
  Qed.

  Lemma map_reduce_distr1 {A R reduce init1 init2 start stop step f F}
    : R init2 (F init1)
      -> (forall init init' x,
             R init' (F init)
             -> R (reduce init' (F (f x))) (F (reduce init (f x))))
      -> R (@map_reduce A A reduce init2 start stop step (fun i => F (f i)))
           (F (@map_reduce A A reduce init1 start stop step f)).
  Proof.
    intros; apply (@map_reduce_distr12 A unit R reduce (fun _ => init1) init2 start stop step (fun i _ => f i) (fun v => F (v tt))); auto.
  Qed.

  Lemma map_reduce_distr2 {A R reduce init1 init2 init3 start stop step f g F}
    : R init3 (F init1 init2)
      -> (forall init1 init2 init3 x,
             R init3 (F init1 init2)
             -> R (reduce init3 (F (f x) (g x)))
                  (F (reduce init1 (f x)) (reduce init2 (g x))))
      -> R (@map_reduce A A reduce init3 start stop step (fun i => F (f i) (g i)))
           (F (@map_reduce A A reduce init1 start stop step f)
              (@map_reduce A A reduce init2 start stop step g)).
  Proof.
    cbv [map_reduce for_loop_lt Fix].
    set (F' := fun (i : int) continue state => _) at 1.
    set (f' := fun (i : int) continue state => _) at 1.
    set (g' := fun (i : int) continue state => _) at 1.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    revert start wf init1 init2 init3.
    fix IH 2.
    intros ? [wf]; intros; cbn [Fix_F]; cbv [Acc_inv].
    unfold g' at 1, f' at 1.
    unfold F' at 1.
    set (wf' := wf _); specialize (fun y => IH _ (wf' y)); clearbody wf'; clear wf.
    cbv [Monad.bind run_body LoopNotation.get LoopBody_Monad bind LoopNotation.set LoopNotation.update].
    subst; break_match; auto.
  Qed.

  Section sum.
    Context {A} {R} {zeroA : has_zero A} {addA : has_add A} {mulA : has_mul A}
      {R_refl : @Reflexive A R}
      {R_trans : @Transitive A R}
      {R_sym : @Symmetric A R}
      {addA_Proper : Proper (R ==> R ==> R) add}
      {addA_assoc : Associative R add}
      {addA_comm : Commutative R add}
      {zeroA_add_l : LeftId R add 0}
      {zeroA_mul_l : LeftZero R mul 0}
      {zeroA_mul_r : RightZero R mul 0}
      {mulA_addA_distr_l : LeftDistributive R mul add}
      {mulA_addA_distr_r : RightDistributive R mul add}
      {coerN : has_coer N A}.

    Lemma sum_const [start stop step] (f : A)
      : sum start stop step (fun x => f)
        = N.iter
            (if start <? stop
             then 1 + ((stop - start - 1) // (if step =? 0 then 1 else step) : int)
             else 0) (fun v : A => v + f) 0.
    Proof using Type.
      cbv [sum]. rewrite map_reduce_const; reflexivity.
    Qed.

    Lemma sum_const_mul
      (H0l : forall x : A, R 0 (coer (0:N) * x))
      (Hsucc : forall (x : A) (n : N), R (coer n * x + x) (coer (N.succ n) * x))
      [start stop step] (f : A)
      : R (sum start stop step (fun x => f))
          (coer
             (if start <? stop return N
              then 1 + (((stop - start - 1) // (if step =? 0 then 1 else step) : int) : N)
              else 0) * f).
    Proof using R_refl R_trans addA_Proper.
      rewrite sum_const.
      set (n := if start <? stop then _ else _); clearbody n.
      induction n as [|n IH] using N.peano_ind; [ now cbn; apply H0l | ].
      rewrite N.iter_succ, IH; apply Hsucc.
    Qed.

    Lemma sum_const0 start stop step
      : R (sum start stop step (fun x => 0)) 0.
    Proof using R_refl R_trans addA_Proper zeroA_add_l.
      rewrite sum_const.
      apply N.iter_invariant; intros *; try intros ->; try reflexivity.
      auto.
    Qed.

    Lemma sum_const_step1 [start stop] (f : A)
      : sum start stop 1 (fun x => f)
        = N.iter
            (if start <? stop
             then N.succ (stop - start - 1)%uint63
             else 0) (fun v : A => v + f) 0.
    Proof using Type.
      rewrite sum_const; cbv [Classes.eqb int_has_eqb PrimInt63.eqb Classes.one Classes.zero int_has_one int_has_zero Classes.int_div int_has_int_div Classes.add Classes.sub int_has_sub] in *; change (1 =? 0)%uint63 with false; break_innermost_match; try reflexivity.
      f_equal.
      replace ((stop - start - 1) / 1)%uint63 with (stop - start - 1)%uint63 by nia.
      lia.
    Qed.

    Lemma sum_const_mul_step1
      (H0l : forall x : A, R 0 (coer (0:N) * x))
      (Hsucc : forall (x : A) (n : N), R (coer n * x + x) (coer (N.succ n) * x))
      [start stop] (f : A)
      : R (sum start stop 1 (fun x => f))
          (coer
             (if start <? stop return N
              then N.succ (stop - start - 1)%uint63
              else 0) * f).
    Proof using R_refl R_trans addA_Proper.
      rewrite sum_const_step1.
      set (n := if start <? stop then _ else _); clearbody n.
      induction n as [|n IH] using N.peano_ind; [ now cbn; apply H0l | ].
      rewrite N.iter_succ, IH; apply Hsucc.
    Qed.

    Lemma sum_distr [start stop step] (f g : int -> A)
      : R (sum start stop step (fun x => f x + g x))
          (sum start stop step f + sum start stop step g).
    Proof using R_refl R_sym R_trans addA_Proper addA_assoc addA_comm zeroA_add_l.
      cbv [sum]; apply map_reduce_distr2.
      { symmetry; apply id_l. }
      { intros * ->.
        repeat ((rewrite -> ?addA_assoc + rewrite <- ?addA_assoc);
                apply addA_Proper; try reflexivity; []).
        apply addA_comm. }
    Qed.

    Lemma mul_sum_distr_l [start stop step] (f : int -> A) x
      : R (x * sum start stop step f) (sum start stop step (fun i => x * f i)).
    Proof using R_refl R_trans addA_Proper mulA_addA_distr_l zeroA_mul_r.
      cbv [sum]; apply @map_reduce_distr1 with (F:=fun fi => x * fi).
      { apply zero_r. }
      { intros *.
        rewrite distr_l.
        intros ->; reflexivity. }
    Qed.

    Lemma mul_sum_distr_r [start stop step] (f : int -> A) x
      : R (sum start stop step f * x) (sum start stop step (fun i => f i * x)).
    Proof using R_refl R_trans addA_Proper mulA_addA_distr_r zeroA_mul_l.
      cbv [sum]; apply @map_reduce_distr1 with (F:=fun fi => fi * x).
      { apply zero_l. }
      { intros *.
        rewrite distr_r.
        intros ->; reflexivity. }
    Qed.

    Lemma sum_swap [start1 stop1 step1 start2 stop2 step2] (f : int -> int -> A)
      : R (sum start1 stop1 step1 (fun i => sum start2 stop2 step2 (f i)))
          (sum start2 stop2 step2 (fun j => sum start1 stop1 step1 (fun i => f i j))).
    Proof using R_refl R_sym R_trans addA_Proper addA_assoc addA_comm zeroA_add_l.
      set (s1 := sum start1 stop1 step1).
      unfold sum in s1; subst s1; cbv beta.
      apply map_reduce_distr12.
      { rewrite sum_const0; reflexivity. }
      { intros * H.
        rewrite H, sum_distr; reflexivity. }
    Qed.
  End sum.
End Reduction.
Export (hints) Reduction.
