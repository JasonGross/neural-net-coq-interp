From Coq Require Import Zify ZifyUint63 Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Arith.Classes.Laws Arith.Instances.Laws Arith.Instances.Zify Default.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose SpecializeBy.
From NeuralNetInterp.Util Require Import Wf_Uint63.
From NeuralNetInterp.Util.Compat Require Import PeanoNat.
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

  Lemma map_reduce_no_init_spec_eq_Nat_iter {A reduce start stop step0 f}
    (step : int := if (step0 =? 0) then 1 else step0)
    : @map_reduce_no_init A reduce start stop step0 f
      = fst (Nat.iter
               (Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
               (fun '(v, i) => (reduce v (f i), i + step))
               (f start, start + step)).
  Proof.
    pose proof
      (@map_reduce_no_init_spec_count A reduce start stop step0 f
         (fun n i v => (v, i) = Nat.iter
                                  (Nat.pred n)
                                  (fun '(v, i) => (reduce v (f i), i + step))
                                  (f start, start + step))) as H.
    fold step in H.
    cbv beta zeta in H.
    rewrite <- H; clear H; [ cbn [fst]; reflexivity | .. ].
    { cbv; reflexivity. }
    cbn [Nat.pred].
    intros ??? IH; rewrite Nat.iter_succ, <- IH; reflexivity.
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

  Lemma map_reduce_spec_eq_Nat_iter {A B reduce init start stop step0 f}
    (step : int := if (step0 =? 0) then 1 else step0)
    : @map_reduce A B reduce init start stop step0 f
      = fst (Nat.iter
               (if start <? stop then S (Z.to_nat ((stop - start - 1) // step : int)) else 0)
               (fun '(v, i) => (reduce v (f i), i + step))
               (init, start)).
  Proof.
    pose proof
      (@map_reduce_spec_count A B reduce init start stop step0 f
         (fun n i v => (v, i) = Nat.iter
                                  n
                                  (fun '(v, i) => (reduce v (f i), i + step))
                                  (init, start))) as H.
    fold step in H.
    cbv beta zeta in H.
    setoid_rewrite <- H; clear H; [ cbn [fst]; reflexivity | .. ].
    { cbv; reflexivity. }
    intros ??? IH; rewrite Nat.iter_succ, <- IH; reflexivity.
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

  Definition in_bounds_alt_at (start stop step i : int) (n : nat)
    := (n <= (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0))
       /\ i = start + step * Uint63.of_Z n.

  Definition in_bounds_alt (start stop step i : int)
    := exists n : nat, in_bounds_alt_at start stop step i n.

  Lemma in_bounds_alt_bounded start stop step i
    : in_bounds_alt start stop step i
      -> if (start + step <? stop)
         then (let maxn : nat := 1 + Z.to_nat ((stop - (start + step) - 1) // step : int) in
               ((start <=? i) = true /\ (i <=? start + step * Uint63.of_Z maxn) = true)
               \/ (((wB <=? (maxn:Z))
                    || (wB <=? Uint63.to_Z step * (maxn:Z))
                    || (wB <=? Uint63.to_Z start + Uint63.to_Z (step * Uint63.of_Z maxn))) = true))
            else i = start.
  Proof.
    cbv [in_bounds_alt in_bounds_alt_at].
    break_innermost_match; intros [n H]; destruct_head'_and; subst.
    all: cbv [Classes.add Classes.sub Classes.one Classes.mul Classes.int_div Classes.leb Classes.ltb
                int_has_add int_has_sub int_has_one int_has_mul int_has_int_div int_has_leb int_has_ltb
                nat_has_add nat_has_one
                Z_has_leb Z_has_add Z_has_mul] in *.
    all: repeat first [ progress subst
                      | match goal with
                        | [ H : ?n <= 0 |- _ ] => assert (n = 0%nat) by lia; clear H
                        end
                      | progress destruct_head'_and
                      | progress change (N.of_nat 0) with 0%N
                      | progress change (Z.of_N 0) with 0%Z
                      | progress change (of_Z 0) with 0%uint63
                      | progress change (Z.of_nat 1) with 1%Z in *
                      | rewrite !nat_N_Z in *
                      | rewrite !Nat2Z.inj_add in *
                      | rewrite !Z2Nat.id in * by nia
                      | match goal with
                        | [ |- ?A \/ ?b = true ] => destruct b eqn:?; first [ right; reflexivity | left ]
                        | [ H : ?n <= ?x |- _ ]
                          => assert (Z.of_nat n <= Z.of_nat x)%Z by lia; clear H
                        end
                      | rewrite !Bool.orb_false_iff in *
                      | lia ].
    set (k := (1 + _)%Z) in *.
    assert (1 <= k)%Z by (clear; subst k; nia).
    clearbody k.
    assert ((wB <=? Z.of_nat n) = false)%Z by lia.
    assert ((wB <=? Uint63.to_Z step * Z.of_nat n) = false)%Z by nia.
    assert ((wB <=? Uint63.to_Z start + Uint63.to_Z step * k) = false -> (wB <=? Uint63.to_Z start + Uint63.to_Z step * Z.of_nat n) = false)%Z by nia.
    cbv [wB] in *.
    vm_compute Z.pow in *.
    repeat match goal with
           | [ H : (?wB <=? ?k)%Z = false |- _ ]
             => unique assert (k mod wB = k)%Z by (apply Z.mod_small; lia)
           | [ H : ?A -> (?wB <=? ?k)%Z = false |- _ ]
             => unique assert (A -> k mod wB = k)%Z by (let pf := fresh in intro pf; specialize (H pf); lia)
           end.
    #[local] Ltac zify_convert_to_euclidean_division_equations_flag ::= constr:(false).
    zify.
    generalize dependent (to_Z start); clear start; intro start; intros.
    generalize dependent (to_Z step); clear step; intro step; intros.
    generalize dependent (to_Z stop); clear stop; intro stop; intros.
    repeat first [ progress specialize_by_assumption
                 | match goal with
                   | [ H : (?x mod ?y = ?x)%Z, H' : context[(?x mod ?y)%Z] |- _ ] => rewrite H in H'
                   | [ H : (?x mod ?y = ?x)%Z |- context[(?x mod ?y)%Z] ] => rewrite H
                   end
                 | progress specialize_by lia ].
    nia.
    #[local] Ltac zify_convert_to_euclidean_division_equations_flag ::= constr:(true).
  Qed.

  #[local] Ltac t_argmaxmin_step_interesting_transitivity_reasoning _ :=
    match goal with
    | [ H : (?x <? ?y) = true \/ _, H' : (?y <? ?z) = true |- (?x <? ?z) = true \/ _ ]
      => (* The main driver of the interesting bit of this lemma, the transitivity part *)
        left; destruct H as [H|H]; [ change (is_true (x <? z)); etransitivity; eassumption | ]
    | [ H : (?x <? ?y) = true, H' : ?x' = ?x |- (?x' <? ?y) = true ]
      => rewrite H'
    | [ H : (?x <? ?y) = false |- (?y <? ?x) = true \/ _ ]
      => let H' := fresh in
         destruct (y <? x) eqn:H';
         [ left; reflexivity
         | right;
           let H'' := fresh in
           assert (H'' : x = y)
             by (now apply antisymmetry; first [ rewrite H | rewrite H' ]);
           try first [ rewrite !H'' | rewrite <- !H'' ] ]
    | [ H_asym : Asymmetric _, H : (?x <=? ?y) = false |- is_true (?y <=? ?x) ]
      => let H' := fresh in
         destruct (y <=? x) eqn:H';
         [ reflexivity
         | exfalso; apply (H_asym x y); cbv beta;
           try first [ rewrite H | rewrite H' ] ]
    | [ H : (?y <=? ?x) = false, H' : (?y <=? ?z) = true |- (?x <=? ?z) = true ]
      => change (is_true (x <=? z)); etransitivity; [ clear H' | exact H' ]
    | [ H : (?x <=? ?y) = true, H' : (?y <=? ?z) = true, H'' : (?x <=? ?z) = false |- _ ]
      => cut (is_true (x <=? z));
         [ now rewrite H''
         | etransitivity; (exact H + exact H') ]
    end.

  #[local] Ltac t_argmaxmin_step _
    := first [ progress destruct_head'_and
             | progress destruct_head'_ex
             | progress destruct_head' iff
             | progress cbn [fst snd] in *
             | progress subst
             | progress intros
             | progress break_innermost_match_hyps_step
             | progress rewrite nat_N_Z in *
             | progress rewrite Nat2Z.inj_succ in *
             | progress specialize_by_assumption
             | progress specialize_by lia
             | reflexivity
             | progress change (of_Z (Z.of_nat 0)) with 0%uint63 in *
             | progress change PrimInt63.mul with (Classes.mul (A:=int)) in *
             | t_argmaxmin_step_interesting_transitivity_reasoning ()
             | match goal with
               | [ H : context[Nat.iter 0] |- _ ] => cbn in *
               | [ |- context[Nat.iter 0] ] => cbn in *
               | [ H : (_, _) = (_, _) |- _ ] => inversion H; clear H
               | [ H : forall i v, (_, _) = (i, v) -> _ |- _ ] => specialize (H _ _ eq_refl)
               | [ H : forall x y z, _ = (x, y, z) -> _ |- _ ] => specialize (H _ _ _ eq_refl)
               | [ H : ?x <= 0 |- _ ] => assert (x = 0); [ clear -H; generalize dependent x; cbv; intros; lia | clear H ]
               | [ |- context[of_Z ?x] ] => change (of_Z x) with 0%uint63 in *
               | [ |- context[(?x * 0%uint63)] ] => replace (x * 0%uint63) with 0%uint63 by (cbv [Classes.mul int_has_mul]; lia)
               | [ |- context[(0%uint63 * ?x)] ] => replace (0%uint63 * x) with 0%uint63 by (cbv [Classes.mul int_has_mul]; lia)
               | [ |- context[(?x + 0)%uint63] ] => replace (x + 0)%uint63 with x by lia
               | [ H : context[(?x * 0%uint63)] |- _ ] => replace (x * 0%uint63) with 0%uint63 in * by (cbv [Classes.mul int_has_mul]; lia)
               | [ H : context[(?x + 0%uint63)] |- _ ] => replace (x + 0%uint63) with x in * by (cbv [Classes.add int_has_add]; lia)
               | [ |- context[of_Z (Z.succ ?x)] ] => replace (of_Z (Z.succ x)) with (1 + of_Z x)%uint63 by lia
               | [ |- context[((1 + ?x)%uint63 * ?y)] ]
                 => replace ((1 + x)%uint63 * y) with (y + x * y)%uint63
                   by (cbv [Classes.add Classes.mul Classes.one int_has_add int_has_mul int_has_one]; generalize x y; clear; intros; nia)
               | [ |- context[(?y * (1 + ?x)%uint63)] ]
                 => replace (y * (1 + x)%uint63) with (y + x * y)%uint63
                   by (cbv [Classes.add Classes.mul Classes.one int_has_add int_has_mul int_has_one]; generalize x y; clear; intros; nia)
               | [ |- context[(?x + (?y + ?z')%uint63)] ]
                 => replace (x + (y + z')%uint63) with (x + y + z') in *
                     by (cbv [Classes.add Classes.mul int_has_add int_has_mul]; lia)
               | [ |- (?x <? ?x) = true \/ _ ] => right
               | [ |- (?x <=? ?x)%uint63 = true ] => clear; lia
               | [ |- exists n : nat, n <= 0 /\ _ ] => exists 0
               | [ H : context[let '(a, b) := ?x in _] |- _ ] => destruct x eqn:?
               | [ |- exists n0, _ /\ ?start + ?step + of_Z (Z.of_nat ?n) * ?step = ?start + ?step * of_Z (Z.of_N (N.of_nat n0)) /\ _ ]
                 => exists (S n)
               | [ |- exists n0, _ /\ ?start + ?step * of_Z (Z.of_nat ?n) = ?start + ?step * of_Z (Z.of_N (N.of_nat n0)) /\ _ ]
                 => exists n
               | [ H : forall n0, _ /\ ?start + ?step + of_Z (Z.of_nat ?n) * ?step = ?start + ?step * of_Z (Z.of_N (N.of_nat n0)) -> _ |- _ ]
                 => specialize (fun pf1 pf2 => H (S n) (conj pf1 pf2))
               | [ H : forall n0, _ /\ ?start + ?step * of_Z (Z.of_nat ?n) = ?start + ?step * of_Z (Z.of_N (N.of_nat n0)) -> _ |- _ ]
                 => specialize (fun pf1 pf2 => H n (conj pf1 pf2))
               | [ H : forall n0, _ /\ ?start = ?start + ?step * of_Z (Z.of_N (N.of_nat n0)) -> _ |- _ ]
                 => specialize (fun pf1 pf2 => H O (conj pf1 pf2))
               | [ ltbA_irref : @Irreflexive ?A _, H : @Classes.ltb ?A ?ltbA ?x ?x = true \/ _ |- _ ]
                 => destruct H as [H|H]; [ exfalso; eapply ltbA_irref, H | ]
               | [ H : ?x <= S ?n, H' : ?x <= ?n -> _ |- _ ]
                 => cut (x = S n \/ x <= n); [ clear H; intros [H|H] | lia ]
               | [ |- (?x <=? ?x) = true ] => change (is_true (x <=? x)); generalize x; reflexivity
               | [ |- (?x <=? ?x) = true ] => generalize x; cbv; clear; intros; lia
               | [ H : S ?x <= ?x -> _ |- _ ] => clear H
               | [ H : ?x = ?x |- _ ] => clear H
               | [ H : (?x <=? ?x) = true |- _ ] => clear H
               | [ H : ?x <= ?x |- _ ] => clear H
               end
             | apply conj; intros
             | solve [ auto ]
             | cbv [Classes.add Classes.mul int_has_add int_has_mul]; lia ].
  #[local] Ltac saturate_argminmax_step _ :=
    match goal with
    | [ H : forall j n, _ -> _ \/ (?f j = ?f _ /\ _), H' : context[?f ?jv] |- _ ]
      => unique pose proof (H jv)
    | [ H : forall j n, _ -> _ \/ (?f j = ?f _ /\ _) |- context[?f ?jv] ]
      => unique pose proof (H jv)
    | [ H : forall j n, _ -> (?f _ <=? ?f j) = true /\ _, H' : context[?f ?jv] |- _ ]
      => unique pose proof (H jv)
    | [ H : forall j n, _ -> (?f _ <=? ?f j) = true /\ _ |- context[?f ?jv] ]
      => unique pose proof (H jv)
    end.

  Lemma argmax_gen_spec_let {A} {ltbA : has_ltb A} {ltbA_trans : @Transitive A Classes.ltb} {ltbA_irref : @Irreflexive A Classes.ltb} {ltbA_total : Antisymmetric A (@eq A) (fun x y => negb (Classes.ltb x y))}
    {start stop step0 f}
    (step : int := if (step0 =? 0) then 1 else step0)
    : let '(i, v) := map_reduce_no_init (@argmax_ int A ltbA) start stop step0 (fun i : int => (i, f i)) in
      v = f i
      /\ (exists n : nat,
             (n <= Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
             /\ (i = start + step * Uint63.of_Z n
                 /\ forall j (n' : nat),
                    ((n' <= Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
                     /\ j = start + step * Uint63.of_Z n')
                    -> (((f j <? f i) = true)
                        \/ (f j = f i /\ (n <= n'))))).
  Proof.
    rewrite map_reduce_no_init_spec_eq_Nat_iter.
    cbv [in_bounds_alt argmax_]; cbn [Nat.pred snd fst]; fold step.
    set (ni := Nat.iter _ _ _).
    destruct ni as [[i v] ui] eqn:Hn; subst ni; cbn [fst].
    let G := lazymatch goal with |- ?G => G end in
    let n := lazymatch type of Hn with context[Nat.iter ?n] => n end in
    cut (ui = start + step + of_Z n * step /\ G); [ easy | ].
    revert i v ui Hn.
    break_innermost_match.
    2: cbn.
    1: let v := lazymatch goal with |- context[Nat.iter ?n] => n end in
       set (n := v).
    1: induction n as [|? IH]; intros *.
    all: try (rewrite Nat.iter_succ; break_match); destruct_head'_prod; cbn [fst snd] in *.
    all: repeat t_argmaxmin_step ().
    all: repeat saturate_argminmax_step ().
    all: repeat t_argmaxmin_step ().
  Qed.

  Lemma argmin_gen_spec_let {A} {lebA : has_leb A} {lebA_trans : @Transitive A Classes.leb} {lebA_refl : @Reflexive A Classes.leb} {lebA_total : @Asymmetric A (fun x y => negb (Classes.leb x y))}
    {start stop step0 f}
    (step : int := if (step0 =? 0) then 1 else step0)
    : let '(i, v) := map_reduce_no_init (@argmin_ int A lebA) start stop step0 (fun i : int => (i, f i)) in
      v = f i
      /\ (exists n : nat,
             (n <= Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
             /\ (i = start + step * Uint63.of_Z n
                 /\ forall j (n' : nat),
                    ((n' <= Nat.pred (S (if start + step <? stop then (1 + Z.to_nat ((stop - (start + step) - 1) // step : int)) else 0)))
                     /\ j = start + step * Uint63.of_Z n')
                    -> (f i <=? f j) = true
                       /\ ((f j <=? f i) = true
                           -> n <= n'))).
  Proof.
    rewrite map_reduce_no_init_spec_eq_Nat_iter.
    cbv [in_bounds_alt argmin_]; cbn [Nat.pred snd fst]; fold step.
    set (ni := Nat.iter _ _ _).
    destruct ni as [[i v] ui] eqn:Hn; subst ni; cbn [fst].
    let G := lazymatch goal with |- ?G => G end in
    let n := lazymatch type of Hn with context[Nat.iter ?n] => n end in
    cut (ui = start + step + of_Z n * step /\ G); [ easy | ].
    revert i v ui Hn.
    break_innermost_match.
    2: cbn.
    1: let v := lazymatch goal with |- context[Nat.iter ?n] => n end in
       set (n := v).
    1: induction n as [|? IH]; intros *.
    all: try (rewrite Nat.iter_succ; break_match); destruct_head'_prod; cbn [fst snd] in *.
    all: repeat t_argmaxmin_step ().
    all: repeat saturate_argminmax_step ().
    all: repeat t_argmaxmin_step ().
  Qed.

  Lemma argmax_spec {A} {ltbA : has_ltb A} {ltbA_trans : @Transitive A Classes.ltb} {ltbA_irref : @Irreflexive A Classes.ltb} {ltbA_total : Antisymmetric A (@eq A) (fun x y => negb (Classes.ltb x y))}
    {start stop step0 f v}
    (step : int := if (step0 =? 0) then 1 else step0)
    : @argmax A ltbA start stop step0 f = v
      <-> (exists n,
              in_bounds_alt_at start stop step v n
              /\ forall j n',
                in_bounds_alt_at start stop step j n'
                -> (((f j <? f v) = true)
                    \/ (f j = f v /\ n <= n'))).
  Proof.
    cbv [argmax].
    generalize (@argmax_gen_spec_let A ltbA ltbA_trans ltbA_irref ltbA_total start stop step0 f).
    cbv [in_bounds_alt_at].
    destruct map_reduce_no_init.
    cbn [Nat.pred fst].
    fold step.
    intros [? [n H]]; subst; split; intro H'; subst; [ exists n; repeat apply conj; now apply H | ].
    repeat (destruct_head'_ex; destruct_head'_and; subst).
    repeat match goal with
           | [ H : forall j n, _ /\ _ -> _ |- _ ]
             => specialize (fun n pf => H _ n (conj pf eq_refl))
           end.
    repeat match goal with
           | [ n : nat, H : forall n' : nat, _ |- _ ] => unique pose proof (H n)
           end.
    specialize_by_assumption.
    destruct_head'_or; destruct_head'_and.
    all: lazymatch goal with
         | [ Hirref : Irreflexive _, H : (?x <? ?x) = true |- _ ] => exfalso; eapply Hirref, H
         | [ Hirref : Irreflexive _, H : (?x <? ?y) = true, H' : (?y <? ?x) = true |- _ ] => exfalso; eapply Hirref; refine (_ : is_true (x <? x)); etransitivity; (exact H + exact H')
         | [ Hirref : Irreflexive _, H : (?x <? ?y) = true, H' : ?y = ?x |- _ ] => exfalso; rewrite H' in H; eapply Hirref, H
         | [ Hirref : Irreflexive _, H : (?x <? ?y) = true, H' : ?x = ?y |- _ ] => exfalso; rewrite H' in H; eapply Hirref, H
         | [ H : ?x <= ?y, H' : ?y <= ?x |- _ ] => assert (x = y) by (clear -H H'; lia); clear H H'; subst; try reflexivity
         end.
  Qed.

  Lemma argmin_spec {A} {lebA : has_leb A} {lebA_trans : @Transitive A Classes.leb} {lebA_refl : @Reflexive A Classes.leb} {lebA_total : @Asymmetric A (fun x y => negb (Classes.leb x y))}
    {start stop step0 f v}
    (step : int := if (step0 =? 0) then 1 else step0)
    : @argmin A lebA start stop step0 f = v
      <-> (exists n,
              in_bounds_alt_at start stop step v n
              /\ forall j n',
                in_bounds_alt_at start stop step j n'
                -> ((f v <=? f j) = true)
                   /\ (f j <=? f v -> n <= n')).
  Proof.
    cbv [argmin].
    generalize (@argmin_gen_spec_let A lebA lebA_trans lebA_refl lebA_total start stop step0 f).
    cbv [in_bounds_alt_at].
    destruct map_reduce_no_init.
    cbn [Nat.pred fst].
    fold step.
    intros [? [n H]]; subst; split; intro H'; subst; [ exists n; repeat apply conj; now apply H | ].
    repeat (destruct_head'_ex; destruct_head'_and; subst).
    repeat match goal with
           | [ H : forall j n, _ /\ _ -> _ |- _ ]
             => specialize (fun n pf => H _ n (conj pf eq_refl))
           end.
    repeat match goal with
           | [ n : nat, H : forall n' : nat, _ |- _ ] => unique pose proof (H n)
           end.
    repeat (specialize_by_assumption; destruct_head'_and).
    repeat f_equal; lia.
  Qed.

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

  Lemma argmin_min_equiv
    {A} {lebA : has_leb A}
    {minA : has_min A}
    {start stop step f}
    (Hleb_min_compat : forall x y : A, Classes.min x y = if x <=? y then x else y)
    : @min A minA start stop step f = f (@argmin A lebA start stop step f).
  Proof.
    cbv [min argmin].
    cbv [map_reduce_no_init for_loop_lt Fix Classes.eqb int_has_eqb Classes.one Classes.zero int_has_one int_has_zero Classes.leb Classes.leb int_has_leb int_has_leb int_has_leb nat_has_zero nat_has_one Classes.add int_has_add] in *.
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
    cbv [Classes.sub Classes.mul Classes.zero Classes.add Classes.leb Classes.leb Classes.one Classes.int_div int_has_int_div int_has_sub int_has_leb int_has_add int_has_mul int_has_leb] in *.
    break_match; break_innermost_match_hyps; auto.
    all: cbv [argmin_]; cbn [fst snd]; rewrite Hleb_min_compat; cbv [Classes.leb]; break_innermost_match; try reflexivity.
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
