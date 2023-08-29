From Coq Require Import Zify ZifyUint63 Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Arith.Classes.Laws Arith.Instances.Laws Arith.Instances.Zify Default.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose.
From NeuralNetInterp.Util Require Import Wf_Uint63.
Import Arith.Classes Arith.Instances.Uint63.
#[local] Open Scope core_scope.

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

  Lemma map_reduce_no_init_spec {A reduce start0 stop step0 f} (P : int -> A -> Prop)
    (step : int := if (step0 =? 0) then 1 else step0)
    (start : int := start0 + step)
    (Hinit : P start (f start0))
    (Hstep : forall i v, P i v -> ((i + step) <? stop) = true -> P (i + step) (reduce v (f i)))
    : P (start + step * (if start <? stop then (stop - start - 1) // step else 0))
        (@map_reduce_no_init A reduce start0 stop step0 f).
  Proof.
    cbv [map_reduce_no_init start].
    eapply for_loop_lt_invariant_gen
      with (P:=fun i v
               => P i v
                  /\ ((i - start) mod step) = 0
                  /\ (start <=? i) = true
                  /\ ((start <? stop) = true -> i = start \/ ((i - step) <? stop) = true)
                  /\ ((start <? stop) = false -> i = start)).
    all: cbv [run_body Monad.bind LoopBody_Monad bind LoopNotation.get LoopNotation.set LoopNotation.update Classes.modulo Classes.sub Classes.add Classes.ltb int_has_modulo int_has_sub int_has_ltb int_has_leb Classes.leb int_has_add Classes.eqb int_has_eqb Classes.zero Classes.one int_has_zero int_has_one int_has_int_div Classes.int_div Classes.mul int_has_mul] in *.
    { repeat apply conj; try assumption; subst step start; break_innermost_match.
      all: lazymatch goal with
           | [ |- ((?x mod ?y) = 0)%uint63 ]
             => first [ replace x with y by lia | replace x with 0%uint63 by lia ];
                assert (y / y = 1 /\ 0 / y = 0)%uint63; nia
           | _ => try lia
           end. }
    3: { subst step start; break_innermost_match; intros; destruct_head'_and.
         all: repeat match goal with H : _ |- _ => specialize (H eq_refl) end.
         all: destruct_head'_or; subst.
         all: try match goal with
                | [ H : ?P ?x ?v |- ?P ?y ?v ] => replace y with x by lia
                | [ H1 : (?x - 1 <? ?y)%uint63 = true, H2 : (?x <? ?y)%uint63 = false |- _ ] => assert (x = y) by (clear -H1 H2; lia); clear H1 H2; subst
                end.
         all: try assumption.
         (*
         3: {
         assert (start = (start + step0) - step0)%uint63 by lia.
         generalize dependent (start + step0)%uint63; intro start'; intros; subst.
         assert (i = start + (i - start))%uint63 by lia.
         generalize dependent (i - start)%uint63; intro i'; intros; subst.
         assert (stop = start + (stop - start))%uint63 by lia.
         generalize dependent (stop - start)%uint63; intro len; intros; subst.
         match goal with
         | [ H : (?x mod ?y = 0)%uint63 |- _ ]
           => let H' := fresh in
              is_var x;
              assert (y <> 0%uint63) by lia;
              assert (x = y * (x / y))%uint63 by (clear -H H'; nia);
              clear H;
              generalize dependent (x / y)%uint63;
              let x' := fresh in
              rename x into x';
              intro x; intros; subst
         end.
         assert (len = (len - 1) + 1)%uint63 by lia.
         generalize dependent (len - 1)%uint63; intro len'; intros; subst.
         assert ((step0 <=? step0 * i')%uint63 = true) by lia.
         assert (len' = step0 * (len' / step0) + (len' mod step0))%uint63 by (clear; nia).
         generalize dependent (len' / step0)%uint63; intro len; intros; subst.
         assert ((0 <=? len' mod step0) = true)%uint63 by (clear; nia).
         let H := lazymatch goal with H : step0 <> 0%uint63 |- _ => H end in
         assert ((len' mod step0 <? step0) = true)%uint63 by (clear -H; nia).
         generalize dependent (len' mod step0)%uint63; intro lenmod; intros; subst.
         assert ((step0 * i' - step0 <? step0 * len + lenmod + 1) = true)%uint63 by lia.
         match goal with
                | [ H : ?P ?x ?v |- ?P ?y ?v ] => replace y with x by nia
                end.
         assert (step0
         nia.
         zify.
         assert (
         assert (
         match goal with
                | [ H : ?P ?x ?v |- ?P ?y ?v ] => replace y with x by nia
                end.
         assert (
         nia.

         Ltac zify_convert_to_euclidean_division_equations_flag ::= constr:(false).
         zify.
         rewrite Z.div_1_r.
         rewrite Z.mul_1_l.
         rewrite !Z.mod_mod by lia.

         Ltac zify_convert_to_euclidean_division_equations_flag ::= constr:(true).
         zify.
         nia.
         lia.

         Z.to_euclidean_division_equations_
         zify.

         zify.
         2: lia.
         zify.
      lia.
      lia.
      lia.
      lia.
      { clear.
        subst step; break_innermost_match.
        nia.
        transitivity (step0 mod step0)%uint63; [ f_equal; lia | ].
        assert (step0 / step0 = 1)%uint63; nia. }
      subst step.

    }

      {
      zify.
      clear H3 H2.
      clear H1.
      specialize (H0 ltac:(lia)).
      specialize (H ltac:(lia)).
      generalize dependent (to_Z step0); clear step0; intros.

      nia.

      nia.
      zify; lia.
      zify.
      nia.
      zify. nia.
      zify; Z.to_euclidean_division_equations; try nia.
      break_innermost_match_hyps; subst; zify; try lia.
      generalize dependent (to_Z step0); clear step0; intro step0; intros.
      generalize dependent (to_Z step); clear step; intro step; intros.
      generalize dependent (to_Z start); clear start; intro start; intros.
      subst.
      repeat match goal with H : _ |- _ => specialize (H ltac:(lia)) end.
      assert (step0 <> 0)%Z by lia.
      clear Heqb H3 H2.
      clear H1.
      clear H7 H6.
      clear H11 H12.
      subst.
      clear H8.
      clear H13.
      assert (r1 = start + step0 - 9223372036854775808 * q1)%Z by lia.
      clear H9.
      subst.
      assert ((step0 * (1 - q) - 9223372036854775808 * (q1 + q0) - ( r)%Z) = 0%Z)%Z by nia.
      clear H4.
      assert (0 <= q1 <= 1)%Z by nia.
      assert (0 <= q0 + q1 <= 1)%Z by nia.
      assert (-1 <= q0 <= 0)%Z by nia.
      assert (0 <= q <= 1)%Z by nia.
      assert (0 = q \/ 1 = q)%Z by lia.
      destruct_head'_or; subst; lia.
      assert (q0 + q1 = 0%Z) by nia.
      assert (q0 = 0) by nia.
      assert (q1 = 0) by nia.
      nia.
      subst.
      clear H3 H2 H1.
      clear Heqb cstr.
      lia.
      assert (r

      zify.
      zify.

      Set Printing All.
      zify.
      assert (step0 = 0%Z).
      Set Printing All.
      zify.
      Print Ltac Zify.zify.

      { clear H13.
        repeat match goal with H : _ |- _ => clear H; assert_succeeds (zify; [ | ]) end.
        exfalso.
        repeat match goal with H : _ |- _ => clear H; assert_succeeds (zify; [ | ]) end.
        generalize dependent (@eqb int int_has_eqb); intro f; intros.
        rename step0 into x.
        assert
        revert x f cstr4.
  zify.
  lia.


        zify.
        assert (step0 = 0%uint63).
        Set Printing All.
        zify.
        zify.
        zify.
        zify.
        zify.
        Zify.zify_op.
        Zify.zify_op.
      Print Ltac zify.
      zify.
      zify.
      zify.
      zify.
      zify.
      zify.
      zify.
    Set Printing All.

    eassumption.
    Print for_loop_lt.
    revert dependent start.
          *)
  Abort.

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

  Lemma argmax_gen_spec {A} {ltbA : has_ltb A} {start stop step f i v}
    (Hstep : step <> 0)
    : map_reduce_no_init (@argmax_ int A ltbA) start stop step (fun i : int => (i, f i)) = (i, v)
      <-> (v = f i /\ forall j, (f j <? f i) = true).
  Proof.
    assert (to_Z step <> 0) by now intro; apply Hstep; apply to_Z_inj.
    cbv [map_reduce_no_init].
    set (v' := (start, f start)).
    assert (Hv'' : f (fst v') = snd v' /\ forall j, (j <? ((start + step) - start) // step) = true -> let fv' := f (start + j * step) in fv' = snd v' \/ (fv' <? snd v') = true).
    { split; [ reflexivity | intros j H'; left; subst v'; cbn; apply f_equal, to_Z_inj ].
      cbv in H'; revert H'.
      rewrite !ltb_spec, !div_spec, !sub_spec, !add_spec, !mul_spec.
      repeat match goal with |- context[to_Z ?x] => unique pose proof (to_Z_bounded x) end.
      rewrite Zminus_mod_idemp_l, Z.add_simpl_l, Z.mod_small by lia.
      rewrite Z.div_same by assumption.
      intro; assert (Hz : to_Z j = to_Z 0) by (transitivity 0%Z; try reflexivity; lia).
      apply to_Z_inj in Hz; subst.
      rewrite to_Z_0, Z.mul_0_l, Z.mod_0_l, Z.add_0_r, Z.mod_small by lia.
      reflexivity. }
    clearbody v'.
(*
      rewrite
      left.
      f_equal.
      2: {
      rewrite Z.mod_small
      Search (?x + ?y - ?x)%Z.
      Search (((_ mod _) - _) mod _)%Z.
      push_Zmod.
      intros.

      intros
    set (fv := f start).
    assert (fv
    apply for_loop_lt_invariant_gen with (P := fun i' v => forall j,
    cbv [argmax].
    cut
  Lemma argmax_gen_spec {A} {ltbA : has_ltb A} {start stop step f i v}
    : map_reduce_no_init (@argmax_ int A ltbA) start stop step (fun i : int => (i, f i)) = (i, v)
      <-> (v = f i /\ forall j, (f j <? f i) = true).
  Proof.
    cbv [map_reduce_no_init].
    apply for_loop_lt_invariant_gen.
    cbv [argmax].
    cut
  Lemma argmax_spec {A} {ltbA : has_ltb A} {start stop step f v}
    : @argmax A ltbA start stop step f = v
      <-> forall j, (f j <? f v) = true.
  Proof.
    cbv [argmax].
    cut
    cbv [map_reduce_no_init].

  Lemma argmax_spec {A} {ltbA : has_ltb A} {start stop step f v}
    : @argmax A ltbA start stop step f = v
      <->
    (start : int) (stop : int) (step : int) (f : int -> A) : int
    := fst (map_reduce_no_init argmax_ start stop step (fun i => (i, f i))).
 *)
  Abort.

  Definition in_bounds (start stop step i : int) : bool
    := ((start <=? i) && ((i =? start) || (i <? stop)) && (((i - start) mod step) =? 0)).

  Lemma argmax_spec {A} {ltbA : has_ltb A} {start stop step f v}
    : @argmax A ltbA start stop step f = v
      <-> (in_bounds start stop step v = true
           /\ forall j,
              in_bounds start stop step j = true
              -> (((f j <? f v) = true)
                  \/ (f j = f v /\ (v <=? j) = true))).
  Proof.
    (* XXX FIXME *)
  Admitted.

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
      {mulA_addA_distr_r : RightDistributive R mul add}.

    Lemma sum_const [start stop step] (f : A)
      : sum start stop step (fun x => f)
        = N.iter
            (if start <? stop
             then 1 + ((stop - start - 1) // (if step =? 0 then 1 else step) : int)
             else 0) (fun v : A => v + f) 0.
    Proof using Type.
      cbv [sum]. rewrite map_reduce_const; reflexivity.
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
