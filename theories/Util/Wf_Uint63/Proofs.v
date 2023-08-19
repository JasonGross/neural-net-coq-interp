From Coq Require Import Zify ZifyUint63 Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Arith.Classes.Laws Arith.Instances.Laws Default.
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
                end.
         all: try assumption.
         clear H4.
         assert (i = stop) by lia; subst.
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

  Lemma map_reduce_const {A B reduce init start stop step f}
    (step' := if step =? 0 then 1 else step)
    : @map_reduce A B reduce init start stop step (fun _ => f)
      = N.iter (if start <? stop then (1 + (stop - start - 1) // step')%core : int else 0) (fun v => reduce v f) init.
  Proof.
    cbv [map_reduce for_loop_lt Fix Classes.ltb int_has_ltb Classes.int_div int_has_int_div Classes.eqb Classes.sub Classes.zero Classes.one int_has_eqb int_has_sub int_has_zero int_has_one Classes.add int_has_add] in *.
    set (f' := fun (i : int) continue state => _) at 1.
    set (wf := Acc_intro_generator _ _ _); clearbody wf.
    revert start wf init.
    fix IH 2.
    intros ? [wf]; intros; cbn [Fix_F]; cbv [Acc_inv].
    unfold f' at 1.
    set (wf' := wf _); specialize (fun y => IH _ (wf' y)); clearbody wf'; clear wf.
    cbv [Monad.bind run_body LoopNotation.get LoopBody_Monad bind LoopNotation.set LoopNotation.update].
    subst; break_match; try reflexivity.
    1: rewrite IH.
    all: clear IH f'.
    all: clear wf'.
    all: fold step'.
    all: repeat match goal with H : _ |- _ => progress fold step' in H end.
    { break_match; try lia.
      replace ((1 + ((stop - (start + step') - 1) / step'))%uint63) with ((stop - start - 1) / step')%uint63.
      replace ((1 + ((stop - start - 1) / step'))%uint63 : N) with (N.succ ((stop - start - 1) / step')%uint63).
      rewrite N.iter_succ_r; reflexivity.
      all: cbv [coer_int_N'].
      all: assert (Hstep : (step' <> 0)%uint63) by (clear; subst step'; break_match; lia).
      all: assert ((0 <=? stop - start - 1) = true)%uint63 by lia.
      all: assert ((step' <=? stop - start - 1) = true)%uint63 by lia.
      all: assert ((1 <=? (stop - start - 1) / step') = true)%uint63 by nia.
      1: assert (to_Z (stop - start - 1) / to_Z step' <= wB / to_Z step')%Z by nia.
      1: assert (step' = 1 \/ (2 <=? step') = true)%uint63 by lia.
      1: assert ((2 <=? step')%uint63 = true -> wB / to_Z step' <= wB / 2 < wB - 2)%Z by (clear; nia).
      1: lia.
      clearbody step'; clear step.
      replace (stop - (start + step') - 1)%uint63 with ((stop - start - 1) - step')%uint63 by lia.
      generalize dependent (stop - start - 1)%uint63; clear -Hstep; intro i; intros.
      assert (i mod step' = (i + step') mod step')%Z.
      zify.
      (*
      nia.
      repeat match goal with
                  | [ H : ?T -> _, H' : ?T |- _ ] => specialize (H H')
                  | [ H : ?T -> _, H' : ~?T |- _ ] => clear H
                  | [ H : ?T -> _ |- _ ]
                    => let H' := fresh in
                       first [ assert (H' : T) by lia; specialize (H H')
                             | assert (H' : ~T) by lia; clear H ]
             end.
      subst.
      nia.
      2: { Ltac zify_convert_to_euclidean_division_equations_flag ::= constr:(false).
        zify.
        set (start' := to_Z start) in *; clearbody start'; clear start.
        set (stop' := to_Z stop) in *; clearbody stop'; clear stop.
        rename step' into step.
        set (step' := to_Z step) in *; clearbody step'; clear step.

           repeat match goal with
                  | [ H : ?T -> _, H' : ?T |- _ ] => specialize (H H')
                  | [ H : ?T -> _, H' : ~?T |- _ ] => clear H
                  | [ H : ?T -> _ |- _ ]
                    => let H' := fresh in
                       first [ assert (H' : T) by lia; specialize (H H')
                             | assert (H' : ~T) by lia; clear H ]
                  end.
           move q0 at bottom.
           move r1 at bottom.
           all: repeat first [ progress intros
                   | progress subst
                   | rewrite -> !Z.add_assoc in *
                   | match goal with
                     | [ H : (?x - ?y = ?z)%Z |- _ ] => is_var x; assert (x = z + y)%Z by lia; clear H
                     | [ H : (?x + ?y = ?z)%Z |- _ ] => is_var x; assert (x = z - y)%Z by lia; clear H
                     | [ H : (?y + ?x = ?z)%Z |- _ ] => is_var x; assert (x = z - y)%Z by lia; clear H
                     | [ H : context[to_Z ?x] |- _ ] => generalize dependent (to_Z x); clear x; intro x
                     | [ H : context[(Zpos ?x * ?q1 + Zpos ?x * ?q2)%Z] |- _ ]
                       => replace (Zpos x * q1 + Zpos x * q2)%Z with (Zpos x * (q1 + q2))%Z in * by (clear; nia)
                     | [ H : ?x = ?y :> Z |- _ ] => assert_fails constr_eq y Z0; assert (x - y = 0)%Z by lia; clear H
                     end ].
      all: repeat match goal with H : _ |- _ => progress ring_simplify in H end.
           assert (q = 0 \/ q = 1)%Z by lia.
           assert (q0 = 0 \/ q0 = -1)%Z by lia.
           assert (q1 = 0 \/ q1 = -1)%Z by lia.
           assert (q3 = 0 \/ q3 = -1)%Z by lia.
           assert (q4 = 0 \/ q4 = -1)%Z by lia.
           assert (q2 = 0 \/ q2 = 1)%Z by lia.
           destruct_head'_or; subst.
           all: repeat match goal with H : _ |- _ => progress ring_simplify in H end.
           all: rewrite ?Z.mul_0_r in *.
           all: rewrite ?Z.add_0_l in *.
           move step' at bottom.

      all: repeat first [ progress intros
                   | progress subst
                   | rewrite -> !Z.add_assoc in *
                   | rewrite <- !Z.add_opp_r in *
                   | rewrite -> !Z.opp_add_distr in *
                   | rewrite <- !Z.mul_opp_r in *
                   | rewrite -> !Z.opp_involutive in *
                   | match goal with
                     | [ H : context[(Zpos ?x * ?q1 + Zpos ?x * ?q2)%Z] |- _ ]
                       => replace (Zpos x * q1 + Zpos x * q2)%Z with (Zpos x * (q1 + q2))%Z in * by (clear; nia)
                     | [ H : context[(Zpos ?x * ?q1 + ?r + Zpos ?x * ?q2)%Z] |- _ ]
                       => replace (Zpos x * q1 + r + Zpos x * q2)%Z with (Zpos x * (q1 + q2) + r)%Z in * by (clear; nia)
                     end ].
      let wB := (eval cbv in wB) in
      repeat first [ rewrite <- !Z.add_assoc in *
                   | match goal with
                     | [ H : context[(wB * ?q1 + wB * ?q2)%Z] |- _ ]
                       => replace (wB * q1 + wB * q2)%Z with (wB * (q1 + q2))%Z in * by (clear; nia)
                     | [ H : context[(wB * ?q1 + (wB * ?q2 + ?k))%Z] |- _ ]
                       => replace ((wB * q1 + (wB * q2 + k))%Z) with (wB * (q1 + q2) + k)%Z in * by (clear; nia)
                     | [ H : context[(?r + (wB * ?q1 + ?k))%Z] |- _ ]
                       => replace ((r + (wB * q1 + k))%Z) with (wB * q1 + (r + k))%Z in * by lia
                     end ].
      assert (q0 <= 0)%Z by nia.
      match goal with
      | [ H : (Zpos ?x * ?q + ?r = 0)%Z |- _ ] => assert (0 <= r < Zpos x)%Z
      end.
      match goal with
      end.
        => replace (r + Zpos x * q1 + r + Zpos x * q2)%Z with (Zpos x * (q1 + q2) + r)%Z in * by (clear; nia)


      ring_simplify in H9.

      match goal with
      end.
               => specialize (H ltac:(lia)) end.
      assert (
      nia.
      nia.
      assert (stop
        lia.
      all: assert ((stop - start - 1 <? stop - start) = true)%uint63 by lia.
      all: assert ((stop - start) = 1 + (stop - start - 1))%uint63 by lia.
      destruct_head'_or; subst.
      nia.
      {
        clear -H9.
        clearbody step'.
        zify.
        nia.
      { zify.
      assert (
      nia.
      clearbody step'; subst.
      unshelve erewrite (_ : forall x, x / 1 = x)%uint63; [ clear; lia | ].
      zify.
      nia.
      clear.
      zify.
      Set Printing All.
      Print Ltac zify.
      assert (stop - start - 1 = stop - start - 1)%uint63.

      zify.
      pose (start -
      zify.

      lia.
      unshelve erewrite (_ : forall x, x / 1 = x)%uint63; [ clear; lia | ].
      clear; lia.
      replace (?x / 1)%uint63 with x by lia.
      assert (1 + stop - start - 1)
      lia.

      clear f'.
      clearbody step'; subst.
      Set P
      all: d
      all: assert (step' = 1 -> (stop - start - 1) / step' =  < wB / to_Z step')%Z by nia.
      clear -H1; clearbody step'.
      generalize (stop - start - 1)%uint63; clear -H1; intro.
      zify.
      repeat match goal with H : _ |- _ => specialize (H ltac:(lia)) end.
      clear H7.
      generalize dependent (to_Z step'); clear step'; intro step'; intros.
      generalize dependent (to_Z i); clear i; intro i; intros.
      subst.
      vm_compute Z.pow in *.
      nia.
      clear H2.
      move q at bottom.
      assert (r0 =
      assert (q = 0%Z) by nia.
      assert (q0 = 0%Z) by nia.
      assert (r = r0) by nia.
      nia.
      nia.

      assert (to_Z ((stop - start - 1) / step') < wB / to_Z step')%Z by nia.
      zify.
      all: assert ((stop - start - 1) / step' <
      2: nia.
      all: assert
      {
      2: {
      replace ((1 + ((stop - (start + step') - 1) / step'))%uint63 : N) with (N.succ (1 + ((stop - start - 1) / step'))%uint63).
      rewrite N.iter_succ_r.
      match goal with
      | [ |-
      Search N.iter N.succ.
      2: { lia.
    2: {
         assert (Hstep' : step' <> 0%uint63) by lia.
         assert ((stop - start - 1 <? step') = true)%uint63 by lia.
         replace ((stop - start - 1) / step')%uint63 with 0%uint63 by nia.
         reflexivity. }
    rewrite IH.
         cbv.

         vm_compute coer_int_N'.
         assert ((stop - start) / step' = 0 \/ (stop - start) / step' = 1)%uint63 by nia.
         assert ((stop - start) mod step' <> 0 -> (stop - start) / step' = 0)%uint63 by nia.
         assert ((stop - start) / step' = 1 -> (stop - start) = step')%uint63 by nia.
         assert (step' <> 0)%uint63 by nia.
         assert ((stop - start) / step' = 0)%uint63 by nia.

         clear -H H0 Hstep'.
         clear H.
         generalize dependent (stop - start)%uint63; intros.
         clearbody step'.
         clear -Hstep' H0.
         zify.
         specialize (H1 ltac:(lia)).
         clear H2.
         generalize dependent (to_Z i); clear i; intro i; intros.
         generalize dependent (to_Z step'); clear step'; intro step'; intros.
         subst.
         assert (0 <= q <= 1)%Z by nia.
         asser
         assert (step' * q + r <= step')%Z by lia.
         assert (step' * q + r <= step')%Z by lia.
         nia.
         zify.
         nia.
         specialize (H2
         nia.
         nia.
         Set Printing All.
         let v := (eval cbv [step'] in step') in
         repeat match goal with H : _ |- _ => progress change v with step' in H end.
    3: { cbn.
    rewrite IH; clear IH f'.
    subst; break_match; auto.
       *)
  Abort.

  Lemma map_reduce_distr1 {A R reduce init1 init2 start stop step f F}
    : R init2 (F init1)
      -> (forall init init' x,
             R init' (F init)
             -> R (reduce init' (F (f x))) (F (reduce init (f x))))
      -> R (@map_reduce A A reduce init2 start stop step (fun i => F (f i)))
           (F (@map_reduce A A reduce init1 start stop step f)).
  Proof.
    cbv [map_reduce for_loop_lt Fix].
    set (F' := fun (i : int) continue state => _) at 1.
    set (f' := fun (i : int) continue state => _) at 1.
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

    Lemma sum_distr {start stop step} {f g : int -> A}
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

    Lemma mul_sum_distr_l {start stop step} {f : int -> A} {x}
      : R (x * sum start stop step f) (sum start stop step (fun i => x * f i)).
    Proof using R_refl R_trans addA_Proper mulA_addA_distr_l zeroA_mul_r.
      cbv [sum]; apply @map_reduce_distr1 with (F:=fun fi => x * fi).
      { apply zero_r. }
      { intros *.
        rewrite distr_l.
        intros ->; reflexivity. }
    Qed.

    Lemma mul_sum_distr_r {start stop step} {f : int -> A} {x}
      : R (sum start stop step f * x) (sum start stop step (fun i => f i * x)).
    Proof using R_refl R_trans addA_Proper mulA_addA_distr_r zeroA_mul_l.
      cbv [sum]; apply @map_reduce_distr1 with (F:=fun fi => fi * x).
      { apply zero_l. }
      { intros *.
        rewrite distr_r.
        intros ->; reflexivity. }
    Qed.
  End sum.


End Reduction.
