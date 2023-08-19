From Coq Require Import Zify ZifyUint63 Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Default.
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
End Reduction.
