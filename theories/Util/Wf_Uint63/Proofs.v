From Coq Require Import Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Default.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose.
From NeuralNetInterp.Util Require Import Wf_Uint63.
Import Arith.Classes.
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

End Reduction.
