From Coq Require Import Zify ZifyUint63 Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Arith.Classes.Laws Arith.Instances.Laws Arith.Instances.Zify Default.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead UniquePose.
From NeuralNetInterp.Util Require Import Wf_Uint63 Wf_Uint63.Proofs.
Import Arith.Classes Arith.Instances.Uint63.
From mathcomp.analysis Require Import Rstruct.
From mathcomp Require Import matrix all_ssreflect all_algebra ssrnum bigop.
#[local] Open Scope core_scope.
#[local] Delimit Scope Z_scope with coq_Z.

Module Reduction.
  Export Wf_Uint63.Reduction.
  Export Wf_Uint63.Proofs.Reduction.

  Lemma sum_equiv_ext {R : ringType} {zeroA : has_zero R} {addA : has_add R}
    {start stop step0} {n}
    (step : PrimInt63.int := (if (step0 =? 0) then 1 else step0)%core)
    {f : _ -> R} {g : ordinal n -> R}
    (H0: 0%core = 0%R)
    (Hfg : forall i o, i = start + step * Uint63.of_Z (nat_of_ord o) -> f i = g o)
    (Hadd : forall x y, x + y = (x + y)%R)
    (Hn : n = if (start <? stop)%uint63
              then S (Z.to_nat (Uint63.to_Z ((stop - start - 1) // step))%core)
              else 0%nat)
    : sum start stop step0 f = (\sum_j g j)%R.
  Proof.
    subst n; revert g Hfg.
    cbv [Reduction.sum].
    eapply (map_reduce_spec_count (fun n _ v => forall (g : 'I_n -> R), (forall i o, i = start + step * Uint63.of_Z (nat_of_ord o) -> f i = g o) -> v = (\sum_j g j)%R)).
    { move => *; now rewrite big_ord0. }
    { intros * Hg Hstart Hi Hst g Hfg.
      rewrite big_ord_recr -Hg //=.
      { erewrite Hfg => //. }
      { move => *; subst.
        apply Hfg.
        cbn; reflexivity. } }
  Qed.

  Lemma sum_equiv {R : ringType} {zeroA : has_zero R} {addA : has_add R}
    {start stop step0}
    (step : PrimInt63.int := (if (step0 =? 0) then 1 else step0)%core)
    {f : _ -> R}
    (H0: 0%core = 0%R)
    (Hadd : forall x y, x + y = (x + y)%R)
    (n := if (start <? stop)%uint63
          then S (Z.to_nat (Uint63.to_Z ((stop - start - 1) // step))%core)
          else 0%nat)
    (g : 'I_n -> R
     := fun o => f (start + step * Uint63.of_Z (nat_of_ord o)))
    : sum start stop step0 f = (\sum_j g j)%R.
  Proof.
    apply sum_equiv_ext; auto; intros; subst; reflexivity.
  Qed.
End Reduction.
