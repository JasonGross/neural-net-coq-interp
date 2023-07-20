From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.
From NeuralNetInterp.Torch Require Import Tensor.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances.
From NeuralNetInterp.MaxOfTwoNumbers Require Import Parameters Model Heuristics TheoremStatement Model.Instances.
Import LoopNotation.
(*From NeuralNetInterp.MaxOfTwoNumbers.Computed Require Import AllLogits.*)
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Local Ltac let_bind_1 _ :=
  match goal with
  | [ |- context G[let n := ?v in @?f n] ]
    => let n' := fresh n in
       set (n' := v); let G := context G[f n'] in change G; cbv beta
  end.

Local Ltac let_bind_hyp _ :=
  match goal with
  | [ H := context G[let n := ?v in @?f n] |- _ ]
    => let n' := fresh n in
       set (n' := v) in (value of H); let G := context G[f n'] in change G in (value of H); cbv beta in H
  end.

Local Ltac let_bind _ := repeat first [ progress cbv beta iota in * | let_bind_1 () | let_bind_hyp () ].

Theorem good_accuracy : TheoremStatement.Accuracy.best (* (abs (real_accuracy - expected_accuracy) <? error)%float = true *).
Proof.
  cbv [real_accuracy].
  cbv beta iota delta [acc_fn]; let_bind ().
  cbv beta iota delta [logits] in *; let_bind ().
  repeat match goal with H : Shape _ |- _ => subst H end.
  cbv beta iota delta [HookedTransformer.logits] in *; let_bind ().
  repeat match goal with H : Shape _ |- _ => subst H end.
  cbv beta iota delta [blocks_params] in *.
  cbv beta iota delta [HookedTransformer.blocks_cps fold_right HookedTransformer.blocks List.map] in *; let_bind ().
  repeat match goal with H : Shape _ |- _ => subst H end.
  vm_compute Shape.tl in *.
  vm_compute of_Z in *.
  vm_compute SliceIndex.transfer_shape in *.
  vm_compute Shape.app in *.
  vm_compute Shape.broadcast2 in *.
  cbv beta iota delta [TransformerBlock.attn_only_out] in *; let_bind ().
  subst maybe_n_heads.
  cbv beta iota delta [Attention.attn_out] in *; let_bind ().
  cbv beta iota delta [Attention.z] in *; let_bind ().
  set (v := Attention.v _ _ _) in *.
  set (pattern := Attention.pattern _ _ _ _ _ _) in *.
  cbv beta iota delta [HookedTransformer.unembed] in *; let_bind ().
  repeat match goal with H : Shape _ |- _ => subst H end.
  cbv beta iota delta [Unembed.forward] in *; let_bind ().
  repeat match goal with H : Shape _ |- _ => subst H end.
  cbv [item mean].
    cbv [reduce_axis_m1 reduce_axis_m1' reshape_snoc_split map RawIndex.curry_radd reshape_m1].
  cbv [raw_get].
  cbv [RawIndex.unreshape].
  cbv [RawIndex.item].
  cbv [RawIndex.tl].
  cbv [RawIndex.combine_radd].
  vm_compute of_Z.
  cbv [RawIndex.unreshape'].
  cbv [Shape.tl].
  cbn [snd].
  cbv [Shape.snoc].
  cbv [RawIndex.snoc].
  cbv [RawIndex.nil].
  cbn [fst snd].
  cbv [Reduction.mean].
  vm_compute Truncating.coer_Z_float.

  rewrite FloatAxioms.ltb_spec, FloatAxioms.abs_spec, FloatAxioms.sub_spec, FloatAxioms.div_spec.
  vm_compute (Prim2SF error).
  vm_compute (Prim2SF expected_accuracy).
  cbv [SFltb SFabs SFcompare SF64sub].
  cbv [
  Search PrimFloat.of_Z.
  Search SFltb.
  Print SFltb.
  Search PrimFloat.abs.



  Set Debug "Cbv".
  cbv -[PrimFloat.ltb PrimFloat.abs PrimFloat.sub Reduction.mean res expected_accuracy error].


  Set Printing
  cbv [
  set (mean_res := item (mean res)).

  let T := open_constr:(_) in
  evar (mean_res' : T);
  replace mean_res with mean_res'; revgoals; [ symmetry | ].
  { subst mean_res mean_res'.

    apply Tensor.item_Proper.
    apply Tensor.mean_Proper.
    subst res.
    apply Tensor.PArray.checkpoint_Proper.
    apply Tensor.of_bool_Proper.
    apply map2_Pr
  Typeclasses eauto := debug.
  setoid_replace (P
  rewrite PArray.checkpoint_correct.


  cbn [Shape.tl Shape.snoc] in *.
  cbv

  rewrite <- computed_accuracy_eq.
  vm_compute; reflexivity.
Qed.
