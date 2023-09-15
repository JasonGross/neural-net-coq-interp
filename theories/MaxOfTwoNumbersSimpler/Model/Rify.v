From Coq Require Import Reals ZArith.
From Coq.Floats Require Import Floats.
From Flocq.Core Require Import Raux Generic_fmt Zaux FLX.
From Flocq.IEEE754 Require Import PrimFloat BinarySingleNaN.
From NeuralNetInterp.Util.Tactics Require Import Head.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances.
From NeuralNetInterp.MaxOfTwoNumbersSimpler Require Import Parameters Model Model.Instances Model.Flocqify Model.ExtraComputations Heuristics.
From mathcomp.analysis Require Import Rstruct.
From mathcomp Require Import matrix all_ssreflect all_algebra ssrnum bigop.
From LAProof.accuracy_proofs Require Import dotprod_model.
From vcfloat Require Import IEEE754_extra.
From vcfloat Require Import FPCompCert.
From mathcomp.ssreflect Require Import seq.
From LAProof Require Import dot_acc.
From NeuralNetInterp.Util Require Import Default Arith.Classes Arith.Instances Arith.Instances.Reals Arith.Flocq Arith.Flocq.Instances Arith.Flocq.Definitions.
Import (hints) Instances.Uint63.
Import Dependent.ProperNotations.
Import Arith.Instances.Truncating Arith.Flocq.Instances.Truncating.
#[local] Open Scope core_scope.

Module Model.
  Export Model.Instances.Model.
  Export Model.ExtraComputations.Model.

  Lemma logit_equiv_bounded_no_checkpoint
    (use_checkpoint1:=false)
    (use_checkpoint2:=false)
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (logits1 := logits (use_checkpoint:=use_checkpoint1) tokens1)
    (logits2 := logits (use_checkpoint:=use_checkpoint2) tokens2)
    : Tensor.eqfR
        (fun (x:R) (y:binary_float prec emax) => (abs (x - (y:R)) <=? (logit_rounding_error:R)) = true)
        logits1 logits2.
  Proof.
    intro i.
    repeat match goal with H := _ |- _ => subst H end.
    cbv [Classes.abs Classes.sub Classes.leb R_has_abs R_has_sub R_has_leb].
    cbv [logits cfg.normalization_type].
    cbv [HookedTransformer.HookedTransformer.logits HookedTransformer.HookedTransformer.ln_final PArray.maybe_checkpoint].
    set (resid1 := HookedTransformer.HookedTransformer.resid_postembed _ _ _).
    set (resid2 := HookedTransformer.HookedTransformer.resid_postembed _ _ _).
    cbv [HookedTransformer.HookedTransformer.blocks_cps PArray.maybe_checkpoint].
    vm_compute List.length.
    cbv [List.fold_right].
    cbv [List.firstn HookedTransformer.HookedTransformer.blocks List.map coer_blocks_params cfg.blocks_params cfg.normalization_type].
    set (attn1 := HookedTransformer.TransformerBlock.attn_only_out _ _ _ _ _ _ _ _ _ _ _ _).
    set (attn2 := HookedTransformer.TransformerBlock.attn_only_out _ _ _ _ _ _ _ _ _ _ _ _).
    cbv in i.
    repeat match goal with H : _ * _ |- _ => destruct H end.
    cbv [HookedTransformer.HookedTransformer.unembed HookedTransformer.Unembed.forward map' map reshape_app_combine reshape_app_combine' RawIndex.uncurry_radd get raw_get reshape_app_split tensor_mul Classes.mul Classes.sub tensor_sub Classes.add tensor_add map2 R_has_add RawIndex.split_radd reshape_app_split' RawIndex.curry_radd broadcast broadcast' reshape_app_combine RawIndex.combine_radd RawIndex.snoc RawIndex.nil repeat' binary_float_has_mul binary_float_has_add R_has_mul].
    cbv [get raw_get].
    cbv [coer_tensor Tensor.map coer coer_trans coer_binary_float_R coer_float_binary_float].
    destruct u.
    revert i1 i0 i.
    #[local] Open Scope core_scope.
    intros i1 i0 i.
    set (sum1 := Wf_Uint63.Reduction.sum _ _ _ _).
    set (sum2 := Wf_Uint63.Reduction.sum _ _ _ _).
    assert (default_nan : { x | Binary.is_nan prec emax x = true }).
    { cbv.
      unshelve econstructor; [ unshelve econstructor | ]; [ .. | reflexivity ].
      2: exact true.
      exact 1%positive.
      reflexivity. }
    let v := open_constr:(_) in
    replace sum2 with v.
    2: {
      repeat match goal with H : _ |- _ => assert_fails constr_eq H default_nan; clear H end.
      lazymatch (eval cbv [sum2] in sum2) with
      | Wf_Uint63.Reduction.sum ?start ?stop ?step (fun i => ?mul (@?f i) (@?g i))
        => pose [seq (Binary.BSN2B _ _ default_nan (f (start + step * ((i:nat):PrimInt63.int))%core)) | i <- iota 0 (Z.to_nat (Uint63.to_Z ((stop - start) // step : PrimInt63.int))%core) ] as fs;
           pose [seq (Binary.BSN2B _ _ default_nan (g (start + step * ((i:nat):PrimInt63.int))%core)) | i <- iota 0 (Z.to_nat (Uint63.to_Z ((stop - start) // step : PrimInt63.int))%core) ] as gs;
           pose (@dotprodF _ Tdouble fs gs) as dotp
      end.
      transitivity (Binary.B2BSN _ _ dotp).
      reflexivity.
      subst sum2 dotp.
      cbv [dotprodF].
      admit. }
    cbv beta.
    let v := open_constr:(_) in
    replace sum1 with v.
    2: {
      repeat match goal with H : _ |- _ => clear H end.
      Print dotprodR.
      lazymatch (eval cbv [sum1] in sum1) with
      | Wf_Uint63.Reduction.sum ?start ?stop ?step (fun i => ?mul (@?f i) (@?g i))
        => pose [seq ((f (start + step * ((i:nat):PrimInt63.int))%core)) | i <- iota 0 (Z.to_nat (Uint63.to_Z ((stop - start) // step : PrimInt63.int))%core) ] as fs;
           pose [seq ((g (start + step * ((i:nat):PrimInt63.int))%core)) | i <- iota 0 (Z.to_nat (Uint63.to_Z ((stop - start) // step : PrimInt63.int))%core) ] as gs;
           pose (@dotprodR fs gs) as dotp
      end.
      transitivity dotp.
      reflexivity.
      subst sum1 dotp.
      cbv [dotprodR].
      admit. }
    cbv beta.
    vm_compute Z.to_nat.
    apply Rle_bool_true.
    Set Printing Coercions.
    lazymatch goal with
    | [ |- ?R (Rabs (?sub (?plus' ?x' ?y') (?f (?plus ?x ?y)))) ?small ]
      => cut (R (Rabs (sub x' (f x))) small);
         [ generalize x x' small; replace y with (Prim2B 0); replace y' with R0 | ]
    end.
    (* https://github.com/VeriNum/LAProof/blob/main/accuracy_proofs/float_acc_lems.v
BPLUS_B2R_zero_r *)
    admit.
    admit.
    admit.
    admit.
    rewrite Binary.B2R_B2BSN.
    epose Rabs_triang.
    lazymatch goal with
    | [ |- ?R (Rabs (?sub ?y (?b2r (dotprodF ?x ?x')))) ?small ]
      => epose proof (@dotprod_forward_error _ Tdouble x x');
         destruct (@dotprod_mixed_error _ Tdouble x x')
    end.
    3: { repeat match goal with H : ex _ |- _ => destruct H | H : and _ _ |- _ => destruct H end.
         cbv [FT2R] in *.
         vm_compute fprec in *.
         vm_compute femax in *.
         let H := match goal with H : Binary.B2R _ _ (dotprodF _ _) = _ |- _ => H end in
         rewrite H; clear H.
         move x at bottom.
         move x0 at bottom.


(*
         match goal with
         | [ |- ?R (?abs (?sub (dotprodR ?x ?y

         Set Printing All.
         Search dotprodR.
         Set Printing All.
         Search

        cut (R (Rabs (sub (f x) x')) small);
         [ generalize x x' small; replace y with (Prim2B 0); replace y' with R0 | ]
    end.

    Search B2R Binary.B2BSN.

    Search Bplus.


      move sum1 at bottom.
      inst

      instantiate (1:=Binary.B2BSN _ _ dotp).
      Search BinarySingleNaN.binary_float Binary.binary_float.
      Print Binary.binary_float.
      Print binary_float.
      vm_compute binary_float in fs.
      vm_compute ftype in f.


      Locate Binary.binary_float.
      Set Printing All.
      Print nans.
      Print Nans.
      Print type.
      Check dotprodF.
      Locate FPCore.Nans.
      Check (dotprodF
      epose @dotprodF.
      Print vcfloat.FPCore.type.

      Locate float.
      Search  float.
      Locate type.
      Print type.
      instantiate (1:=(dotprod
    assert (

    revert

    cbv [RawIndexType].
    cbv [cfg.b_U].

    vm_compute cfg.b_U.
    Import Wf_Uint63.
    cbv [].
    vm_compute
    destruct_hea
    destruct i as [
    Set Printing Coercions.
    c
    cbv beta iota delta [acc_fn].
    Set Printing Implicit.*)
  Admitted. (* XXX FIXME *)

  Lemma acc_fn_equiv_bounded_no_checkpoint
    (use_checkpoint1:=false)
    (use_checkpoint2:=false)
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (logits1 := logits (use_checkpoint:=use_checkpoint1) tokens1)
    (logits2 := logits (use_checkpoint:=use_checkpoint2) tokens2)
    (acc1 := acc_fn (use_checkpoint:=use_checkpoint1) logits1 tokens1)
    (acc2 := acc_fn (use_checkpoint:=use_checkpoint2) logits2 tokens2)
    : Tensor.eqfR
        (fun (x:R) (y:binary_float prec emax) => (abs (x - (y:R)) <=? (total_rounding_error:R)) = true)
        acc1 acc2.
  Proof.
    intro i.
    repeat match goal with H := _ |- _ => subst H end.
    cbv [Classes.abs Classes.sub Classes.leb R_has_abs R_has_sub R_has_leb].
    cbv [acc_fn].
    cbv beta iota delta [acc_fn].
  Admitted. (* XXX FIXME *)

  Lemma acc_fn_equiv_bounded
    {use_checkpoint1 use_checkpoint2}
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (logits1 := logits (use_checkpoint:=use_checkpoint1) tokens1)
    (logits2 := logits (use_checkpoint:=use_checkpoint2) tokens2)
    (acc1 := acc_fn (use_checkpoint:=use_checkpoint1) logits1 tokens1)
    (acc2 := acc_fn (use_checkpoint:=use_checkpoint2) logits2 tokens2)
    : Tensor.eqfR
        (fun (x:R) (y:binary_float prec emax) => (abs (x - (y:R)) <=? (total_rounding_error:R)) = true)
        acc1 acc2.
  Proof.
    intro i.
    rewrite <- (acc_fn_equiv_bounded_no_checkpoint i).
    apply f_equal2; [ | reflexivity ].
    apply f_equal.
    apply f_equal2.
    all: repeat match goal with H := _ |- _ => subst H end.
    all: try apply f_equal.
    all: eapply acc_fn_Proper_dep.
    all: try solve [ repeat intro; subst; constructor ].
    all: try now apply all_tokens_Proper.
    all: eapply logits_Proper_dep.
    all: try solve [ repeat intro; subst; constructor ].
    all: try now apply all_tokens_Proper.
  Qed.

  Lemma all_tokens_residual_error_equiv_bounded_no_checkpoint
    (use_checkpoint1:=false)
    (use_checkpoint2:=false)
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (resid_postembed1 := resid_postembed (use_checkpoint:=use_checkpoint1) tokens1)
    (resid_postembed2 := resid_postembed (use_checkpoint:=use_checkpoint2) tokens2)
    (resid_err1 := Unembed.forward resid_postembed1)
    (resid_err2 := Unembed.forward resid_postembed2)
    : Tensor.eqfR
        (fun (x:R) (y:binary_float prec emax) => (abs (x - (y:R)) <=? (residual_error_rounding_error:R)) = true)
        resid_err1
        resid_err2.
  Proof.
    intro i.
    repeat match goal with H := _ |- _ => subst H end.
    cbv [Classes.abs Classes.sub Classes.leb R_has_abs R_has_sub R_has_leb].
  Admitted. (* XXX FIXME *)

  Lemma all_tokens_residual_error_equiv_bounded
    {use_checkpoint1 use_checkpoint2}
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (resid_postembed1 := resid_postembed (use_checkpoint:=use_checkpoint1) tokens1)
    (resid_postembed2 := resid_postembed (use_checkpoint:=use_checkpoint2) tokens2)
    (resid_err1 := Unembed.forward resid_postembed1)
    (resid_err2 := Unembed.forward resid_postembed2)
    : Tensor.eqfR
        (fun (x:R) (y:binary_float prec emax) => (abs (x - (y:R)) <=? (residual_error_rounding_error:R)) = true)
        resid_err1
        resid_err2.
  Proof.
    intro i.
    rewrite <- (all_tokens_residual_error_equiv_bounded_no_checkpoint i).
    apply f_equal2; [ | reflexivity ].
    apply f_equal.
    apply f_equal2.
    all: repeat match goal with H := _ |- _ => subst H end.
    all: try apply f_equal.
    (*all: eapply Unembed.forward_Proper_dep.
    all: try solve [ repeat intro; subst; constructor ].
    all: try now apply all_tokens_Proper.
    all: eapply logits_Proper_dep.
    all: try solve [ repeat intro; subst; constructor ].
    all: try now apply all_tokens_Proper.*)
  Admitted.

  Lemma logit_delta_equiv_bounded
    {use_checkpoint1 use_checkpoint2}
    (tokens1 := all_tokens (use_checkpoint:=use_checkpoint1))
    (tokens2 := all_tokens (use_checkpoint:=use_checkpoint2))
    (logits1 := HookedTransformer.logits (use_checkpoint:=use_checkpoint1) tokens1)
    (logits2 := HookedTransformer.logits (use_checkpoint:=use_checkpoint2) tokens2)
    (logit_delta1 := logit_delta (use_checkpoint:=use_checkpoint1) tokens1 logits1)
    (logit_delta2 := logit_delta (use_checkpoint:=use_checkpoint2) tokens2 logits2)
    : (fun (x:R) (y:binary_float prec emax) => (abs (x - (y:R)) <=? (logit_delta_rounding_error:R)) = true)
        logit_delta1
        logit_delta2.
  Proof.
  Admitted.
End Model.
