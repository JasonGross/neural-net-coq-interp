From Coq Require Import Morphisms RelationClasses RelationPairs Relation_Definitions.
From Coq Require Vector.
From Coq.Structures Require Import Equalities.
From Coq Require Import Floats Uint63 ZArith NArith.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead.
From NeuralNetInterp.Util Require Import Default SolveProperEqRel Option Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Util.List.Instances Require Import Forall2 Forall2.Map.
From NeuralNetInterp.Torch Require Import Tensor Tensor.Instances Slicing Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
From NeuralNetInterp.TransformerLens.HookedTransformer Require Import Config Instances Module Module.Instances.
From NeuralNetInterp.TrainingComputations Require Import interp_max_utils.
Import Dependent.ProperNotations Dependent.RelationPairsNotations.
Import Instances.Truncating Instances.Uint63.
#[local] Open Scope tensor_scope.
#[local] Open Scope core_scope.

Module ModelComputationsInstances (cfg : Config) (Import Model : ModelSig cfg) (Import ModelInstances : ModelInstancesSig cfg Model) (Import ModelComputations : ModelComputationsSig cfg Model).
  Import (hints) cfg.

  #[export] Instance logit_delta_Proper_dep
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR
           ==> Dependent.idR)
        (@logit_delta).
  Proof.
    cbv [logit_delta]; repeat intro;
      repeat lazymatch goal with
        | [ H : (?x =? ?y) = true, H' : (?x =? ?y') = false |- ?G ]
          => exfalso; cut (y = y');
             [ revert H H'; generalize x y y'; clear; intros; subst; congruence
             | clear H H'; now HookedTransformer.t ]
        | _ => HookedTransformer.t_step
        end.
  Qed.
End ModelComputationsInstances.

Module Type ModelComputationsInstancesSig (cfg : Config) (Model : ModelSig cfg) (ModelInstances : ModelInstancesSig cfg Model) (ModelComputations : ModelComputationsSig cfg Model) := Nop <+ ModelComputationsInstances cfg Model ModelInstances ModelComputations.
