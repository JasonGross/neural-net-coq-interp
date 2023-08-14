From Coq Require Import Morphisms RelationClasses RelationPairs List.
From NeuralNetInterp.Util.List Require Import Forall2Map.
From NeuralNetInterp.Util Require Import Morphisms.Dependent RelationClasses.Dependent Program.Basics.Dependent.
Import Dependent.ProperNotations.

Module List.
  #[export] Instance map_Proper_dep
  : Dependent.Proper2
      ((Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 Dependent.idR)
         ==> Dependent.lift2_1 (List.Forall2 ∘ Dependent.idR)
         ==> Dependent.lift2_2 (List.Forall2 ∘ Dependent.idR))
      (@List.map).
  Proof.
    cbv -[List.map]; intros; rewrite List.Forall2_map.
    eapply Forall2_impl; [ | eassumption ]; eauto.
  Qed.
End List.
Export (hints) List.
