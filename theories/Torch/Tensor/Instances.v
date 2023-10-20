From Coq.Structures Require Import Equalities.
From Coq Require Import ZArith Sint63 Uint63 List PArray Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Nat.
From NeuralNetInterp.Util Require Import Wf_Uint63 Wf_Uint63.Instances PArray.Proofs PArray.Instances List.Proofs Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool (*PrimitiveProd*).
From NeuralNetInterp.Util.Tactics Require Import BreakMatch.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Hetero Relation_Definitions.Dependent.
From NeuralNetInterp.Util.Classes Require Morphisms.Dependent RelationClasses.Dependent.
From NeuralNetInterp.Torch Require Import Tensor.
Import Dependent.ProperNotations.
Import (hints) Morphisms.Dependent RelationClasses.Dependent.

Module Tensor.
  #[export] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
  Export (hints,coercions) Torch.Tensor.
  #[export] Set Warnings Append "uniform-inheritance,ambiguous-paths".
  Definition eqfR_rank {r} : Dependent.relation (@tensor_of_rank r)
    := fun A B R x y => forall i, R (x i) (y i).
  #[global] Arguments eqfR_rank {r} [A B] R x y.
  Notation eqf_rank := (eqfR_rank eq).
  Definition eqfR {r s} : Dependent.relation (@tensor r s)
    := eqfR_rank.
  #[global] Arguments eqfR {r s} [A B] R x y.
  Notation eqf := (eqfR eq).

  #[export] Instance eqf_Reflexive_dep {r s} : Dependent.Reflexive (@eqfR r s).
  Proof. repeat intro; subst; reflexivity. Qed.
  #[export] Instance eqf_Reflexive {r s A R} {_ : Reflexive R} : Reflexive (@eqfR r s A A R)
    := _.
  #[export] Instance eqf_Symmetric_dep {r s} : Dependent.Symmetric (@eqfR r s).
  Proof. cbv; repeat intro; eauto. Qed.
  #[export] Instance eqf_Symmetric {r s A R} {_ : Symmetric R} : Symmetric (@eqfR r s A A R).
  Proof. cbv; repeat intro; subst; symmetry; auto. Qed.
  #[export] Instance eqf_Transitive_dep {r s} : Dependent.Transitive (@eqfR r s).
  Proof. intros A B C RAB RBC RAC HT x y z H1 H2; repeat intro; subst; eauto. Qed.
  #[export] Instance eqf_Transitive {r s A R} {_ : Transitive R} : Transitive (@eqfR r s A A R).
  Proof. intros x y z H1 H2; repeat intro; subst; etransitivity; [ eapply H1 | eapply H2 ]; reflexivity. Qed.

  #[export] Instance eqfR_Proper_flip {r s A R1 R2 R3}
    {HP : Proper (R1 ==> R2 ==> Basics.flip Basics.impl) R3}
    : Proper (@Tensor.eqfR r s A A R1 ==> @Tensor.eqfR r s A A R2 ==> Basics.flip Basics.impl) (@Tensor.eqfR r s A A R3).
  Proof. repeat intro; eapply HP; eauto. Qed.

  #[export] Instance eqfR_Proper {r s A R1 R2 R3}
    {HP : Proper (R1 ==> R2 ==> Basics.impl) R3}
    : Proper (@Tensor.eqfR r s A A R1 ==> @Tensor.eqfR r s A A R2 ==> Basics.impl) (@Tensor.eqfR r s A A R3).
  Proof. repeat intro; eapply HP; eauto. Qed.

  #[export] Instance eqf_Proper_flip {r s A}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Basics.flip Basics.impl) (@Tensor.eqfR r s A A eq)
    := _.
  #[export] Instance eqf_Proper {r s A}
    : Proper (Tensor.eqf ==> Tensor.eqf ==> Basics.impl) (@Tensor.eqfR r s A A eq)
    := _.

  Module PArray.
    Import Tensor.PArray.
    #[export] Instance concretize_Proper {r s A default} : Proper (eqf ==> eq) (@concretize r s A default).
    Proof.
      cbv [eqf Proper respectful]; revert A default s; induction r; cbn [concretize]; intros A default s t1 t2 H; auto; [].
      destruct s.
      eapply IHr; repeat intro; subst.
      apply PArray.init_default_Proper; try reflexivity; repeat intro; subst.
      apply H.
    Qed.

    #[export] Instance reabstract_Proper {r s A} : Proper (pointwise_relation _ eqf ==> eq ==> eqf) (@reabstract r A s).
    Proof. cbv [reabstract pointwise_relation eqf eqf_rank]; repeat intro; subst; destruct andb; eauto. Qed.

    #[export] Instance checkpoint_Proper {r s A default} : Proper (eqf ==> eqf) (@checkpoint r s A default).
    Proof. cbv [checkpoint]; repeat intro; subst; apply reabstract_Proper; try apply concretize_Proper; repeat intro; auto. Qed.

    Definition checkpoint_correct_eqf {r s A default} t : eqf (@checkpoint r s A default t) t
      := fun idxs => checkpoint_correct.

    #[export] Instance checkpoint_Proper_dep {r s} : Dependent.Proper (Dependent.idR ==> eqfR ==> eqfR) (@checkpoint r s).
    Proof. repeat intro; rewrite !checkpoint_correct_eqf; auto. Qed.

    #[export] Instance maybe_checkpoint_Proper {r s A default use_checkpoint} : Proper (eqf ==> eqf) (@maybe_checkpoint r s A default use_checkpoint).
    Proof. cbv [maybe_checkpoint]; break_innermost_match; try exact _; repeat first [ assumption | intro ]. Qed.

    Definition maybe_checkpoint_correct_eqf {r s A default use_checkpoint} t : eqf (@maybe_checkpoint r s A default use_checkpoint t) t
      := fun idxs => maybe_checkpoint_correct.

    #[export] Instance maybe_checkpoint_Proper_dep {r s} : Dependent.Proper (Dependent.idR ==> Dependent.const (fun _ _ => True) ==> eqfR ==> eqfR) (@maybe_checkpoint r s).
    Proof. repeat intro; rewrite !maybe_checkpoint_correct_eqf; auto. Qed.
  End PArray.
  Export (hints) PArray.

  Module List.
    Import Tensor.List.
    #[export] Instance concretize_Proper {r s A} : Proper (eqf ==> eq) (@concretize r s A).
    Proof.
      cbv [eqf Proper respectful]; revert A s; induction r; cbn [concretize]; intros A s t1 t2 H; auto; [].
      destruct s.
      eapply IHr; repeat intro; subst.
      apply map_ext; intro.
      apply H.
    Qed.

    #[export] Instance reabstract_Proper {r s A default} : Proper (pointwise_relation _ eqf ==> eq ==> eqf) (@reabstract r s A default).
    Proof. cbv [reabstract pointwise_relation eqf eqf_rank]; repeat intro; subst; match goal with |- context[match ?x with _ => _ end] => destruct x end; eauto. Qed.

    #[export] Instance checkpoint_Proper {r s A default} : Proper (eqf ==> eqf) (@checkpoint r s A default).
    Proof. cbv [checkpoint]; repeat intro; subst; apply reabstract_Proper; try apply concretize_Proper; repeat intro; auto. Qed.

    Definition checkpoint_correct_eqf {r s A default t} : eqf (@checkpoint r s A default t) t
      := fun idxs => checkpoint_correct.

    #[export] Instance checkpoint_Proper_dep {r s} : Dependent.Proper (Dependent.idR ==> eqfR ==> eqfR) (@checkpoint r s).
    Proof. repeat intro; rewrite !checkpoint_correct_eqf; auto. Qed.

    #[export] Instance maybe_checkpoint_Proper {r s A default use_checkpoint} : Proper (eqf ==> eqf) (@maybe_checkpoint r s A default use_checkpoint).
    Proof. cbv [maybe_checkpoint]; break_innermost_match; try exact _; repeat first [ assumption | intro ]. Qed.

    Definition maybe_checkpoint_correct_eqf {r s A default use_checkpoint} t : eqf (@maybe_checkpoint r s A default use_checkpoint t) t
      := fun idxs => maybe_checkpoint_correct.

    #[export] Instance maybe_checkpoint_Proper_dep {r s} : Dependent.Proper (Dependent.idR ==> Dependent.const (fun _ _ => True) ==> eqfR ==> eqfR) (@maybe_checkpoint r s).
    Proof. repeat intro; rewrite !maybe_checkpoint_correct_eqf; auto. Qed.
  End List.
  Export (hints) List.

  #[export] Instance raw_get_Proper_dep {r s} : Dependent.Proper (eqfR ==> Dependent.const eq ==> Dependent.idR) (@raw_get r s).
  Proof. cbv -[tensor RawIndex]; intros; subst; eauto. Qed.
  #[export] Instance raw_get_Proper {r s A} : Proper (eqf ==> eq ==> eq) (@raw_get r s A).
  Proof. apply raw_get_Proper_dep. Qed.
  #[export] Instance get_Proper_dep {r s} : Dependent.Proper (eqfR ==> Dependent.const eq ==> Dependent.idR) (@get r s).
  Proof. cbv -[tensor RawIndex adjust_indices_for]; intros; subst; eauto. Qed.
  #[export] Instance get_Proper {r s A} : Proper (eqf ==> eq ==> eq) (@get r s A).
  Proof. apply get_Proper_dep. Qed.
  #[export] Instance item_Proper_dep : Dependent.Proper (eqfR ==> Dependent.idR) (@item).
  Proof. cbv [item]; repeat intro; apply raw_get_Proper_dep; eauto. Qed.
  #[export] Instance item_Proper {A} : Proper (eqf ==> eq) (@item A) := _.
  (*
Definition curried_raw_get {r A} {s : Shape r} (t : tensor A s) : @RawIndex.curriedT r A
  := RawIndex.curry (fun idxs => raw_get t idxs).
Definition curried_get {r A} {s : Shape r} (t : tensor A s) : @Index.curriedT r A
  := Index.curry (fun idxs => get t idxs).
   *)

  Local Ltac t_step :=
    first [ progress subst
          | intro
          | reflexivity
          | match goal with
            | [ H : context[_ = _] |- _ ] => rewrite H
            end
          | solve [ eauto ]
          | match goal with
            | [ |- context[match ?x with _ => _ end] ] => destruct x eqn:?
            end ].
  Local Ltac t := repeat t_step.

  #[export] Instance map_Proper_dep {r s} : Dependent.Proper2 ((Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 Dependent.idR) ==> Dependent.lift2_1 eqfR ==> Dependent.lift2_2 eqfR) (@map r s).
  Proof. cbv -[tensor RawIndex]; t. Qed.
  #[export] Instance map2_Proper_dep {r sA sB} : Dependent.Proper3 ((Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_2 Dependent.idR ==> Dependent.lift3_3 Dependent.idR) ==> Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) (@map2 r sA sB).
  Proof. cbv -[tensor RawIndex]; t. Qed.
  #[export] Instance map3_Proper_dep {r sA sB sC} : Dependent.Proper4 ((Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_2 Dependent.idR ==> Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_4 Dependent.idR) ==> Dependent.lift4_1 eqfR ==> Dependent.lift4_2 eqfR ==> Dependent.lift4_3 eqfR ==> Dependent.lift4_4 eqfR) (@map3 r sA sB sC).
  Proof. cbv -[tensor RawIndex]; t. Qed.

  #[export] Instance map_Proper {r s A B RB} : Proper (pointwise_relation _ RB ==> eqf ==> eqfR RB) (@map r s A B).
  Proof. repeat intro; eapply map_Proper_dep; try eassumption; repeat intro; hnf in *; subst; eauto. Qed.
  #[export] Instance map2_Proper {r sA sB A B C R} : Proper (pointwise_relation _ (pointwise_relation _ R) ==> eqf ==> eqf ==> eqfR R) (@map2 r sA sB A B C).
  Proof. repeat intro; eapply map2_Proper_dep; try eassumption; repeat intro; hnf in *; subst; cbv [pointwise_relation] in *; eauto. Qed.
  #[export] Instance map3_Proper {r sA sB sC A B C D R} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ R)) ==> eqf ==> eqf ==> eqf ==> eqfR R) (@map3 r sA sB sC A B C D).
  Proof. repeat intro; eapply map3_Proper_dep; try eassumption; repeat intro; hnf in *; subst; cbv [pointwise_relation] in *; eauto. Qed.
  (*
Definition map_dep {r A B} {s : Shape r} (f : forall a : A, B a) (t : tensor A s) : tensor_dep B t
  := fun i => f (t i).
   *)
  #[export] Instance where__Proper_dep {r sA sB sC} : Dependent.Proper (Dependent.const eqf ==> eqfR ==> eqfR ==> eqfR) (@where_ r sA sB sC).
  Proof. intros ???; cbv [where_ Bool.where_]; apply map3_Proper_dep; repeat intro; hnf in *; subst; break_innermost_match; assumption. Qed.

  #[export] Instance where__Proper {r sA sB sC A} : Proper (eqf ==> eqf ==> eqf ==> eqf) (@where_ r sA sB sC A).
  Proof. apply where__Proper_dep. Qed.

  #[export] Instance tensor_add_Proper_dep {r sA sB} : Dependent.Proper3 ((Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_2 Dependent.idR ==> Dependent.lift3_3 Dependent.idR) ==> Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) (@tensor_add r sA sB).
  Proof. cbv [tensor_add add]; apply map2_Proper_dep. Qed.
  #[export] Instance tensor_add_Proper {r sA sB A B C addA RA RB RC} {_ : Proper (RA ==> RB ==> RC) addA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_add r sA sB A B C addA).
  Proof. apply tensor_add_Proper_dep; assumption. Qed.
  #[export] Instance add_Proper {r sA sB A B C addA RA RB RC} {_ : Proper (RA ==> RB ==> RC) addA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.add _ _ _ (@tensor_add r sA sB A B C addA))
    := _.
  #[export] Instance tensor_sub_Proper_dep {r sA sB} : Dependent.Proper3 ((Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_2 Dependent.idR ==> Dependent.lift3_3 Dependent.idR) ==> Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) (@tensor_sub r sA sB).
  Proof. cbv [tensor_sub sub]; apply map2_Proper_dep. Qed.
  #[export] Instance tensor_sub_Proper {r sA sB A B C subA RA RB RC} {_ : Proper (RA ==> RB ==> RC) subA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_sub r sA sB A B C subA).
  Proof. apply tensor_sub_Proper_dep; assumption. Qed.
  #[export] Instance sub_Proper {r sA sB A B C subA RA RB RC} {_ : Proper (RA ==> RB ==> RC) subA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.sub _ _ _ (@tensor_sub r sA sB A B C subA))
    := _.
  #[export] Instance tensor_mul_Proper_dep {r sA sB} : Dependent.Proper3 ((Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_2 Dependent.idR ==> Dependent.lift3_3 Dependent.idR) ==> Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) (@tensor_mul r sA sB).
  Proof. cbv [tensor_mul mul]; apply map2_Proper_dep. Qed.
  #[export] Instance tensor_mul_Proper {r sA sB A B C mulA RA RB RC} {_ : Proper (RA ==> RB ==> RC) mulA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_mul r sA sB A B C mulA).
  Proof. apply tensor_mul_Proper_dep; assumption. Qed.
  #[export] Instance mul_Proper {r sA sB A B C mulA RA RB RC} {_ : Proper (RA ==> RB ==> RC) mulA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.mul _ _ _ (@tensor_mul r sA sB A B C mulA))
    := _.
  #[export] Instance tensor_div_by_Proper_dep {r sA sB} : Dependent.Proper3 ((Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_2 Dependent.idR ==> Dependent.lift3_3 Dependent.idR) ==> Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) (@tensor_div_by r sA sB).
  Proof. cbv [tensor_div_by div]; apply map2_Proper_dep. Qed.
  #[export] Instance tensor_div_by_Proper {r sA sB A B C div_byA RA RB RC} {_ : Proper (RA ==> RB ==> RC) div_byA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_div_by r sA sB A B C div_byA).
  Proof. apply tensor_div_by_Proper_dep; assumption. Qed.
  #[export] Instance div_Proper {r sA sB A B C div_byA RA RB RC} {_ : Proper (RA ==> RB ==> RC) div_byA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.div _ _ _ (@tensor_div_by r sA sB A B C div_byA))
    := _.

  #[export] Instance tensor_sqrt_Proper_dep {r s} : Dependent.Proper ((Dependent.idR ==> Dependent.idR) ==> eqfR ==> eqfR) (@tensor_sqrt r s).
  Proof. intros ???; cbv [tensor_sqrt sqrt]; apply map_Proper_dep. Qed.
  #[export] Instance tensor_sqrt_Proper {r s A sqrtA R} {_ : Proper (R ==> R) sqrtA} : Proper (eqfR R ==> eqfR R) (@tensor_sqrt r s A sqrtA).
  Proof. apply tensor_sqrt_Proper_dep; assumption. Qed.
  #[export] Instance sqrt_Proper {r s A sqrtA R} {_ : Proper (R ==> R) sqrtA} : Proper (eqfR R ==> eqfR R) (@Classes.sqrt _ (@tensor_sqrt r s A sqrtA))
    := _.
  #[export] Instance tensor_opp_Proper_dep {r s} : Dependent.Proper ((Dependent.idR ==> Dependent.idR) ==> eqfR ==> eqfR) (@tensor_opp r s).
  Proof. intros ???; cbv [tensor_opp opp]; apply map_Proper_dep. Qed.
  #[export] Instance tensor_opp_Proper {r s A oppA R} {_ : Proper (R ==> R) oppA} : Proper (eqfR R ==> eqfR R) (@tensor_opp r s A oppA).
  Proof. apply tensor_opp_Proper_dep; assumption. Qed.
  #[export] Instance opp_Proper {r s A oppA R} {_ : Proper (R ==> R) oppA} : Proper (eqfR R ==> eqfR R) (@Classes.opp _ (@tensor_opp r s A oppA))
    := _.

  #[export] Instance reshape_app_split'_Proper_rank_dep {r1 r2} : Dependent.Proper (Dependent.const (fun _ _ => True) ==> Dependent.const (fun _ _ => True) ==> eqfR_rank ==> fun _ _ R => eqfR_rank (eqfR_rank R)) (fun A s1 s2 => @reshape_app_split' r1 r2 s1 s2 A).
  Proof.
    cbv [reshape_app_split' RawIndex.curry_radd].
    repeat intro; eauto.
  Qed.
  #[export] Instance reshape_app_split'_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (fun s1 s2 => @reshape_app_split' r1 r2 s1 s2 A).
  Proof. apply reshape_app_split'_Proper_rank_dep. Qed.
  #[export] Instance reshape_app_combine'_Proper_rank_dep {r1 r2} : Dependent.Proper (Dependent.const (fun _ _ => True) ==> Dependent.const (fun _ _ => True) ==> (fun _ _ R => eqfR_rank (eqfR_rank R)) ==> eqfR_rank) (fun A s1 s2 => @reshape_app_combine' r1 r2 s1 s2 A).
    cbv [reshape_app_combine' RawIndex.uncurry_radd].
    repeat intro; destruct RawIndex.split_radd; cbv [eqfR eqfR_rank pointwise_relation] in *; eauto.
  Qed.
  #[export] Instance reshape_app_combine'_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (fun s1 s2 => @reshape_app_combine' r1 r2 s1 s2 A).
  Proof. apply reshape_app_combine'_Proper_rank_dep. Qed.
  #[export] Instance reshape_app_split_Proper_rank_dep {r1 r2} : Dependent.Proper (Dependent.const (fun _ _ => True) ==> Dependent.const (fun _ _ => True) ==> eqfR_rank ==> (fun _ _ R => eqfR_rank (eqfR_rank R))) (fun A s1 s2 => @reshape_app_split r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_app_split_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (fun s1 s2 => @reshape_app_split r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_app_combine_Proper_rank_dep {r1 r2} : Dependent.Proper (Dependent.const (fun _ _ => True) ==> Dependent.const (fun _ _ => True) ==> (fun _ _ R => eqfR_rank (eqfR_rank R)) ==> eqfR_rank) (fun A s1 s2 => @reshape_app_combine r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_app_combine_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (fun s1 s2 => @reshape_app_combine r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_snoc_split_Proper_rank_dep {r} : Dependent.Proper (Dependent.const (fun _ _ => True) ==> Dependent.const (fun _ _ => True) ==> eqfR_rank ==> (fun _ _ R => eqfR_rank (eqfR_rank R))) (fun A s1 s2 => @reshape_snoc_split r s1 s2 A).
  Proof.
    cbv [reshape_snoc_split RawIndex.curry_radd].
    repeat intro; eauto.
  Qed.
  #[export] Instance reshape_snoc_split_Proper_rank {r A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (fun s1 s2 => @reshape_snoc_split r s1 s2 A).
  Proof. apply reshape_snoc_split_Proper_rank_dep. Qed.
  #[export] Instance reshape_snoc_combine_Proper_rank_dep {r} : Dependent.Proper (Dependent.const (fun _ _ => True) ==> Dependent.const (fun _ _ => True) ==> (fun _ _ R => eqfR_rank (eqfR_rank R)) ==> eqfR_rank) (fun A s1 s2 => @reshape_snoc_combine r s1 s2 A).
  Proof.
    cbv [reshape_snoc_combine RawIndex.uncurry_radd].
    repeat intro; destruct RawIndex.split_radd; cbv [eqfR eqfR_rank pointwise_relation] in *; eauto.
  Qed.
  #[export] Instance reshape_snoc_combine_Proper_rank {r A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (fun s1 s2 => @reshape_snoc_combine r s1 s2 A).
  Proof. apply reshape_snoc_combine_Proper_rank_dep. Qed.

  #[export] Instance reshape_app_split'_Proper_dep {r1 r2 s1 s2} : Dependent.Proper (eqfR ==> (fun _ _ R => eqfR_rank (eqfR_rank R))) (@reshape_app_split' r1 r2 s1 s2).
  Proof. repeat intro; eapply reshape_app_split'_Proper_rank_dep; trivial. Qed.
  #[export] Instance reshape_app_split'_Proper {r1 r2 s1 s2 A R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_app_split' r1 r2 s1 s2 A).
  Proof. repeat intro; eapply reshape_app_split'_Proper_rank; trivial. Qed.
  #[export] Instance reshape_app_combine'_Proper_dep {r1 r2 s1 s2} : Dependent.Proper ((fun _ _ R => eqfR_rank (eqfR_rank R)) ==> eqfR) (@reshape_app_combine' r1 r2 s1 s2).
  Proof. repeat intro; eapply reshape_app_combine'_Proper_rank_dep; trivial. Qed.
  #[export] Instance reshape_app_combine'_Proper {r1 r2 s1 s2 A R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_app_combine' r1 r2 s1 s2 A).
  Proof. repeat intro; eapply reshape_app_combine'_Proper_rank; trivial. Qed.
  #[export] Instance reshape_app_split_Proper_dep {r1 r2 s1 s2} : Dependent.Proper (eqfR ==> (fun _ _ R => eqfR_rank (eqfR_rank R))) (@reshape_app_split r1 r2 s1 s2) := _.
  #[export] Instance reshape_app_split_Proper {r1 r2 s1 s2 A R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_app_split r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_app_combine_Proper_dep {r1 r2 s1 s2} : Dependent.Proper ((fun _ _ R => eqfR_rank (eqfR_rank R)) ==> eqfR) (@reshape_app_combine r1 r2 s1 s2) := _.
  #[export] Instance reshape_app_combine_Proper {r1 r2 s1 s2 A R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_app_combine r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_snoc_split_Proper_dep {r s1 s2} : Dependent.Proper (eqfR ==> (fun _ _ R => eqfR_rank (eqfR_rank R))) (@reshape_snoc_split r s1 s2).
  Proof. repeat intro; eapply reshape_snoc_split_Proper_rank_dep; trivial. Qed.
  #[export] Instance reshape_snoc_split_Proper {r s1 s2 A R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_snoc_split r s1 s2 A).
  Proof. repeat intro; eapply reshape_snoc_split_Proper_rank; trivial. Qed.
  #[export] Instance reshape_snoc_combine_Proper_dep {r s1 s2} : Dependent.Proper ((fun _ _ R => eqfR_rank (eqfR_rank R)) ==> eqfR) (@reshape_snoc_combine r s1 s2).
  Proof. repeat intro; eapply reshape_snoc_combine_Proper_rank_dep; trivial. Qed.
  #[export] Instance reshape_snoc_combine_Proper {r s1 s2 A R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_snoc_combine r s1 s2 A).
  Proof. repeat intro; eapply reshape_snoc_combine_Proper_rank; trivial. Qed.
  (*
Definition uncurry {r A} {s : Shape r} : @RawIndex.curriedT r A -> tensor A s
  := RawIndex.uncurry.
Definition curry {r A} {s : Shape r} : tensor A s -> @RawIndex.curriedT r A
  := RawIndex.curry.
   *)

  #[export] Instance map'_Proper_dep {ra1 ra2 rb sa1 sa2 sb} : Dependent.Proper2 ((Dependent.lift2_1 eqfR ==> Dependent.lift2_2 eqfR) ==> Dependent.lift2_1 eqfR ==> Dependent.lift2_2 eqfR) (@map' ra1 ra2 rb sa1 sa2 sb).
  Proof.
    cbv [map']; repeat intro.
    apply reshape_app_combine_Proper_dep.
    eapply map_Proper_dep; try eassumption.
    apply reshape_app_split_Proper_dep; eassumption.
  Qed.
  #[export] Instance map'_Proper {ra1 ra2 rb sa1 sa2 sb A B RA RB} : Proper ((eqfR RA ==> eqfR RB) ==> eqfR RA ==> eqfR RB) (@map' ra1 ra2 rb sa1 sa2 sb A B).
  Proof. apply map'_Proper_dep. Qed.
  #[export] Instance map2'_Proper_dep {ri1 ri2 ro sA1 sB1 sA2 sB2 so} : Dependent.Proper3 ((Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) ==> Dependent.lift3_1 eqfR ==> Dependent.lift3_2 eqfR ==> Dependent.lift3_3 eqfR) (@map2' ri1 ri2 ro sA1 sB1 sA2 sB2 so).
  Proof.
    cbv [map2']; repeat intro.
    apply reshape_app_combine_Proper_dep.
    eapply map2_Proper_dep; try eassumption.
    all: apply reshape_app_split_Proper_dep; eassumption.
  Qed.
  #[export] Instance map2'_Proper {ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C RA RB RC} : Proper ((eqfR RA ==> eqfR RB ==> eqfR RC) ==> eqfR RA ==> eqfR RB ==> eqfR RC) (@map2' ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C).
  Proof. apply map2'_Proper_dep. Qed.

  #[export] Instance map'_Proper_2 {ra1 ra2 rb sa1 sa2 sb A B f RA RB}
    {Hf : Proper (eqfR RA ==> eqfR RB) f}
    : Proper (eqfR RA ==> eqfR RB) (@map' ra1 ra2 rb sa1 sa2 sb A B f)
    := _.
  #[export] Instance map2'_Proper_2 {ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C f RA RB RC}
    {_ : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) f}
    : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@map2' ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C f)
    := _.

  #[export] Instance broadcast'_Proper_dep {r} : Dependent.Proper (Dependent.idR ==> eqfR) (@broadcast' r).
  Proof.
    cbv [broadcast' repeat']; repeat intro; assumption.
  Qed.
  #[export] Instance broadcast'_Proper {r A R} : Proper (R ==> eqfR R) (@broadcast' r A).
  Proof. apply broadcast'_Proper_dep. Qed.
  #[export] Instance broadcast_Proper_dep {r r' s} : Dependent.Proper (eqfR ==> eqfR) (@broadcast r r' s).
  Proof.
    cbv [broadcast]; repeat intro.
    apply reshape_app_combine_Proper_dep.
    eapply broadcast'_Proper_dep; assumption.
  Qed.
  #[export] Instance broadcast_Proper {r r' s A R} : Proper (eqfR R ==> eqfR R) (@broadcast r r' s A).
  Proof. apply broadcast_Proper_dep. Qed.
  #[export] Instance repeat_Proper_dep {r r' s s'} : Dependent.Proper (eqfR ==> eqfR) (@repeat r r' s s').
  Proof.
    cbv [repeat repeat']; repeat intro.
    apply reshape_app_combine_Proper_dep.
    intro; assumption.
  Qed.
  #[export] Instance repeat_Proper {r r' s s' A R} : Proper (eqfR R ==> eqfR R) (@repeat r r' s s' A).
  Proof. apply repeat_Proper_dep. Qed.

  #[export] Instance keepdim_gen_Proper_dep {r s} : Dependent.Proper2 ((Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 eqfR) ==> Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 eqfR) (@keepdim_gen r s).
  Proof.
    cbv [keepdim_gen]; repeat intro.
    apply broadcast_Proper_dep; cbv in *; eauto.
  Qed.
  #[export] Instance keepdim_gen_Proper {r s A B R} : Proper (pointwise_relation _ (eqfR R) ==> eq ==> eqfR R) (@keepdim_gen r s A B).
  Proof. intros ???; apply keepdim_gen_Proper_dep; repeat intro; hnf in *; subst; cbv in *; eauto. Qed.
  #[export] Instance keepdim_Proper_dep : Dependent.Proper2 ((Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 Dependent.idR) ==> Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 eqfR) (@keepdim).
  Proof.
    cbv [keepdim]; intros ??? ??? ?? H ??? ?; subst; eapply keepdim_gen_Proper_dep; t.
    cbv in *; eauto.
  Qed.
  #[export] Instance keepdim_Proper {A B R} : Proper (pointwise_relation _ R ==> eq ==> eqfR R) (@keepdim A B).
  Proof. intros ???; apply keepdim_Proper_dep; cbv; intros; subst; eauto. Qed.

  #[export] Instance reduce_axis_m1'_Proper_dep {r s1 s2} : Dependent.Proper2 ((Dependent.const2 eq ==> Dependent.const2 eq ==> Dependent.const2 eq ==> (Dependent.const2 eq ==> Dependent.lift2_1 Dependent.idR) ==> Dependent.lift2_2 Dependent.idR) ==> Dependent.lift2_1 eqfR ==> Dependent.lift2_2 eqfR) (@reduce_axis_m1' r s1 s2).
  Proof.
    cbv [reduce_axis_m1']; repeat intro.
    eapply map_Proper_dep; try eapply reshape_snoc_split_Proper_dep; try eassumption.
    repeat intro.
    match goal with H : _ |- _ => apply H end; repeat intro; hnf in *; subst; eauto.
  Qed.
  #[export] Instance reduce_axis_m1'_Proper {r s1 s2 A B RA RB} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB) (@reduce_axis_m1' r s1 s2 A B).
  Proof. intros ???; apply reduce_axis_m1'_Proper_dep; cbv in *; intros; subst; eauto. Qed.

  #[export] Instance reduce_axis_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper2 ((Dependent.const2 eq ==> Dependent.const2 eq ==> Dependent.const2 eq ==> (Dependent.const2 eq ==> Dependent.lift2_1 Dependent.idR) ==> Dependent.lift2_2 Dependent.idR) ==> Dependent.lift2_1 eqfR ==> Dependent.lift2_2 eqfR)
        (@reduce_axis_m1 r s1 s2 keepdim).
  Proof.
    cbv [reduce_axis_m1]; destruct keepdim; repeat intro.
    all: eapply reduce_axis_m1'_Proper_dep; try eassumption.
  Qed.
  #[export] Instance reduce_axis_m1_Proper {r s1 s2 keepdim A B RA RB}
    : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB)
        (@reduce_axis_m1 r s1 s2 A B keepdim).
  Proof. intros ???; apply reduce_axis_m1_Proper_dep; cbv in *; intros; subst; eauto. Qed.

  #[export] Instance reduce_axis_m1'_Proper_2 {r s1 s2 A B reduction RA RB}
   {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1' r s1 s2 A B reduction)
    := _.

  #[export] Instance reduce_axis_m1_Proper_2 {r s1 s2 keepdim A B reduction RA RB}
    {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1 r s1 s2 keepdim A B reduction)
    := _.

  #[export] Instance reduce_axis_m1_Proper_2_keepdim_false {r s1 s2 A B reduction RA RB}
    {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1 r s1 s2 false A B reduction)
    := reduce_axis_m1_Proper_2 (keepdim:=false).
  #[export] Instance reduce_axis_m1_Proper_2_keepdim_true {r s1 s2 A B reduction RA RB}
    {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1 r s1 s2 true A B reduction)
    := reduce_axis_m1_Proper_2 (keepdim:=true).

  (*#[export] Instance reduce_axis_m1'_Proper' {r s1 s2 A B reduction RA} : Proper (eqfR RA ==> eqf) (@reduce_axis_m1' r s1 s2 A B reduction).
Proof.
  cbv [reduce_axis_m1'].
  intros ?? Ht.
  eapply map_Proper_R; try eapply reshape_snoc_split_Proper; try eassumption.
  intros ?? Ht'.
  cbv [pointwise_relation eqfR respectful] in *.
  eauto.
Qed.

#[export] Instance reduce_axis_m1_Proper {r s1 s2 A B keepdim RA RB}
  : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB)
      (@reduce_axis_m1 r s1 s2 A B keepdim).
Proof.
  cbv [reduce_axis_m1]; destruct keepdim; intros ?? H ?? Ht ?.
  all: eapply reduce_axis_m1'_Proper; try eassumption.
Qed.

   *)

  #[export] Instance unsqueeze_dim_m1_Proper_dep {r s} : Dependent.Proper (eqfR ==> eqfR) (@unsqueeze_dim_m1 r s).
  Proof. cbv; eauto. Qed.

  #[export] Instance unsqueeze_dim_m1_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@unsqueeze_dim_m1 r s A).
  Proof. apply unsqueeze_dim_m1_Proper_dep. Qed.

  #[export] Instance gather_dim_m1_Proper_dep {r ssinput ssindex sinput' sindex'} : Dependent.Proper (eqfR ==> Dependent.const eqf ==> eqfR) (@gather_dim_m1 r ssinput ssindex sinput' sindex').
  Proof.
    intros ??? ?? H1 ?? H2; cbv [gather_dim_m1]; intro.
    hnf in *.
    rewrite H2.
    apply H1.
  Qed.
  #[export] Instance gather_dim_m1_Proper {r ssinput ssindex sinput' sindex' A R} : Proper (eqfR R ==> eqf ==> eqfR R) (@gather_dim_m1 r ssinput ssindex sinput' sindex' A).
  Proof. apply gather_dim_m1_Proper_dep. Qed.

  #[export] Instance squeeze_Proper_dep {r s} : Dependent.Proper (eqfR ==> eqfR) (@squeeze r s).
  Proof. cbv; eauto. Qed.
  #[export] Instance squeeze_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@squeeze r s A).
  Proof. apply squeeze_Proper_dep. Qed.
  #[export] Instance reshape_all_Proper_dep {r s} : Dependent.Proper (eqfR ==> eqfR) (@reshape_all r s).
  Proof. intros ??? ?? H ?; cbv [reshape_all]; apply H. Qed.
  #[export] Instance reshape_all_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@reshape_all r s A).
  Proof. apply reshape_all_Proper_dep. Qed.
  #[export] Instance unreshape_all_Proper_dep {r s} : Dependent.Proper (eqfR ==> eqfR) (@unreshape_all r s).
  Proof. intros ??? ?? H ?; cbv [unreshape_all]; apply H. Qed.
  #[export] Instance unreshape_all_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@unreshape_all r s A).
  Proof. apply unreshape_all_Proper_dep. Qed.

  #[export] Instance sum_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@sum_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [sum_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.sum_Proper_dep; eauto.
  Qed.

  #[export] Instance sum_dim_m1_Proper {r s1 s2 keepdim A zeroA addA}
    : Proper (eqf ==> eqf) (@sum_dim_m1 r s1 s2 keepdim A zeroA addA).
  Proof. apply sum_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance prod_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@prod_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [prod_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.prod_Proper_dep; eauto.
  Qed.

  #[export] Instance prod_dim_m1_Proper {r s1 s2 keepdim A oneA mulA}
    : Proper (eqf ==> eqf) (@prod_dim_m1 r s1 s2 keepdim A oneA mulA).
  Proof. apply prod_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance max_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@max_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [max_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.max_Proper_dep; eauto.
  Qed.

  #[export] Instance max_dim_m1_Proper {r s1 s2 keepdim A maxA}
    : Proper (eqf ==> eqf) (@max_dim_m1 r s1 s2 keepdim A maxA).
  Proof. apply max_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance min_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@min_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [min_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.min_Proper_dep; eauto.
  Qed.

  #[export] Instance min_dim_m1_Proper {r s1 s2 keepdim A minA}
    : Proper (eqf ==> eqf) (@min_dim_m1 r s1 s2 keepdim A minA).
  Proof. apply min_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance argmax_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.const eq)
           ==> eqfR
           ==> Dependent.const eqf)
        (@argmax_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [argmax_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.argmax_Proper_dep; eauto.
  Qed.

  #[export] Instance argmax_dim_m1_Proper {r s1 s2 keepdim A ltbA}
    : Proper (eqf ==> eqf) (@argmax_dim_m1 r s1 s2 keepdim A ltbA).
  Proof. apply argmax_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance argmin_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.const eq)
           ==> eqfR
           ==> Dependent.const eqf)
        (@argmin_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [argmin_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.argmin_Proper_dep; eauto.
  Qed.

  #[export] Instance argmin_dim_m1_Proper {r s1 s2 keepdim A lebA}
    : Proper (eqf ==> eqf) (@argmin_dim_m1 r s1 s2 keepdim A lebA).
  Proof. apply argmin_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance mean_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@mean_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [mean_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.mean_Proper_dep; eauto.
  Qed.

  #[export] Instance mean_dim_m1_Proper {r s1 s2 keepdim A zeroA coerZA addA divA}
    : Proper (eqf ==> eqf) (@mean_dim_m1 r s1 s2 keepdim A zeroA coerZA addA divA).
  Proof. apply mean_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance var_dim_m1_Proper_dep {r s1 s2 keepdim}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const eq
           ==> eqfR
           ==> eqfR)
        (@var_dim_m1 r s1 s2 keepdim).
  Proof.
    repeat intro; cbv [var_dim_m1].
    eapply @reduce_axis_m1_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.var_Proper_dep; eauto.
  Qed.

  #[export] Instance var_dim_m1_Proper {r s1 s2 keepdim A zeroA coerZA addA subA mulA divA correction}
    : Proper (eqf ==> eqf) (@var_dim_m1 r s1 s2 keepdim A zeroA coerZA addA subA mulA divA correction).
  Proof. apply var_dim_m1_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance sum_Proper_dep {r s}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@sum r s).
  Proof.
    repeat intro; cbv [sum].
    eapply @reduce_axis_m1_Proper_dep; try eapply reshape_all_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.sum_Proper_dep; eauto.
  Qed.

  #[export] Instance sum_Proper {r s A zeroA addA}
    : Proper (eqf ==> eqf) (@sum r s A zeroA addA).
  Proof. apply sum_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance prod_Proper_dep {r s}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@prod r s).
  Proof.
    repeat intro; cbv [prod].
    eapply @reduce_axis_m1_Proper_dep; try eapply reshape_all_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.prod_Proper_dep; eauto.
  Qed.

  #[export] Instance prod_Proper {r s A oneA mulA}
    : Proper (eqf ==> eqf) (@prod r s A oneA mulA).
  Proof. apply prod_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance max_Proper_dep {r s}
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@max r s).
  Proof.
    repeat intro; cbv [max].
    eapply @reduce_axis_m1_Proper_dep; try eapply reshape_all_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.max_Proper_dep; eauto.
  Qed.

  #[export] Instance max_Proper {r s A maxA}
    : Proper (eqf ==> eqf) (@max r s A maxA).
  Proof. apply max_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance min_Proper_dep {r s}
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@min r s).
  Proof.
    repeat intro; cbv [min].
    eapply @reduce_axis_m1_Proper_dep; try eapply reshape_all_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.min_Proper_dep; eauto.
  Qed.

  #[export] Instance min_Proper {r s A minA}
    : Proper (eqf ==> eqf) (@min r s A minA).
  Proof. apply min_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance mean_Proper_dep {r s}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> eqfR
           ==> eqfR)
        (@mean r s).
  Proof.
    repeat intro; cbv [mean].
    eapply @reduce_axis_m1_Proper_dep; try eapply reshape_all_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.mean_Proper_dep; eauto.
  Qed.

  #[export] Instance mean_Proper {r s A zeroA coerZA addA divA}
    : Proper (eqf ==> eqf) (@mean r s A zeroA coerZA addA divA).
  Proof. apply mean_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance var_Proper_dep {r s}
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const eq
           ==> eqfR
           ==> eqfR)
        (@var r s).
  Proof.
    repeat intro; cbv [var].
    eapply @reduce_axis_m1_Proper_dep; try eapply reshape_all_Proper_dep; try eassumption.
    repeat intro; eapply Reduction.var_Proper_dep; eauto.
  Qed.

  #[export] Instance var_Proper {r s A zeroA coerZA addA subA mulA divA correction}
    : Proper (eqf ==> eqf) (@var r s A zeroA coerZA addA subA mulA divA correction).
  Proof. apply var_Proper_dep; repeat intro; subst; reflexivity. Qed.

  #[export] Instance softmax_dim_m1_Proper_dep {r s0 s'}
    : Dependent.Proper4
        ((Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_3 Dependent.idR)
           ==> (Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_2 Dependent.idR)
           ==> (Dependent.lift4_2 Dependent.idR ==> Dependent.lift4_3 Dependent.idR)
           ==> Dependent.lift4_3 Dependent.idR
           ==> (Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_4 Dependent.idR)
           ==> (Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_1 Dependent.idR)
           ==> Dependent.const4 (fun _ _ => True)
           ==> Dependent.lift4_1 Dependent.idR
           ==> Dependent.lift4_3 Dependent.idR
           ==> Dependent.lift4_1 eqfR
           ==> Dependent.lift4_4 eqfR)
        (@softmax_dim_m1 r s0 s').
  Proof.
    repeat intro; cbv [softmax_dim_m1 Classes.div Classes.sub].
    eapply tensor_div_by_Proper_dep; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct.
    all: try (eapply sum_dim_m1_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try (eapply map_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try (eapply tensor_sub_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try (eapply max_dim_m1_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: repeat intro; cbv in *; subst; eauto.
  Qed.

  #[export] Instance softmax_dim_m1_Proper {r s0 s' A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultA defaultB}
    : Proper (eqf ==> eqf) (@softmax_dim_m1 r s0 s' A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultA defaultB).
  Proof. eapply softmax_dim_m1_Proper_dep; repeat intro; hnf in *; try instantiate (1:=eq); subst; reflexivity. Qed.

  Lemma softmax_dim_m1_equiv {r s0 s' A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultA defaultB t}
    : eqf
        (@softmax_dim_m1 r s0 s' A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultA defaultB t)
        (@softmax_dim_m1 r s0 s' A A' B C addB subA expA zeroB divB maxA false defaultA defaultB t).
  Proof. eapply softmax_dim_m1_Proper_dep; cbv; repeat intro; try instantiate (1:=eq); subst; reflexivity. Qed.

  #[export] Instance log_softmax_dim_m1_Proper_dep {r s0 s'}
    : Dependent.Proper6
        ((Dependent.lift6_3 Dependent.idR ==> Dependent.lift6_3 Dependent.idR ==> Dependent.lift6_3 Dependent.idR)
           ==> (Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_2 Dependent.idR)
           ==> (Dependent.lift6_2 Dependent.idR ==> Dependent.lift6_3 Dependent.idR)
           ==> Dependent.lift6_3 Dependent.idR
           ==> (Dependent.lift6_4 Dependent.idR ==> Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_5 Dependent.idR)
           ==> (Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_5 Dependent.idR ==> Dependent.lift6_6 Dependent.idR)
           ==> (Dependent.lift6_3 Dependent.idR ==> Dependent.lift6_4 Dependent.idR)
           ==> (Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_1 Dependent.idR)
           ==> Dependent.const6 (fun _ _ => True)
           ==> Dependent.lift6_1 Dependent.idR
           ==> Dependent.lift6_3 Dependent.idR
           ==> Dependent.lift6_5 Dependent.idR
           ==> Dependent.lift6_1 eqfR
           ==> Dependent.lift6_6 eqfR)
        (@log_softmax_dim_m1 r s0 s').
  Proof.
    repeat intro; cbv [log_softmax_dim_m1 sub add].
    all: repeat try (first [ eapply tensor_sub_Proper_dep
                           | eapply tensor_add_Proper_dep
                           | eapply map_Proper_dep
                           | eapply sum_dim_m1_Proper_dep
                           | eapply max_dim_m1_Proper_dep ];
                     try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
  Qed.

  #[export] Instance log_softmax_dim_m1_Proper {r s0 s' A A' B C C' D addB subA expA zeroB addC subA' lnA maxA use_checkpoint defaultA defaultB defaultC'}
    : Proper (eqf ==> eqf) (@log_softmax_dim_m1 r s0 s' A A' B C C' D addB subA expA zeroB addC subA' lnA maxA use_checkpoint defaultA defaultB defaultC').
  Proof. eapply log_softmax_dim_m1_Proper_dep; repeat intro; hnf in *; try instantiate (1:=eq); subst; reflexivity. Qed.

  Lemma log_softmax_dim_m1_equiv {r s0 s' A A' B C C' D addB subA expA zeroB addC subA' lnA maxA use_checkpoint defaultA defaultB defaultC' t}
    : eqf
        (@log_softmax_dim_m1 r s0 s' A A' B C C' D addB subA expA zeroB addC subA' lnA maxA use_checkpoint defaultA defaultB defaultC' t)
        (@log_softmax_dim_m1 r s0 s' A A' B C C' D addB subA expA zeroB addC subA' lnA maxA false defaultA defaultB defaultC' t).
  Proof. eapply log_softmax_dim_m1_Proper_dep; cbv; repeat intro; try instantiate (1:=eq); subst; reflexivity. Qed.

  #[export] Instance softmax_Proper_dep {r s}
    : Dependent.Proper4
        ((Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_3 Dependent.idR)
           ==> (Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_2 Dependent.idR)
           ==> (Dependent.lift4_2 Dependent.idR ==> Dependent.lift4_3 Dependent.idR)
           ==> Dependent.lift4_3 Dependent.idR
           ==> (Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_3 Dependent.idR ==> Dependent.lift4_4 Dependent.idR)
           ==> (Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_1 Dependent.idR ==> Dependent.lift4_1 Dependent.idR)
           ==> Dependent.const4 (fun _ _ => True)
           ==> Dependent.lift4_3 Dependent.idR
           ==> Dependent.lift4_1 eqfR
           ==> Dependent.lift4_4 eqfR)
        (@softmax r s).
  Proof.
    repeat intro; cbv [softmax Classes.div Classes.sub item raw_get].
    eapply tensor_div_by_Proper_dep; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct.
    all: try eapply broadcast'_Proper_dep.
    all: try (eapply sum_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try (eapply map_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try (eapply tensor_sub_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try eapply broadcast'_Proper_dep.
    all: try eapply item_Proper_dep; repeat intro.
    all: try (eapply max_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: repeat intro; cbv in *; subst; eauto.
  Qed.

  #[export] Instance softmax_Proper {r s A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultB}
    : Proper (eqf ==> eqf) (@softmax r s A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultB).
  Proof. eapply softmax_Proper_dep; repeat intro; hnf in *; try instantiate (1:=eq); subst; reflexivity. Qed.

  Lemma softmax_equiv {r s A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultB t}
    : eqf
        (@softmax r s A A' B C addB subA expA zeroB divB maxA use_checkpoint defaultB t)
        (@softmax r s A A' B C addB subA expA zeroB divB maxA false defaultB t).
  Proof. eapply softmax_Proper_dep; cbv; repeat intro; try instantiate (1:=eq); subst; reflexivity. Qed.

  #[export] Instance log_softmax_Proper_dep {r s}
    : Dependent.Proper6
        ((Dependent.lift6_3 Dependent.idR ==> Dependent.lift6_3 Dependent.idR ==> Dependent.lift6_3 Dependent.idR)
           ==> (Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_2 Dependent.idR)
           ==> (Dependent.lift6_2 Dependent.idR ==> Dependent.lift6_3 Dependent.idR)
           ==> (Dependent.lift6_3 Dependent.idR ==> Dependent.lift6_4 Dependent.idR)
           ==> Dependent.lift6_3 Dependent.idR
           ==> (Dependent.lift6_4 Dependent.idR ==> Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_5 Dependent.idR)
           ==> (Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_5 Dependent.idR ==> Dependent.lift6_6 Dependent.idR)
           ==> (Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_1 Dependent.idR ==> Dependent.lift6_1 Dependent.idR)
           ==> Dependent.const6 (fun _ _ => True)
           ==> Dependent.lift6_3 Dependent.idR
           ==> Dependent.lift6_1 eqfR
           ==> Dependent.lift6_6 eqfR)
        (@log_softmax r s).
  Proof.
    repeat intro; cbv [log_softmax sub add item raw_get ln].
    all: repeat try (first [ eapply tensor_sub_Proper_dep
                           | eapply tensor_add_Proper_dep
                           | eapply broadcast'_Proper_dep
                           | match goal with H : _ |- _ => eapply H; clear H end ];
                     try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: try (eapply sum_Proper_dep; try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
    all: repeat try (first [ eapply tensor_sub_Proper_dep
                           | eapply map_Proper_dep
                           | eapply broadcast'_Proper_dep
                           | eapply max_Proper_dep
                           | match goal with H : _ |- _ => eapply H; clear H end ];
                     try eassumption; repeat intro; hnf; rewrite ?PArray.maybe_checkpoint_correct).
  Qed.

  #[export] Instance log_softmax_Proper {r s A A' B C C' D addB subA expA lnA zeroB addC subA' maxA use_checkpoint defaultB}
    : Proper (eqf ==> eqf) (@log_softmax r s A A' B C C' D addB subA expA lnA zeroB addC subA' maxA use_checkpoint defaultB).
  Proof. eapply log_softmax_Proper_dep; repeat intro; hnf in *; try instantiate (1:=eq); subst; reflexivity. Qed.

  Lemma log_softmax_equiv {r s A A' B C C' D addB subA expA lnA zeroB addC subA' maxA use_checkpoint defaultB t}
    : eqf
        (@log_softmax r s A A' B C C' D addB subA expA lnA zeroB addC subA' maxA use_checkpoint defaultB t)
        (@log_softmax r s A A' B C C' D addB subA expA lnA zeroB addC subA' maxA false defaultB t).
  Proof. eapply log_softmax_Proper_dep; cbv; repeat intro; try instantiate (1:=eq); subst; reflexivity. Qed.

  #[export] Instance to_bool_Proper_dep {r s} : Dependent.Proper (Dependent.idR ==> (Dependent.idR ==> Dependent.idR ==> Dependent.const eq) ==> eqfR ==> Dependent.const eqf) (@to_bool r s).
  Proof.
    repeat intro; cbv [to_bool]; eapply map_Proper_dep; repeat intro; hnf in *;
      try match goal with H : _ |- _ => eapply H end; hnf.
    cbv in *; match goal with H : _ |- _ => now erewrite H by eauto end.
  Qed.

  #[export] Instance to_bool_Proper {r s A zero eqb} : Proper (eqf ==> eqf) (@to_bool r s A zero eqb).
  Proof. apply to_bool_Proper_dep; cbv; intros; subst; reflexivity. Qed.

  #[export] Instance of_bool_Proper_dep {r s} : Dependent.Proper (Dependent.idR ==> Dependent.idR ==> Dependent.const eqf ==> eqfR) (@of_bool r s).
  Proof.
    repeat intro; cbv [of_bool]; eapply map_Proper_dep; repeat intro; hnf in *;
      try match goal with H : _ |- _ => eapply H end; hnf.
    match goal with H : _ |- _ => erewrite H by eauto end.
    cbv; break_innermost_match; assumption.
  Qed.

  #[export] Instance of_bool_Proper {A r s zero one} : Proper (eqf ==> eqf) (@of_bool A r s zero one).
  Proof. apply of_bool_Proper_dep; reflexivity. Qed.

  (*(* TODO: nary *)
Definition tupleify {s1 s2 A B} (t1 : tensor A [s1]) (t2 : tensor B [s2]) : tensor (A * B) [s1; s2]
  := fun '((tt, a), b) => (raw_get t1 [a], raw_get t2 [b]).
Definition cartesian_prod {s1 s2 A} (t1 : tensor A [s1]) (t2 : tensor A [s2]) : tensor A [s1 * s2; 2]
  := fun '((tt, idx), tuple_idx)
     => let '(a, b) := raw_get (reshape_all (tupleify t1 t2)) [idx] in
        nth_default a [a; b] (Z.to_nat (Uint63.to_Z (tuple_idx mod 2))).
   *)
  #[export] Instance tril_Proper_dep {rnk s r c} : Dependent.Proper (Dependent.idR ==> Dependent.const eq ==> eqfR ==> eqfR) (@tril rnk s r c).
  Proof. cbv [tril]; repeat intro; subst; break_innermost_match; eauto. Qed.
  #[export] Instance tril_Proper {rnk s r c A zeroA diagonal} : Proper (eqf ==> eqf) (@tril rnk s r c A zeroA diagonal).
  Proof. apply tril_Proper_dep; reflexivity. Qed.
  #[export] Instance triu_Proper_dep {rnk s r c} : Dependent.Proper (Dependent.idR ==> Dependent.const eq ==> eqfR ==> eqfR) (@triu rnk s r c).
  Proof. cbv [triu]; repeat intro; subst; break_innermost_match; eauto. Qed.
  #[export] Instance triu_Proper {rnk s r c A zeroA diagonal} : Proper (eqf ==> eqf) (@triu rnk s r c A zeroA diagonal).
  Proof. apply triu_Proper_dep; reflexivity. Qed.

  (* probably not needed, but might speed things up a bit *)
  #[export] Instance : Params (@Tensor.eqfR) 4 := {}.
  #[export] Instance : Params (@Tensor.map2) 6 := {}.
  #[export] Instance : Params (@Tensor.of_bool) 5 := {}.
End Tensor.
#[export] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
Export (hints,coercions) Tensor.
#[export] Set Warnings Append "uniform-inheritance,ambiguous-paths".
