From Coq.Structures Require Import Equalities.
From Coq Require Import ZArith Sint63 Uint63 List PArray Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Nat.
From NeuralNetInterp.Util Require Import Wf_Uint63 Wf_Uint63.Instances PArray.Proofs PArray.Instances List.Proofs Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool (*PrimitiveProd*).
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Hetero Relation_Definitions.Dependent.
From NeuralNetInterp.Util.Classes Require Morphisms.Dependent.
From NeuralNetInterp.Torch Require Import Tensor.
Import Dependent.ProperNotations.

Module Tensor.
  Definition eqfR_rank {r} : Dependent.relation (@tensor_of_rank r)
    := fun A B R x y => forall i, R (x i) (y i).
  #[global] Arguments eqfR_rank {r} [A B] R x y.
  Notation eqf_rank := (eqfR_rank eq).
  Definition eqfR {r s} : Dependent.relation (@tensor r s)
    := eqfR_rank.
  #[global] Arguments eqfR {r s} [A B] R x y.
  Notation eqf := (eqfR eq).

  #[export] Instance eqf_Reflexive {r s A R} {_ : Reflexive R} : Reflexive (@eqfR r s A A R).
  Proof. repeat intro; subst; reflexivity. Qed.
  #[export] Instance eqf_Symmetric {r s A R} {_ : Symmetric R} : Symmetric (@eqfR r s A A R).
  Proof. cbv; repeat intro; subst; symmetry; auto. Qed.
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

  Locate Proper.
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

  Goal True.
    epose (fun r s => Dependent.Proper2 ((Dependent.id2R1 ==> RB) ==> eqfR RA ==> eqfR RB) (@map r s)
  #[export] Instance map_Proper_dep {r s} : Dependent.Proper2 ((Dependent.id2R1 ==> RB) ==> eqfR RA ==> eqfR RB) (@map r s).
  Proof. cbv -[tensor RawIndex]; t. Qed.
  #[export] Instance map2_Proper_R {r sA sB A B C RA RB RC} : Proper ((RA ==> RB ==> RC) ==> eqfR RA ==> eqfR RB ==> eqfR RC) (@map2 r sA sB A B C).
  Proof. cbv -[tensor RawIndex]; t. Qed.
  #[export] Instance map3_Proper_R {r sA sB sC A B C D RA RB RC RD} : Proper ((RA ==> RB ==> RC ==> RD) ==> eqfR RA ==> eqfR RB ==> eqfR RC ==> eqfR RD) (@map3 r sA sB sC A B C D).
  Proof. cbv -[tensor RawIndex]; t. Qed.

  #[export] Instance map_Proper {r s A B RB} : Proper (pointwise_relation _ RB ==> eqf ==> eqfR RB) (@map r s A B).
  Proof. repeat intro; eapply map_Proper_R; try eassumption; repeat intro; subst; eauto. Qed.
  #[export] Instance map2_Proper {r sA sB A B C R} : Proper (pointwise_relation _ (pointwise_relation _ R) ==> eqf ==> eqf ==> eqfR R) (@map2 r sA sB A B C).
  Proof. repeat intro; eapply map2_Proper_R; try eassumption; repeat intro; subst; cbv [pointwise_relation] in *; eauto. Qed.
  #[export] Instance map3_Proper {r sA sB sC A B C D R} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ R)) ==> eqf ==> eqf ==> eqf ==> eqfR R) (@map3 r sA sB sC A B C D).
  Proof. repeat intro; eapply map3_Proper_R; try eassumption; repeat intro; subst; cbv [pointwise_relation] in *; eauto. Qed.
  (*
Definition map_dep {r A B} {s : Shape r} (f : forall a : A, B a) (t : tensor A s) : tensor_dep B t
  := fun i => f (t i).
   *)
  #[export] Instance where__Proper {r sA sB sC A} : Proper (eqf ==> eqf ==> eqf ==> eqf) (@where_ r sA sB sC A).
  Proof. apply map3_Proper; repeat intro; reflexivity. Qed.

  #[export] Instance tensor_add_Proper {r sA sB A B C addA RA RB RC} {_ : Proper (RA ==> RB ==> RC) addA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_add r sA sB A B C addA).
  Proof. cbv [tensor_add add]; repeat intro; eapply map2_Proper_R; eassumption. Qed.
  #[export] Instance tensor_sub_Proper {r sA sB A B C subA RA RB RC} {_ : Proper (RA ==> RB ==> RC) subA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_sub r sA sB A B C subA).
  Proof. cbv [tensor_sub sub]; repeat intro; eapply map2_Proper_R; eassumption. Qed.
  #[export] Instance tensor_mul_Proper {r sA sB A B C mulA RA RB RC} {_ : Proper (RA ==> RB ==> RC) mulA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_mul r sA sB A B C mulA).
  Proof. cbv [tensor_mul mul]; repeat intro; eapply map2_Proper_R; eassumption. Qed.
  #[export] Instance tensor_div_by_Proper {r sA sB A B C div_byA RA RB RC} {_ : Proper (RA ==> RB ==> RC) div_byA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@tensor_div_by r sA sB A B C div_byA).
  Proof. cbv [tensor_div_by div]; repeat intro; eapply map2_Proper_R; eassumption. Qed.
  #[export] Instance tensor_sqrt_Proper {r s A sqrtA R} {_ : Proper (R ==> R) sqrtA} : Proper (eqfR R ==> eqfR R) (@tensor_sqrt r s A sqrtA).
  Proof. cbv [tensor_sqrt sqrt]; repeat intro; eapply map_Proper_R; eassumption. Qed.
  #[export] Instance tensor_opp_Proper {r s A oppA R} {_ : Proper (R ==> R) oppA} : Proper (eqfR R ==> eqfR R) (@tensor_opp r s A oppA).
  Proof. cbv [tensor_opp opp]; repeat intro; eapply map_Proper_R; eassumption. Qed.

  #[export] Instance add_Proper {r sA sB A B C addA RA RB RC} {_ : Proper (RA ==> RB ==> RC) addA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.add _ _ _ (@tensor_add r sA sB A B C addA))
    := _.
  #[export] Instance sub_Proper {r sA sB A B C subA RA RB RC} {_ : Proper (RA ==> RB ==> RC) subA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.sub _ _ _ (@tensor_sub r sA sB A B C subA))
    := _.
  #[export] Instance mul_Proper {r sA sB A B C mulA RA RB RC} {_ : Proper (RA ==> RB ==> RC) mulA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.mul _ _ _ (@tensor_mul r sA sB A B C mulA))
    := _.
  #[export] Instance div_Proper {r sA sB A B C div_byA RA RB RC} {_ : Proper (RA ==> RB ==> RC) div_byA} : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@Classes.div _ _ _ (@tensor_div_by r sA sB A B C div_byA))
    := _.
  #[export] Instance sqrt_Proper {r s A sqrtA R} {_ : Proper (R ==> R) sqrtA} : Proper (eqfR R ==> eqfR R) (@Classes.sqrt _ (@tensor_sqrt r s A sqrtA))
    := _.
  #[export] Instance opp_Proper {r s A oppA R} {_ : Proper (R ==> R) oppA} : Proper (eqfR R ==> eqfR R) (@Classes.opp _ (@tensor_opp r s A oppA))
    := _.

  #[export] Instance reshape_app_split'_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (fun s1 s2 => @reshape_app_split' r1 r2 s1 s2 A).
  Proof.
    cbv [reshape_app_split' RawIndex.curry_radd].
    repeat intro; eauto.
  Qed.
  #[export] Instance reshape_app_combine'_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (fun s1 s2 => @reshape_app_combine' r1 r2 s1 s2 A).
  Proof.
    cbv [reshape_app_combine' RawIndex.uncurry_radd].
    repeat intro; destruct RawIndex.split_radd; cbv [eqfR eqfR_rank pointwise_relation] in *; eauto.
  Qed.
  #[export] Instance reshape_app_split_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (fun s1 s2 => @reshape_app_split r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_app_combine_Proper_rank {r1 r2 A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (fun s1 s2 => @reshape_app_combine r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_snoc_split_Proper_rank {r A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (fun s1 s2 => @reshape_snoc_split r s1 s2 A).
  Proof.
    cbv [reshape_snoc_split RawIndex.curry_radd].
    repeat intro; eauto.
  Qed.
  #[export] Instance reshape_snoc_combine_Proper_rank {r A R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (fun s1 s2 => @reshape_snoc_combine r s1 s2 A).
  Proof.
    cbv [reshape_snoc_combine RawIndex.uncurry_radd].
    repeat intro; destruct RawIndex.split_radd; cbv [eqfR eqfR_rank pointwise_relation] in *; eauto.
  Qed.

  #[export] Instance reshape_app_split'_Proper {r1 r2 s1 s2 A R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_app_split' r1 r2 s1 s2 A).
  Proof. repeat intro; eapply reshape_app_split'_Proper_rank; trivial. Qed.
  #[export] Instance reshape_app_combine'_Proper {r1 r2 s1 s2 A R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_app_combine' r1 r2 s1 s2 A).
  Proof. repeat intro; eapply reshape_app_combine'_Proper_rank; trivial. Qed.
  #[export] Instance reshape_app_split_Proper {r1 r2 s1 s2 A R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_app_split r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_app_combine_Proper {r1 r2 s1 s2 A R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_app_combine r1 r2 s1 s2 A) := _.
  #[export] Instance reshape_snoc_split_Proper {r s1 s2 A R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_snoc_split r s1 s2 A).
  Proof. repeat intro; eapply reshape_snoc_split_Proper_rank; trivial. Qed.
  #[export] Instance reshape_snoc_combine_Proper {r s1 s2 A R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_snoc_combine r s1 s2 A).
  Proof. repeat intro; eapply reshape_snoc_combine_Proper_rank; trivial. Qed.
  (*
Definition uncurry {r A} {s : Shape r} : @RawIndex.curriedT r A -> tensor A s
  := RawIndex.uncurry.
Definition curry {r A} {s : Shape r} : tensor A s -> @RawIndex.curriedT r A
  := RawIndex.curry.
   *)

  #[export] Instance map'_Proper {ra1 ra2 rb sa1 sa2 sb A B RA RB} : Proper ((eqfR RA ==> eqfR RB) ==> eqfR RA ==> eqfR RB) (@map' ra1 ra2 rb sa1 sa2 sb A B).
  Proof.
    cbv [map']; repeat intro.
    apply reshape_app_combine_Proper.
    eapply map_Proper_R; try eassumption.
    apply reshape_app_split_Proper; eassumption.
  Qed.
  #[export] Instance map2'_Proper {ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C RA RB RC} : Proper ((eqfR RA ==> eqfR RB ==> eqfR RC) ==> eqfR RA ==> eqfR RB ==> eqfR RC) (@map2' ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C).
  Proof.
    cbv [map2']; repeat intro.
    apply reshape_app_combine_Proper.
    eapply map2_Proper_R; try eassumption.
    all: apply reshape_app_split_Proper; eassumption.
  Qed.

  #[export] Instance map'_Proper_2 {ra1 ra2 rb sa1 sa2 sb A B f RA RB}
    {Hf : Proper (eqfR RA ==> eqfR RB) f}
    : Proper (eqfR RA ==> eqfR RB) (@map' ra1 ra2 rb sa1 sa2 sb A B f)
    := _.
  #[export] Instance map2'_Proper_2 {ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C f RA RB RC}
    {_ : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) f}
    : Proper (eqfR RA ==> eqfR RB ==> eqfR RC) (@map2' ri1 ri2 ro sA1 sB1 sA2 sB2 so A B C f)
    := _.

  #[export] Instance broadcast_Proper {r r' s A R} : Proper (eqfR R ==> eqfR R) (@broadcast r r' s A).
  Proof.
    cbv [broadcast broadcast' repeat']; intros ??? ?.
    apply reshape_app_combine_Proper.
    intro; assumption.
  Qed.
  #[export] Instance repeat_Proper {r r' s s' A R} : Proper (eqfR R ==> eqfR R) (@repeat r r' s s' A).
  Proof.
    cbv [repeat repeat']; intros ???.
    apply reshape_app_combine_Proper.
    intro; assumption.
  Qed.

  #[export] Instance keepdim_gen_Proper {r s A B R} : Proper (pointwise_relation _ (eqfR R) ==> eq ==> eqfR R) (@keepdim_gen r s A B).
  Proof.
    cbv [keepdim_gen]; intros ?? H ???; subst.
    apply broadcast_Proper; apply H.
  Qed.
  #[export] Instance keepdim_Proper {A B R} : Proper (pointwise_relation _ R ==> eq ==> eqfR R) (@keepdim A B).
  Proof.
    cbv [keepdim]; intros ?? H ??? ?; subst; eapply keepdim_gen_Proper; t.
  Qed.

  #[export] Instance reduce_axis_m1'_Proper {r s1 s2 A B RA RB} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB) (@reduce_axis_m1' r s1 s2 A B).
  Proof.
    cbv [reduce_axis_m1'].
    intros ?? H ?? Ht.
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

  #[export] Instance reduce_axis_m1'_Proper_2 {r s1 s2 A B reduction RA RB}
   {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1' r s1 s2 A B reduction)
    := _.

  #[export] Instance reduce_axis_m1_Proper_2 {r s1 s2 A B keepdim reduction RA RB}
    {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1 r s1 s2 A B keepdim reduction)
    := _.

  #[export] Instance reduce_axis_m1_Proper_2_keepdim_false {r s1 s2 A B reduction RA RB}
    {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1 r s1 s2 A B false reduction)
    := _.
  #[export] Instance reduce_axis_m1_Proper_2_keepdim_true {r s1 s2 A B reduction RA RB}
    {_ : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB)))) reduction}
    : Proper (eqfR RA ==> eqfR RB) (@reduce_axis_m1 r s1 s2 A B true reduction)
    := _.

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
  #[export] Instance softmax_dim_m1_Proper {r s0 s' A B C addB expA zeroB divB}
    : Proper (eqf ==> eqf) (@softmax_dim_m1 r s0 s' A B C addB expA zeroB divB).
  Proof.
    intros ?? Ht.
    cbv [softmax_dim_m1 div].
    eapply tensor_div_by_Proper.
    all: try eapply (@reduce_axis_m1_Proper r s0 s' B B true).
    all: try (eapply map_Proper; try eassumption; repeat intro).
    all: try eapply Reduction.sum_Proper_pointwise.
    all: try exact eq_refl.
    Unshelve.
    repeat intro; subst; reflexivity.
  Qed.

  #[export] Instance log_softmax_dim_m1_Proper {r s0 s' A B C D addB lnA expA zeroB divB}
    : Proper (eqf ==> eqf) (@log_softmax_dim_m1 r s0 s' A B C D addB lnA expA zeroB divB).
  Proof.
    intros ?? Ht.
    cbv [log_softmax_dim_m1 div].
    eapply tensor_div_by_Proper; try eassumption.
    eapply map_Proper; try eassumption.
    all: try eapply (@reduce_axis_m1_Proper r s0 s' B B true).
    all: try (eapply map_Proper; try eassumption; repeat intro).
    all: try eapply Reduction.sum_Proper_pointwise.
    all: try (repeat intro; exact eq_refl).
    Unshelve.
    repeat intro; subst; reflexivity.
  Qed.

  #[export] Instance unsqueeze_dim_m1_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@unsqueeze_dim_m1 r s A).
  Proof. intros ?? H; cbv; intros; apply H. Qed.

  #[export] Instance gather_dim_m1_Proper {r ssinput ssindex sinput' sindex' A R} : Proper (eqfR R ==> eqf ==> eqfR R) (@gather_dim_m1 r ssinput ssindex sinput' sindex' A).
  Proof.
    intros ?? H1 ?? H2; cbv [gather_dim_m1]; intro.
    rewrite H2.
    apply H1.
  Qed.

  #[export] Instance squeeze_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@squeeze r s A).
  Proof. intros ?? H; cbv; intros; apply H. Qed.
  #[export] Instance reshape_m1_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@reshape_m1 r s A).
  Proof. intros ?? H ?; cbv [reshape_m1]; apply H. Qed.
  #[export] Instance unreshape_m1_Proper {r s A R} : Proper (eqfR R ==> eqfR R) (@unreshape_m1 r s A).
  Proof. intros ?? H ?; cbv [unreshape_m1]; apply H. Qed.

  #[export] Instance to_bool_Proper {A r s zero eqb} : Proper (eqf ==> eqf) (@to_bool A r s zero eqb).
  Proof.
    intros ?? H ?; cbv [to_bool]; apply map_Proper; try assumption; repeat intro; reflexivity.
  Qed.

  #[export] Instance of_bool_Proper {A r s zero one} : Proper (eqf ==> eqf) (@of_bool A r s zero one).
  Proof.
    intros ?? H ?; cbv [of_bool]; apply map_Proper; try assumption; repeat intro; reflexivity.
  Qed.


  #[export] Instance mean_Proper {r s A B C zero addA div_boyABC coerB} : Proper (eqf ==> eqf) (@mean r s A B C zero addA div_boyABC coerB).
  Proof.
    cbv [mean]; intros ?? H ?.
    eapply reduce_axis_m1_Proper.
    1: eapply Reduction.mean_Proper_pointwise.
    apply reshape_m1_Proper.
    assumption.
  Qed.

  (*(* TODO: nary *)
Definition tupleify {s1 s2 A B} (t1 : tensor A [s1]) (t2 : tensor B [s2]) : tensor (A * B) [s1; s2]
  := fun '((tt, a), b) => (raw_get t1 [a], raw_get t2 [b]).
Definition cartesian_prod {s1 s2 A} (t1 : tensor A [s1]) (t2 : tensor A [s2]) : tensor A [s1 * s2; 2]
  := fun '((tt, idx), tuple_idx)
     => let '(a, b) := raw_get (reshape_m1 (tupleify t1 t2)) [idx] in
        nth_default a [a; b] (Z.to_nat (Uint63.to_Z (tuple_idx mod 2))).
   *)
  #[export] Instance tril_Proper {rnk s A zeroA r c diagonal} : Proper (eqf ==> eqf) (@tril rnk s A zeroA r c diagonal).
  Proof.
    cbv [tril]; intros ?? H [[? ?] ?].
    rewrite H.
    reflexivity.
  Qed.
  #[export] Instance triu_Proper {rnk s A zeroA r c diagonal} : Proper (eqf ==> eqf) (@triu rnk s A zeroA r c diagonal).
  Proof.
    cbv [triu]; intros ?? H [[? ?] ?].
    rewrite H.
    reflexivity.
  Qed.

  (* probably not needed, but might speed things up a bit *)
  #[export] Instance : Params (@Tensor.eqfR) 4 := {}.
  #[export] Instance : Params (@Tensor.map2) 6 := {}.
  #[export] Instance : Params (@Tensor.of_bool) 5 := {}.
End Tensor.
Export (hints) Tensor.
