From Coq.Structures Require Import Equalities.
From Coq Require Import ZArith Sint63 Uint63 List PArray Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Nat.
From NeuralNetInterp.Util Require Import Wf_Uint63 Wf_Uint63.Instances PArray.Proofs PArray.Instances List.Proofs Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool (*PrimitiveProd*).
From NeuralNetInterp.Torch Require Import Tensor.

Module Tensor.
  Definition eqfR_rank {A r} R : relation (@tensor_of_rank A r)
    := pointwise_relation _ R.
  Notation eqf_rank := (eqfR_rank eq).
  Definition eqfR {r A s} R : relation (@tensor r A s)
    := eqfR_rank R.
  Notation eqf := (eqfR eq).

  #[export] Instance eqf_Reflexive {r A s R} {_ : Reflexive R} : Reflexive (@eqfR r A s R).
  Proof. repeat intro; subst; reflexivity. Qed.
  #[export] Instance eqf_Symmetric {r A s R} {_ : Symmetric R} : Symmetric (@eqfR r A s R).
  Proof. cbv; repeat intro; subst; symmetry; auto. Qed.
  #[export] Instance eqf_Transitive {r A s R} {_ : Transitive R} : Transitive (@eqfR r A s R).
  Proof. intros x y z H1 H2; repeat intro; subst; etransitivity; [ eapply H1 | eapply H2 ]; reflexivity. Qed.

  Module PArray.
    Import Tensor.PArray.
    #[export] Instance concretize_Proper {r A default s} : Proper (eqf ==> eq) (@concretize r A default s).
    Proof.
      cbv [eqf Proper respectful]; revert A default s; induction r; cbn [concretize]; intros A default s t1 t2 H; auto; [].
      destruct s.
      eapply IHr; repeat intro; subst.
      apply PArray.init_default_Proper; try reflexivity; repeat intro; subst.
      apply H.
    Qed.

    #[export] Instance reabstract_Proper {r A s} : Proper (pointwise_relation _ eqf ==> eq ==> eqf) (@reabstract r A s).
    Proof. cbv [reabstract pointwise_relation eqf eqf_rank]; repeat intro; subst; destruct andb; eauto. Qed.

    #[export] Instance checkpoint_Proper {r A default s} : Proper (eqf ==> eqf) (@checkpoint r A default s).
    Proof. cbv [checkpoint]; repeat intro; subst; apply reabstract_Proper; try apply concretize_Proper; repeat intro; auto. Qed.
  End PArray.
  Export (hints) PArray.

  Module List.
    Import Tensor.List.
    #[export] Instance concretize_Proper {r A s} : Proper (eqf ==> eq) (@concretize r A s).
    Proof.
      cbv [eqf Proper respectful]; revert A s; induction r; cbn [concretize]; intros A s t1 t2 H; auto; [].
      destruct s.
      eapply IHr; repeat intro; subst.
      apply map_ext; intro.
      apply H.
    Qed.

    #[export] Instance reabstract_Proper {r A default s} : Proper (pointwise_relation _ eqf ==> eq ==> eqf) (@reabstract r A default s).
    Proof. cbv [reabstract pointwise_relation eqf eqf_rank]; repeat intro; subst; match goal with |- context[match ?x with _ => _ end] => destruct x end; eauto. Qed.

    #[export] Instance checkpoint_Proper {r A default s} : Proper (eqf ==> eqf) (@checkpoint r A default s).
    Proof. cbv [checkpoint]; repeat intro; subst; apply reabstract_Proper; try apply concretize_Proper; repeat intro; auto. Qed.
  End List.
  Export (hints) List.

  #[export] Instance raw_get_Proper {r A s} : Proper (eqf ==> eq ==> eq) (@raw_get r A s).
  Proof. cbv -[tensor RawIndex]; intros; subst; eauto. Qed.
  #[export] Instance get_Proper {r A s} : Proper (eqf ==> eq ==> eq) (@get r A s).
  Proof. cbv -[tensor RawIndex adjust_indices_for]; intros; subst; eauto. Qed.
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

  #[export] Instance map_Proper_R {r A B s RA RB} : Proper ((RA ==> RB) ==> eqfR RA ==> eqfR RB) (@map r A B s).
  Proof. cbv -[tensor RawIndex]; t. Qed.
  #[export] Instance map2_Proper_R {r A B C sA sB RA RB RC} : Proper ((RA ==> RB ==> RC) ==> eqfR RA ==> eqfR RB ==> eqfR RC) (@map2 r A B C sA sB).
  Proof. cbv -[tensor RawIndex]; t. Qed.
  #[export] Instance map3_Proper_R {r A B C D sA sB sC RA RB RC RD} : Proper ((RA ==> RB ==> RC ==> RD) ==> eqfR RA ==> eqfR RB ==> eqfR RC ==> eqfR RD) (@map3 r A B C D sA sB sC).
  Proof. cbv -[tensor RawIndex]; t. Qed.

  #[export] Instance map_Proper {r A B s RB} : Proper (pointwise_relation _ RB ==> eqf ==> eqfR RB) (@map r A B s).
  Proof. repeat intro; eapply map_Proper_R; try eassumption; repeat intro; subst; eauto. Qed.
  #[export] Instance map2_Proper {r A B C sA sB R} : Proper (pointwise_relation _ (pointwise_relation _ R) ==> eqf ==> eqf ==> eqfR R) (@map2 r A B C sA sB).
  Proof. repeat intro; eapply map2_Proper_R; try eassumption; repeat intro; subst; cbv [pointwise_relation] in *; eauto. Qed.
  #[export] Instance map3_Proper {r A B C D sA sB sC R} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ R)) ==> eqf ==> eqf ==> eqf ==> eqfR R) (@map3 r A B C D sA sB sC).
  Proof. repeat intro; eapply map3_Proper_R; try eassumption; repeat intro; subst; cbv [pointwise_relation] in *; eauto. Qed.
  (*
Definition map_dep {r A B} {s : Shape r} (f : forall a : A, B a) (t : tensor A s) : tensor_dep B t
  := fun i => f (t i).
   *)
  #[export] Instance where__Proper {r A sA sB sC} : Proper (eqf ==> eqf ==> eqf ==> eqf) (@where_ r A sA sB sC).
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

  #[export] Instance reshape_app_split'_Proper_rank {A r1 r2 R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (@reshape_app_split' A r1 r2).
  Proof.
    cbv [reshape_app_split' RawIndex.curry_radd].
    repeat intro; eauto.
  Qed.
  #[export] Instance reshape_app_combine'_Proper_rank {A r1 r2 R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (@reshape_app_combine' A r1 r2).
  Proof.
    cbv [reshape_app_combine' RawIndex.uncurry_radd].
    repeat intro; destruct RawIndex.split_radd; cbv [eqfR eqfR_rank pointwise_relation] in *; eauto.
  Qed.
  #[export] Instance reshape_app_split_Proper_rank {A r1 r2 R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (@reshape_app_split A r1 r2) := _.
  #[export] Instance reshape_app_combine_Proper_rank {A r1 r2 R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (@reshape_app_combine A r1 r2) := _.
  #[export] Instance reshape_snoc_split_Proper_rank {A r R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank R ==> eqfR_rank (eqfR_rank R)) (@reshape_snoc_split A r).
  Proof.
    cbv [reshape_snoc_split RawIndex.curry_radd].
    repeat intro; eauto.
  Qed.
  #[export] Instance reshape_snoc_combine_Proper_rank {A r R} : Proper ((fun _ _ => True) ==> (fun _ _ => True) ==> eqfR_rank (eqfR_rank R) ==> eqfR_rank R) (@reshape_snoc_combine A r).
  Proof.
    cbv [reshape_snoc_combine RawIndex.uncurry_radd].
    repeat intro; destruct RawIndex.split_radd; cbv [eqfR eqfR_rank pointwise_relation] in *; eauto.
  Qed.

  #[export] Instance reshape_app_split'_Proper {A r1 r2 s1 s2 R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_app_split' A r1 r2 s1 s2).
  Proof. repeat intro; eapply reshape_app_split'_Proper_rank; trivial. Qed.
  #[export] Instance reshape_app_combine'_Proper {A r1 r2 s1 s2 R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_app_combine' A r1 r2 s1 s2).
  Proof. repeat intro; eapply reshape_app_combine'_Proper_rank; trivial. Qed.
  #[export] Instance reshape_app_split_Proper {A r1 r2 s1 s2 R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_app_split A r1 r2 s1 s2) := _.
  #[export] Instance reshape_app_combine_Proper {A r1 r2 s1 s2 R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_app_combine A r1 r2 s1 s2) := _.
  #[export] Instance reshape_snoc_split_Proper {A r s1 s2 R} : Proper (eqfR R ==> eqfR (eqfR R)) (@reshape_snoc_split A r s1 s2).
  Proof. repeat intro; eapply reshape_snoc_split_Proper_rank; trivial. Qed.
  #[export] Instance reshape_snoc_combine_Proper {A r s1 s2 R} : Proper (eqfR (eqfR R) ==> eqfR R) (@reshape_snoc_combine A r s1 s2).
  Proof. repeat intro; eapply reshape_snoc_combine_Proper_rank; trivial. Qed.
  (*
Definition uncurry {r A} {s : Shape r} : @RawIndex.curriedT r A -> tensor A s
  := RawIndex.uncurry.
Definition curry {r A} {s : Shape r} : tensor A s -> @RawIndex.curriedT r A
  := RawIndex.curry.
   *)

  #[export] Instance map'_Proper {ra1 ra2 rb A B sa1 sa2 sb RA RB} : Proper ((eqfR RA ==> eqfR RB) ==> eqfR RA ==> eqfR RB) (@map' ra1 ra2 rb A B sa1 sa2 sb).
  Proof.
    cbv [map']; repeat intro.
    apply reshape_app_combine_Proper.
    eapply map_Proper_R; try eassumption.
    apply reshape_app_split_Proper; eassumption.
  Qed.
  #[export] Instance map2'_Proper {ri1 ri2 ro A B C sA1 sB1 sA2 sB2 so RA RB RC} : Proper ((eqfR RA ==> eqfR RB ==> eqfR RC) ==> eqfR RA ==> eqfR RB ==> eqfR RC) (@map2' ri1 ri2 ro A B C sA1 sB1 sA2 sB2 so).
  Proof.
    cbv [map2']; repeat intro.
    apply reshape_app_combine_Proper.
    eapply map2_Proper_R; try eassumption.
    all: apply reshape_app_split_Proper; eassumption.
  Qed.

  #[export] Instance broadcast_Proper {r A s R} : Proper (eqfR R ==> forall_relation (fun r' => eqfR R)) (@broadcast r A s).
  Proof.
    cbv [broadcast broadcast' repeat']; intros ??? ?.
    apply reshape_app_combine_Proper.
    intro; assumption.
  Qed.
  #[export] Instance repeat_Proper {r A s R} : Proper (eqfR R ==> forall_relation (fun r' => forall_relation (fun s' => eqfR R))) (@repeat r A s).
  Proof.
    cbv [repeat repeat']; intros ??? ? ?.
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

  #[export] Instance reduce_axis_m1'_Proper {r A B s1 s2 RA RB} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB) (@reduce_axis_m1' r A B s1 s2).
  Proof.
    cbv [reduce_axis_m1'].
    intros ?? H ?? Ht.
    eapply map_Proper_R; try eapply reshape_snoc_split_Proper; try eassumption.
    intros ?? Ht'.
    cbv [pointwise_relation eqfR respectful] in *.
    eauto.
  Qed.

  #[export] Instance reduce_axis_m1_Proper {r A B s1 s2 keepdim RA RB}
    : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB)
        (@reduce_axis_m1 r A B s1 s2 keepdim).
  Proof.
    cbv [reduce_axis_m1]; destruct keepdim; intros ?? H ?? Ht ?.
    all: eapply reduce_axis_m1'_Proper; try eassumption.
  Qed.


  (*#[export] Instance reduce_axis_m1'_Proper' {r A B s1 s2 reduction RA} : Proper (eqfR RA ==> eqf) (@reduce_axis_m1' r A B s1 s2 reduction).
Proof.
  cbv [reduce_axis_m1'].
  intros ?? Ht.
  eapply map_Proper_R; try eapply reshape_snoc_split_Proper; try eassumption.
  intros ?? Ht'.
  cbv [pointwise_relation eqfR respectful] in *.
  eauto.
Qed.

#[export] Instance reduce_axis_m1_Proper {r A B s1 s2 keepdim RA RB}
  : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ RA ==> RB))) ==> eqfR RA ==> eqfR RB)
      (@reduce_axis_m1 r A B s1 s2 keepdim).
Proof.
  cbv [reduce_axis_m1]; destruct keepdim; intros ?? H ?? Ht ?.
  all: eapply reduce_axis_m1'_Proper; try eassumption.
Qed.

   *)
  #[export] Instance softmax_dim_m1_Proper {r A B C addB expA zeroB divB s0 s'}
    : Proper (eqf ==> eqf) (@softmax_dim_m1 r A B C addB expA zeroB divB s0 s').
  Proof.
    intros ?? Ht.
    cbv [softmax_dim_m1 div].
    eapply tensor_div_by_Proper.
    all: try eapply (@reduce_axis_m1_Proper r B B s0 s' true).
    all: try (eapply map_Proper; try eassumption; repeat intro).
    all: try eapply Reduction.sum_Proper_pointwise.
    all: try exact eq_refl.
    Unshelve.
    repeat intro; subst; reflexivity.
  Qed.

  #[export] Instance log_softmax_dim_m1_Proper {r A B C D addB lnA expA zeroB divB s0 s'}
    : Proper (eqf ==> eqf) (@log_softmax_dim_m1 r A B C D addB lnA expA zeroB divB s0 s').
  Proof.
    intros ?? Ht.
    cbv [log_softmax_dim_m1 div].
    eapply tensor_div_by_Proper; try eassumption.
    eapply map_Proper; try eassumption.
    all: try eapply (@reduce_axis_m1_Proper r B B s0 s' true).
    all: try (eapply map_Proper; try eassumption; repeat intro).
    all: try eapply Reduction.sum_Proper_pointwise.
    all: try (repeat intro; exact eq_refl).
    Unshelve.
    repeat intro; subst; reflexivity.
  Qed.

  #[export] Instance unsqueeze_dim_m1_Proper {A r s R} : Proper (eqfR R ==> eqfR R) (@unsqueeze_dim_m1 A r s).
  Proof. intros ?? H; cbv; intros; apply H. Qed.

  #[export] Instance gather_dim_m1_Proper {A r ssinput ssindex sinput' sindex' R} : Proper (eqfR R ==> eqf ==> eqfR R) (@gather_dim_m1 A r ssinput ssindex sinput' sindex').
  Proof.
    intros ?? H1 ?? H2; cbv [gather_dim_m1]; intro.
    rewrite H2.
    apply H1.
  Qed.

  #[export] Instance squeeze_Proper {A r s R} : Proper (eqfR R ==> eqfR R) (@squeeze A r s).
  Proof. intros ?? H; cbv; intros; apply H. Qed.
  #[export] Instance reshape_m1_Proper {A r s R} : Proper (eqfR R ==> eqfR R) (@reshape_m1 A r s).
  Proof. intros ?? H ?; cbv [reshape_m1]; apply H. Qed.
  #[export] Instance unreshape_m1_Proper {A r s R} : Proper (eqfR R ==> eqfR R) (@unreshape_m1 A r s).
  Proof. intros ?? H ?; cbv [unreshape_m1]; apply H. Qed.

  #[export] Instance to_bool_Proper {A zero eqb r s} : Proper (eqf ==> eqf) (@to_bool A zero eqb r s).
  Proof.
    intros ?? H ?; cbv [to_bool]; apply map_Proper; try assumption; repeat intro; reflexivity.
  Qed.

  #[export] Instance of_bool_Proper {A zero one r s} : Proper (eqf ==> eqf) (@of_bool A zero one r s).
  Proof.
    intros ?? H ?; cbv [of_bool]; apply map_Proper; try assumption; repeat intro; reflexivity.
  Qed.


  #[export] Instance mean_Proper {r A s B C zero addA div_boyABC coerB} : Proper (eqf ==> eqf) (@mean r A s B C zero addA div_boyABC coerB).
  Proof.
    cbv [mean]; intros ?? H ?.
    eapply reduce_axis_m1_Proper.
    1: eapply Reduction.mean_Proper_pointwise.
    apply reshape_m1_Proper.
    assumption.
  Qed.

  (*(* TODO: nary *)
Definition tupleify {A B s1 s2} (t1 : tensor A [s1]) (t2 : tensor B [s2]) : tensor (A * B) [s1; s2]
  := fun '((tt, a), b) => (raw_get t1 [a], raw_get t2 [b]).
Definition cartesian_prod {A s1 s2} (t1 : tensor A [s1]) (t2 : tensor A [s2]) : tensor A [s1 * s2; 2]
  := fun '((tt, idx), tuple_idx)
     => let '(a, b) := raw_get (reshape_m1 (tupleify t1 t2)) [idx] in
        nth_default a [a; b] (Z.to_nat (Uint63.to_Z (tuple_idx mod 2))).
   *)
  #[export] Instance tril_Proper {A zeroA rnk s r c diagonal} : Proper (eqf ==> eqf) (@tril A zeroA rnk s r c diagonal).
  Proof.
    cbv [tril]; intros ?? H [[? ?] ?].
    rewrite H.
    reflexivity.
  Qed.
  #[export] Instance triu_Proper {A zeroA rnk s r c diagonal} : Proper (eqf ==> eqf) (@triu A zeroA rnk s r c diagonal).
  Proof.
    cbv [triu]; intros ?? H [[? ?] ?].
    rewrite H.
    reflexivity.
  Qed.
End Tensor.
Export (hints) Tensor.
