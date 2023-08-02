From Coq Require Import Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Default Wf_Uint63.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead.
From NeuralNetInterp.Util.Relations Require Relation_Definitions.Hetero Relation_Definitions.Dependent.
From NeuralNetInterp.Util.Classes Require Morphisms.Dependent RelationPairs.Hetero.
Import Dependent.ProperNotations.
Local Open Scope uint63_scope.

Definition LoopBody_relation : Dependent.relation2 (@LoopBody_)
  := fun S S' RS A A' RA x y
     => match x, y with
        | break x, break y => RS x y
        | continue x, continue y => RS x y
        | ret x stx, ret y sty
          => RA x y /\ RS stx sty
        | break _, _
        | continue _, _
        | ret _ _, _
          => False
        end.
Global Arguments LoopBody_relation {_ _} _ {_ _} _.

#[export] Instance LoopBody_relation_Reflexive {S A RS RA} {_ : Reflexive RS} {_ : Reflexive RA} : Reflexive (@LoopBody_relation S S RS A A RA).
Proof. cbv; repeat intro; repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end; try split; reflexivity. Qed.
#[export] Instance LoopBody_relation_Symmetric {S A RS RA} {_ : Symmetric RS} {_ : Symmetric RA} : Symmetric (@LoopBody_relation S S RS A A RA).
Proof. cbv; repeat intro; repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end; try split; try symmetry; intuition eauto. Qed.
#[export] Instance LoopBody_relation_Transitive {S A RS RA} {_ : Transitive RS} {_ : Transitive RA} : Transitive (@LoopBody_relation S S RS A A RA).
Proof.
  cbv; intros ???; repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end; repeat intro; try split; try (now (idtac + exfalso)).
  all: repeat match goal with H : _ /\ _ |- _ => destruct H end.
  all: try (etransitivity; multimatch goal with H : _ |- _ => exact H end).
Qed.

Definition LoopBody_eq_relation {A} : Dependent.relation (fun S => @LoopBody_ S A)
  := fun S S' RS => @LoopBody_relation S S' RS A A eq.

#[export] Instance for_loop_lt_Proper_dep : Dependent.Proper (Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR ==> LoopBody_eq_relation) ==> Dependent.idR ==> Dependent.idR) (@for_loop_lt).
Proof.
  intros ??? i _ <- ??? ??? f g H init init' Hinit; subst.
  cbv [for_loop_lt Fix].
  set (wf := Acc_intro_generator _ _ _); clearbody wf.
  revert i wf init init' Hinit.
  fix IH 2.
  intros i wf init init' Hinit.
  destruct wf as [wf].
  cbn [Fix_F Acc_inv].
  set (wf' := wf _).
  specialize (fun v => IH _ (wf' v)).
  clearbody wf'.
  cbv [run_body].
  do 2 (destruct Sumbool.sumbool_of_bool; eauto).
  all: match goal with
       | [ H :  Dependent.respectful _ (_ ==> LoopBody_eq_relation) _ _ _ ?f ?g |- context[?f ?i ?init] ]
         => generalize (H i i eq_refl init _ ltac:(eassumption))
       end.
  all: destruct f, g; cbv [LoopBody_eq_relation LoopBody_relation]; intros; try now (idtac + exfalso); eauto.
  all: repeat first [ progress subst
                    | match goal with
                      | [ H : _ /\ _ |- _ ] => destruct H
                      | [ |- context[match ?x with _ => _ end] ] => destruct x eqn:?
                      end ].
  all: eauto.
Qed.

#[export] Instance for_loop_lt_Proper_R {A R} : Proper (eq ==> eq ==> eq ==> (pointwise_relation _ (R ==> LoopBody_relation R eq)) ==> R ==> R) (@for_loop_lt A).
Proof.
  repeat intro; eapply for_loop_lt_Proper_dep; cbv in *; intros; subst; eauto.
  let H := multimatch goal with H : _ |- _ => H end in
  now eapply H; eauto.
Qed.

#[export] Instance for_loop_lt_Proper {A} : Proper (eq ==> eq ==> eq ==> (pointwise_relation _ (pointwise_relation _ eq)) ==> eq ==> eq) (@for_loop_lt A).
Proof.
  intros i _ <- ??? ??? f g H init _ <-; subst.
  apply (@for_loop_lt_Proper_R A eq); try reflexivity; repeat intro; subst; rewrite H; reflexivity.
Qed.

#[export] Instance map_reduce_Proper_dep
  : Dependent.Proper2 ((Dependent.lift2_2 Dependent.idR ==> Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 Dependent.idR) ==> Dependent.lift2_2 Dependent.idR ==> Dependent.const2 eq ==> Dependent.const2 eq ==> Dependent.const2 eq ==> (Dependent.const2 eq ==> Dependent.lift2_1 Dependent.idR) ==> Dependent.lift2_2 Dependent.idR) (@map_reduce).
Proof.
  cbv [map_reduce]; repeat intro.
  eapply for_loop_lt_Proper_dep; try eassumption; repeat intro.
  cbv; split; subst; cbv in *; eauto.
Qed.

#[export] Instance map_reduce_Proper_R {A B RA RB}
  : Proper ((RB ==> RA ==> RB) ==> RB ==> eq ==> eq ==> eq ==> (pointwise_relation _ RA) ==> RB) (@map_reduce A B).
Proof. repeat intro; eapply map_reduce_Proper_dep; cbv in *; try eassumption; intros; subst; eauto. Qed.

#[export] Instance map_reduce_Proper {A B}
  : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq ==> eq ==> (pointwise_relation _ eq) ==> eq) (@map_reduce A B).
Proof.
  repeat intro; eapply map_reduce_Proper_R; try eassumption; cbv in *; intros; subst; eauto.
Qed.

#[export] Instance map_reduce_no_init_Proper_dep
  : Dependent.Proper ((Dependent.idR ==> Dependent.idR ==> Dependent.idR) ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR) ==> Dependent.idR) (@map_reduce_no_init).
Proof.
  cbv [map_reduce_no_init]; repeat intro.
  eapply for_loop_lt_Proper_dep; try eassumption; repeat intro.
  all: subst; eauto.
  cbv; split; subst; cbv in *; eauto.
Qed.
#[export] Instance map_reduce_no_init_Proper_R {A R}
  : Proper ((R ==> R ==> R) ==> eq ==> eq ==> eq ==> (pointwise_relation _ R) ==> R) (@map_reduce_no_init A).
Proof.
  repeat intro; eapply map_reduce_no_init_Proper_dep; cbv in *; intros; subst; eauto.
Qed.

#[export] Instance map_reduce_no_init_Proper {A}
  : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq ==> (pointwise_relation _ eq) ==> eq) (@map_reduce_no_init A).
Proof.
  repeat intro; eapply map_reduce_no_init_Proper_R; try eassumption; cbv in *; intros; subst; eauto.
Qed.

Module Reduction.
  Export Wf_Uint63.Reduction.
  #[export] Instance sum_Proper_dep
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR)
           ==> Dependent.idR)
        (@sum).
  Proof.
    cbv [sum]; repeat intro.
    eapply map_reduce_Proper_dep; repeat intro; cbv in *; eauto.
  Qed.
  #[export] Instance sum_Proper_pointwise {A zeroA addA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@sum A zeroA addA).
  Proof. repeat intro; eapply sum_Proper_dep; cbv in *; intros; subst; eauto. Qed.
  #[export] Instance sum_Proper {A zeroA addA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@sum A zeroA addA start stop step).
  Proof. apply sum_Proper_pointwise. Qed.
  #[export] Instance prod_Proper_dep
    : Dependent.Proper
        (Dependent.idR
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR)
           ==> Dependent.idR)
        (@prod).
  Proof.
    cbv [prod]; repeat intro.
    eapply map_reduce_Proper_dep; repeat intro; cbv in *; eauto.
  Qed.
  #[export] Instance prod_Proper_pointwise {A oneA mulA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@prod A oneA mulA).
  Proof. repeat intro; eapply prod_Proper_dep; cbv in *; intros; subst; eauto. Qed.
  #[export] Instance prod_Proper {A oneA mulA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@prod A oneA mulA start stop step).
  Proof. apply prod_Proper_pointwise. Qed.
  #[export] Instance max_Proper_dep
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR)
           ==> Dependent.idR)
        (@max).
  Proof.
    cbv [max]; repeat intro.
    eapply map_reduce_no_init_Proper_dep; repeat intro; cbv [pointwise_relation respectful] in *; hnf in *; subst; eauto.
    cbv in *; eauto.
  Qed.
  #[export] Instance max_Proper_pointwise {A maxA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@max A maxA).
  Proof. repeat intro; eapply max_Proper_dep; cbv in *; intros; subst; eauto. Qed.
  #[export] Instance max_Proper {A maxA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@max A maxA start stop step).
  Proof. apply max_Proper_pointwise. Qed.
  #[export] Instance min_Proper_dep
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR)
           ==> Dependent.idR)
        (@min).
  Proof.
    cbv [min]; repeat intro.
    eapply map_reduce_no_init_Proper_dep; repeat intro; cbv [pointwise_relation respectful] in *; hnf in *; subst; eauto.
    cbv in *; eauto.
  Qed.
  #[export] Instance min_Proper_pointwise {A minA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@min A minA).
  Proof. repeat intro; eapply min_Proper_dep; cbv in *; intros; subst; eauto. Qed.
  #[export] Instance min_Proper {A minA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@min A minA start stop step).
  Proof. apply min_Proper_pointwise. Qed.
  #[export] Instance argmax_Proper_dep
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.const eq)
           ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR)
           ==> Dependent.const eq)
        (@argmax).
  Proof.
    cbv [argmax]; intros A B R; repeat intro.
    match goal with
    | [ |- fst ?x = fst ?y ]
      => cut (Hetero.RelProd eq R x y); [ now intros [? ?]; hnf in * | ]
    end.
    eapply map_reduce_no_init_Proper_dep; cbv [argmax_ Dependent.respectful] in *; try assumption.
    all: repeat intro; hnf in *; destruct_head'_and; hnf in *; destruct_head'_prod; cbv in *; subst; eauto.
    match goal with H : _ |- _ => erewrite H by eassumption end.
    break_innermost_match; eauto.
  Qed.
  #[export] Instance argmax_Proper_pointwise {A ltbA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@argmax A ltbA).
  Proof. repeat intro; eapply (argmax_Proper_dep _ _ eq); cbv in *; intros; subst; eauto. Qed.
  #[export] Instance argmax_Proper {A ltbA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@argmax A ltbA start stop step).
  Proof. apply argmax_Proper_pointwise. Qed.
  #[export] Instance argmin_Proper_dep
    : Dependent.Proper
        ((Dependent.idR ==> Dependent.idR ==> Dependent.const eq)
           ==> Dependent.const eq ==> Dependent.const eq ==> Dependent.const eq ==> (Dependent.const eq ==> Dependent.idR)
           ==> Dependent.const eq)
        (@argmin).
  Proof.
    cbv [argmin]; intros A B R; repeat intro.
    match goal with
    | [ |- fst ?x = fst ?y ]
      => cut (Hetero.RelProd eq R x y); [ now intros [? ?]; hnf in * | ]
    end.
    eapply map_reduce_no_init_Proper_dep; cbv [argmin_ Dependent.respectful] in *; try assumption.
    all: repeat intro; hnf in *; destruct_head'_and; hnf in *; destruct_head'_prod; cbv in *; subst; eauto.
    match goal with H : _ |- _ => erewrite H by eassumption end.
    break_innermost_match; eauto.
  Qed.
  #[export] Instance argmin_Proper_pointwise {A lebA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@argmin A lebA).
  Proof. repeat intro; eapply (argmin_Proper_dep _ _ eq); cbv in *; intros; subst; eauto. Qed.
  #[export] Instance argmin_Proper {A lebA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@argmin A lebA start stop step).
  Proof. apply argmin_Proper_pointwise. Qed.
  #[export] Instance mean_Proper_dep
    : Dependent.Proper3
        (Dependent.lift3_1 Dependent.idR
           ==> (Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_1 Dependent.idR)
           ==> (Dependent.lift3_1 Dependent.idR ==> Dependent.lift3_2 Dependent.idR ==> Dependent.lift3_3 Dependent.idR)
           ==> (Dependent.const3 eq ==> Dependent.lift3_2 Dependent.idR)
           ==> Dependent.const3 eq ==> Dependent.const3 eq ==> Dependent.const3 eq ==> (Dependent.const3 eq ==> Dependent.lift3_1 Dependent.idR)
           ==> Dependent.lift3_3 Dependent.idR)
        (@mean).
  Proof.
    intros ?? RA ?? RB ?? RC ?? zeroA_Proper_dep ?? addA_Proper_dep ?? div_by_Proper_dep ?? coerB_Proper_dep; repeat intro; hnf in *.
    cbv [mean].
    apply div_by_Proper_dep; repeat intro; hnf in *.
    all: first [ apply coerB_Proper_dep | eapply sum_Proper_dep ]; repeat intro; hnf in *.
    all: subst; eauto.
    all: cbv in *; eauto.
  Qed.
  #[export] Instance mean_Proper_pointwise {A B C zeroA addA div_byABC coerB} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@mean A B C zeroA addA div_byABC coerB).
  Proof. repeat intro; eapply mean_Proper_dep; cbv in *; intros; try reflexivity; subst; try reflexivity; subst; eauto. Qed.
  #[export] Instance mean_Proper {A B C zeroA addA div_byABC coerB start stop step} : Proper (pointwise_relation _ eq ==> eq) (@mean A B C zeroA addA div_byABC coerB start stop step).
  Proof. apply mean_Proper_pointwise. Qed.
  Print var.
  #[export] Instance var_Proper_dep
    : Dependent.Proper2
        (Dependent.lift2_1 Dependent.idR
           ==> (Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_1 Dependent.idR)
           ==> (Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_1 Dependent.idR)
           ==> (Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_1 Dependent.idR)
           ==> (Dependent.lift2_1 Dependent.idR ==> Dependent.lift2_2 Dependent.idR ==> Dependent.lift2_1 Dependent.idR)
           ==> (Dependent.const2 eq ==> Dependent.lift2_2 Dependent.idR)
           ==> Dependent.const2 eq
           ==> Dependent.const2 eq ==> Dependent.const2 eq ==> Dependent.const2 eq ==> (Dependent.const2 eq ==> Dependent.lift2_1 Dependent.idR)
           ==> Dependent.lift2_1 Dependent.idR)
        (@var).
  Proof.
    intros ?? RA ?? RB ?? zeroA_Proper_dep ?? addA_Proper_dep ?? mulA_Proper_dep ?? subA_Proper_dep ?? div_by_Proper_dep ?? coerB_Proper_dep; repeat intro; hnf in *.
    cbv [var sqr].
    apply div_by_Proper_dep; repeat intro; hnf in *.
    all: first [ apply coerB_Proper_dep | eapply sum_Proper_dep ]; repeat intro; hnf in *.
    all: try apply mulA_Proper_dep; repeat intro; hnf in *.
    all: try apply subA_Proper_dep; repeat intro; hnf in *.
    all: subst; eauto.
    all: cbv in * |- ; eauto.
    all: eapply mean_Proper_dep; repeat intro; hnf in *.
    all: eauto.
  Qed.
  #[export] Instance var_Proper_pointwise {A B zeroA addA mulA subA div_byAB coerB correction} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@var A B zeroA addA mulA subA div_byAB coerB correction).
  Proof. repeat intro; eapply var_Proper_dep; cbv in *; intros; try reflexivity; subst; try reflexivity; subst; eauto. Qed.
  #[export] Instance var_Proper {A B zeroA addA mulA subA div_byAB coerB correction start stop step} : Proper (pointwise_relation _ eq ==> eq) (@var A B zeroA addA mulA subA div_byAB coerB correction start stop step).
  Proof. apply var_Proper_pointwise. Qed.
End Reduction.
Export (hints) Reduction.
