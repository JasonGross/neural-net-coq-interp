From Coq Require Import Bool Uint63 ZArith Wellfounded Wf_Z Wf_nat Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import Monad Notations Arith.Classes Arith.Instances Default Wf_Uint63.
Local Open Scope uint63_scope.

Definition LoopBody_relation {S A} (RS : relation S) (RA : relation A) : relation (@LoopBody_ S A)
  := fun x y
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

#[export] Instance LoopBody_relation_Reflexive {S A RS RA} {_ : Reflexive RS} {_ : Reflexive RA} : Reflexive (@LoopBody_relation S A RS RA).
Proof. cbv; repeat intro; repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end; try split; reflexivity. Qed.
#[export] Instance LoopBody_relation_Symmetric {S A RS RA} {_ : Symmetric RS} {_ : Symmetric RA} : Symmetric (@LoopBody_relation S A RS RA).
Proof. cbv; repeat intro; repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end; try split; try symmetry; intuition eauto. Qed.
#[export] Instance LoopBody_relation_Transitive {S A RS RA} {_ : Transitive RS} {_ : Transitive RA} : Transitive (@LoopBody_relation S A RS RA).
Proof.
  cbv; intros ???; repeat match goal with |- context[match ?x with _ => _ end] => destruct x eqn:? end; repeat intro; try split; try (now (idtac + exfalso)).
  all: repeat match goal with H : _ /\ _ |- _ => destruct H end.
  all: try (etransitivity; multimatch goal with H : _ |- _ => exact H end).
Qed.

#[export] Instance for_loop_lt_Proper_R {A R} : Proper (eq ==> eq ==> eq ==> (pointwise_relation _ (R ==> LoopBody_relation R eq)) ==> R ==> R) (@for_loop_lt A).
Proof.
  intros i _ <- ??? ??? f g H init init' Hinit; subst.
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
       | [ H : pointwise_relation _ (_ ==> LoopBody_relation _ eq) ?f ?g |- context[?f ?i ?init] ]
         => generalize (H i init _ ltac:(eassumption))
       end.
  all: destruct f, g; cbv [LoopBody_relation]; intros; try now (idtac + exfalso); eauto.
  all: repeat first [ progress subst
                    | match goal with
                      | [ H : _ /\ _ |- _ ] => destruct H
                      | [ |- context[match ?x with _ => _ end] ] => destruct x eqn:?
                      end ].
  all: eauto.
Qed.

#[export] Instance for_loop_lt_Proper {A} : Proper (eq ==> eq ==> eq ==> (pointwise_relation _ (pointwise_relation _ eq)) ==> eq ==> eq) (@for_loop_lt A).
Proof.
  intros i _ <- ??? ??? f g H init _ <-; subst.
  apply (@for_loop_lt_Proper_R A eq); try reflexivity; repeat intro; subst; rewrite H; reflexivity.
Qed.

#[export] Instance map_reduce_Proper_R {A B RA RB}
  : Proper ((RB ==> RA ==> RB) ==> RB ==> eq ==> eq ==> eq ==> (pointwise_relation _ RA) ==> RB) (@map_reduce A B).
Proof.
  cbv [map_reduce]; repeat intro.
  eapply for_loop_lt_Proper_R; try eassumption; repeat intro.
  cbv; split; subst; cbv [pointwise_relation respectful] in *; eauto.
Qed.

#[export] Instance map_reduce_Proper {A B}
  : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq ==> eq ==> (pointwise_relation _ eq) ==> eq) (@map_reduce A B).
Proof.
  repeat intro; eapply map_reduce_Proper_R; try eassumption; cbv in *; intros; subst; eauto.
Qed.

#[export] Instance map_reduce_no_init_Proper_R {A R}
  : Proper ((R ==> R ==> R) ==> eq ==> eq ==> eq ==> (pointwise_relation _ R) ==> R) (@map_reduce_no_init A).
Proof.
  cbv [map_reduce_no_init]; repeat intro.
  eapply for_loop_lt_Proper_R; try eassumption; try (subst; reflexivity); repeat intro; eauto.
  all: cbv; try split; subst; cbv [pointwise_relation respectful] in *; eauto.
Qed.

#[export] Instance map_reduce_no_init_Proper {A}
  : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq ==> (pointwise_relation _ eq) ==> eq) (@map_reduce_no_init A).
Proof.
  repeat intro; eapply map_reduce_no_init_Proper_R; try eassumption; cbv in *; intros; subst; eauto.
Qed.

Module Reduction.
  Export Wf_Uint63.Reduction.
  #[export] Instance sum_Proper_pointwise {A zeroA addA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@sum A zeroA addA).
  Proof.
    cbv [sum]; repeat intro.
    eapply map_reduce_Proper; repeat intro; cbv [pointwise_relation respectful] in *; subst; eauto.
  Qed.
  #[export] Instance sum_Proper {A zeroA addA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@sum A zeroA addA start stop step).
  Proof. apply sum_Proper_pointwise. Qed.
  #[export] Instance prod_Proper_pointwise {A oneA mulA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@prod A oneA mulA).
  Proof.
    cbv [prod]; repeat intro.
    eapply map_reduce_Proper; repeat intro; cbv [pointwise_relation respectful] in *; subst; eauto.
  Qed.
  #[export] Instance prod_Proper {A oneA mulA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@prod A oneA mulA start stop step).
  Proof. apply prod_Proper_pointwise. Qed.
  #[export] Instance max_Proper_pointwise {A maxA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@max A maxA).
  Proof.
    cbv [max]; repeat intro.
    eapply map_reduce_no_init_Proper; repeat intro; cbv [pointwise_relation respectful] in *; subst; eauto.
  Qed.
  #[export] Instance max_Proper {A maxA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@max A maxA start stop step).
  Proof. apply max_Proper_pointwise. Qed.
  #[export] Instance min_Proper_pointwise {A minA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@min A minA).
  Proof.
    cbv [min]; repeat intro.
    eapply map_reduce_no_init_Proper; repeat intro; cbv [pointwise_relation respectful] in *; subst; eauto.
  Qed.
  #[export] Instance min_Proper {A minA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@min A minA start stop step).
  Proof. apply min_Proper_pointwise. Qed.
  #[export] Instance argmax_Proper_pointwise {A ltbA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@argmax A ltbA).
  Proof.
    cbv [argmax]; repeat intro.
    erewrite map_reduce_no_init_Proper; try reflexivity.
    cbv in *; intros; congruence.
  Qed.
  #[export] Instance argmax_Proper {A ltbA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@argmax A ltbA start stop step).
  Proof. apply argmax_Proper_pointwise. Qed.
  #[export] Instance argmin_Proper_pointwise {A lebA} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@argmin A lebA).
  Proof.
    cbv [argmin]; repeat intro.
    erewrite map_reduce_no_init_Proper; try reflexivity.
    cbv in *; intros; congruence.
  Qed.
  #[export] Instance argmin_Proper {A lebA start stop step} : Proper (pointwise_relation _ eq ==> eq) (@argmin A lebA start stop step).
  Proof. apply argmin_Proper_pointwise. Qed.
  #[export] Instance mean_Proper_pointwise {A B C zeroA addA div_byABC coerB} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@mean A B C zeroA addA div_byABC coerB).
  Proof.
    cbv [mean]; repeat intro.
    erewrite sum_Proper_pointwise by eassumption.
    reflexivity.
  Qed.
  #[export] Instance mean_Proper {A B C zeroA addA div_byABC coerB start stop step} : Proper (pointwise_relation _ eq ==> eq) (@mean A B C zeroA addA div_byABC coerB start stop step).
  Proof. apply mean_Proper_pointwise. Qed.
  #[export] Instance var_Proper_pointwise {A B zeroA addA mulA subA div_byAB coerB correction} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq ==> eq)))) (@var A B zeroA addA mulA subA div_byAB coerB correction).
  Proof.
    cbv [var]; repeat intro.
    erewrite sum_Proper_pointwise; [ reflexivity | ].
    intro.
    erewrite mean_Proper_pointwise by eassumption.
    cbv [pointwise_relation] in *.
    congruence.
  Qed.
  #[export] Instance var_Proper {A B zeroA addA mulA subA div_byAB coerB correction start stop step} : Proper (pointwise_relation _ eq ==> eq) (@var A B zeroA addA mulA subA div_byAB coerB correction start stop step).
  Proof. apply var_Proper_pointwise. Qed.
End Reduction.
Export (hints) Reduction.
