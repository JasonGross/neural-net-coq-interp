From Coq Require Import Bool ZArith NArith Sint63 Uint63 List PArray Wellfounded Lia Setoid Morphisms.
From NeuralNetInterp.Util Require Import PArray.

Module PArray.
  Local Ltac t_step :=
    first [ progress subst
          | intro
          | apply Wf_Uint63.for_loop_lt_Proper
          | progress cbv [Monad.bind Wf_Uint63.LoopBody_Monad Wf_Uint63.bind Wf_Uint63.get Wf_Uint63.set Wf_Uint63.update]
          | progress cbv [pointwise_relation] in *
          | reflexivity
          | match goal with
            | [ |- context[match ?x with _ => _ end] ] => destruct x eqn:?
            | [ H : context[_ = _] |- _ ] => rewrite H
            end ].
  Local Ltac t := repeat t_step.

  #[export] Instance fill_array_of_list_map_Proper {A B} : Proper ((pointwise_relation _ eq) ==> eq ==> eq ==> eq ==> eq) (@fill_array_of_list_map A B).
  Proof.
    intros f g H ls _ <- start _ <- arr _ <-; cbv in H.
    revert start arr; induction ls as [|?? IH]; cbn; intros; try reflexivity.
    erewrite IH, H by reflexivity.
    reflexivity.
  Qed.

  #[export] Instance array_of_list_map_Proper {A B} : Proper (eq ==> (pointwise_relation _ eq) ==> eq ==> eq) (@array_of_list_map A B).
  Proof.
    cbv [array_of_list_map].
    repeat intro; eapply fill_array_of_list_map_Proper; try assumption; subst; reflexivity.
  Qed.

  #[local] Hint Extern 0 (subrelation ?R1 ?R2) => unify R1 R2; apply subrelation_refl : typeclass_instances.

  #[export] Instance map_default_Proper {A B} : Proper (eq ==> (pointwise_relation _ eq) ==> eq ==> eq) (@map_default A B).
  Proof.
    cbv [map_default].
    intros ??? f g H ???; subst.
    apply Wf_Uint63.for_loop_lt_Proper; try reflexivity; [].
    repeat intro; subst.
    cbv.
    erewrite H by reflexivity; reflexivity.
  Qed.

  #[export] Instance map_Proper {A B} : Proper ((pointwise_relation _ eq) ==> eq ==> eq) (@map A B).
  Proof.
    cbv [map].
    intros f g H ???; subst.
    rewrite !H; reflexivity.
  Qed.

  #[export] Instance init_default_Proper {A} : Proper (eq ==> eq ==> pointwise_relation _ eq ==> eq) (@init_default A).
  Proof.
    cbv [init_default].
    intros ??? ??? f g H; subst.
    apply Wf_Uint63.for_loop_lt_Proper; try reflexivity.
    repeat intro.
    cbv.
    rewrite H; reflexivity.
  Qed.

  #[export] Instance init_Proper {A} : Proper (eq ==> pointwise_relation _ eq ==> eq) (@init A).
  Proof. cbv [init]; repeat intro; subst; apply init_default_Proper; try reflexivity; eauto. Qed.

  #[export] Instance map2_default_Proper {A B C} : Proper (eq ==> pointwise_relation _ (pointwise_relation _ eq) ==> pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq) (@map2_default A B C).
  Proof. cbv [map2_default]. t. Qed.

  #[export] Instance map2_Proper {A B C} : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq) (@map2 A B C).
  Proof.
    cbv [map2 pointwise_relation]; repeat intro; subst; apply map2_default_Proper; eauto.
  Qed.

  #[export] Instance broadcast_map2_Proper {A B C} : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq) (@broadcast_map2 A B C).
  Proof. cbv [broadcast_map2]. t. Qed.

  #[export] Instance broadcast_map3_Proper {A B C D} : Proper (pointwise_relation _ (pointwise_relation _ (pointwise_relation _ eq)) ==> eq ==> eq ==> eq ==> eq) (@broadcast_map3 A B C D).
  Proof. cbv [broadcast_map3]. t. Qed.

  #[export] Instance reduce_Proper {A B} : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq ==> eq) (@reduce A B).
  Proof. cbv [reduce]. t. Qed.

  #[export] Instance reduce_no_init_Proper {A} : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> eq) (@reduce_no_init A).
  Proof. cbv [reduce_no_init]. t. Qed.

  #[export] Instance reduce_map_Proper {A A' B} : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> eq ==> pointwise_relation _ eq ==> eq ==> eq) (@reduce_map A A' B).
  Proof. cbv [reduce_map]. repeat intro; subst; apply reduce_Proper. all: t. Qed.

  #[export] Instance reduce_map_no_init_Proper {A A'} : Proper (pointwise_relation _ (pointwise_relation _ eq) ==> pointwise_relation _ eq ==> eq ==> eq) (@reduce_map_no_init A A').
  Proof. cbv [reduce_map_no_init]. t. Qed.
End PArray.
Export (hints) PArray.
