From Coq Require Import Bool ZArith NArith Sint63 Uint63 List PArray Wellfounded Lia.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63 Slice Arith.Classes Arith.Instances Default Notations Bool PArray.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Open Scope uint63_scope.
Import NeuralNetInterp.Util.PArray.

#[local] Coercion is_true : bool >-> Sortclass.
Module PArray.
  Lemma get_init_default {A default len f i}
    : (@init_default A default len f).[i] = if (i <? len) && (i <? max_length) then f i else default.
  Proof.
    cbv [init_default].
    cbv [for_loop_lt Classes.ltb Classes.leb].
    set (init := PArray.make _ _).
    cbn [Uint63.eqb].
    set (initi := 0).
    assert (Hlen : PArray.length init = (if len ≤? max_length then len else max_length))
      by now
           subst init; rewrite length_make; destruct (PrimInt63.leb len _) eqn:H;
      rewrite ?H; try reflexivity;
      try now case lebP.
    assert (Hsmall : initi <=? (if len ≤? max_length then len else max_length)).
    { destruct (to_Z_bounded (if len ≤? max_length then len else max_length)).
      subst initi; case lebP; rewrite to_Z_0; generalize dependent (to_Z (if len ≤? max_length then len else max_length)); try reflexivity; try lia. }
    assert (H : init.[i] = if i <? initi then f i else default).
    { pose proof (H' := Uint63.ltb_spec i initi).
      subst init.
      rewrite get_make.
      subst initi.
      rewrite to_Z_0 in H'.
      pose proof (to_Z_bounded i).
      destruct (i <? 0)%uint63; [ | reflexivity ].
      lia. }
    clearbody init initi.
    cbv [Fix].
    set (wf:=Acc_intro_generator _ _ _).
    clearbody wf.
    revert init H Hlen Hsmall.
    cbv [pointed] in *.
    revert initi wf.
    fix IH 2.
    intros initi wf init H Hlen.
    destruct wf.
    cbn [Fix_F Acc_inv].
    repeat destruct Sumbool.sumbool_of_bool; auto; intros.
    all: cbv [run_body Monad.bind LoopBody_Monad Wf_Uint63.bind LoopNotation.get LoopNotation.set LoopNotation.update andb is_true] in *.
    all: lazymatch goal with
         | [ |- context[Fix_F _ _ ?wf] ] => specialize (IH _ wf)
         | _ => clear IH
         end.
    1: rewrite IH; clear IH; rewrite ?length_set, ?Hlen; try reflexivity.
    all: repeat first [ assumption
                      | reflexivity
                      | progress rewrite Uint63.add_spec in *
                      | progress rewrite Uint63.sub_spec in *
                      | progress rewrite Uint63.to_Z_1 in *
                      | progress subst
                      | match goal with
                        | [ H : PrimInt63.ltb _ _ = true |- _ ] => rewrite Uint63.ltb_spec in H
                        | [ H : PrimInt63.leb _ _ = true |- _ ] => rewrite Uint63.leb_spec in H
                        | [ |- context[if PrimInt63.ltb ?x ?y then _ else _] ]
                          => destruct (Uint63.ltb_spec x y); destruct (PrimInt63.ltb x y)
                        | [ H : context[if PrimInt63.ltb ?x ?y then _ else _] |- _ ]
                          => destruct (Uint63.ltb_spec x y); destruct (PrimInt63.ltb x y)
                        | [ H : context[if PrimInt63.leb ?x ?y then _ else _] |- _ ]
                          => destruct (Uint63.leb_spec x y); destruct (PrimInt63.leb x y)
                        | [ |- PrimInt63.leb _ _ = true ] => apply Uint63.leb_spec
                        | [ H : true = true -> _ |- _ ] => specialize (H eq_refl)
                        | [ H : false = true -> _ |- _ ] => clear H
                        | [ H : _ -> true = true |- _ ] => clear H
                        | [ H : (?x < ?y)%Z -> false = true, H' : (?x < ?y + 1)%Z |- _ ]
                          => assert (x = y) by lia; clear H H'
                        | [ H : (?x <? ?y) = false |- _ ]
                          => let H' := fresh in
                             pose proof (Uint63.ltb_spec x y) as H';
                             rewrite H in H'; destruct H'; clear H
                        | [ H : context[to_Z ?x] |- _ ]
                          => lazymatch goal with
                             | [ H' : (0 <= to_Z x < wB)%Z |- _ ] => fail
                             | _ => idtac
                             end;
                             pose proof (to_Z_bounded x)
                        | [ H : context[(?x mod ?y)%Z] |- _ ]
                          => rewrite (Z.mod_small x y) in H by lia
                        | [ |- context[(?x mod ?y)%Z] ]
                          => rewrite (Z.mod_small x y) by lia
                        | [ H : to_Z _ = to_Z _ |- _ ] => apply to_Z_inj in H
                        | [ |- context[ ?xs.[?i<-?v].[?j] ] ]
                          => progress replace i with j by (apply to_Z_inj; lia)
                        end
                      | rewrite get_set_other by (intro; subst; lia)
                      | rewrite get_set_same by (apply Uint63.ltb_spec; rewrite ?Hlen; lia)
                      | exfalso; lia
                      | lia ].
  Qed.
End PArray.
