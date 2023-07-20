From Coq Require Import Sint63 Uint63 Setoid Morphisms Eqdep_dec.
From NeuralNetInterp.Torch Require Import Tensor Slicing Tensor.Instances.
From NeuralNetInterp.Util Require Import Slice Arith.Classes Arith.Instances PolymorphicOption Nat Notations.
From NeuralNetInterp.Util.Tactics Require Import DestructHead BreakMatch.
Set Implicit Arguments.

Definition invert_FancyIndexType {r s ri ro} (x : @FancyIndexType r s ri ro)
  : (tensor IndexType s * (ri = 1 /\ ro = 0)) + (SliceIndexType ri ro)
  := match x with
     | tensor_index t => inl (t, (conj eq_refl eq_refl))
     | normal_index s => inr s
     end.

Definition uninvert_FancyIndexType {r s ri ro} (x : (tensor IndexType s * (ri = 1 /\ ro = 0)) + (SliceIndexType ri ro)) : @FancyIndexType r s ri ro
  := match x with
     | inr s => normal_index s
     | inl (t, pf)
       => match ri, ro return ri = 1 /\ ro = 0 -> FancyIndexType s ri ro with
          | 1, 0 => fun _ => tensor_index t
          | 1, n => fun pf => match (ltac:(discriminate) : n = 0 -> False) (proj2 pf) with end
          | n, _ => fun pf => match (ltac:(discriminate) : n = 1 -> False) (proj1 pf) with end
          end pf
     end.

Lemma un_invert_FancyIndexType {r s ri ro} (x : @FancyIndexType r s ri ro) : uninvert_FancyIndexType (invert_FancyIndexType x) = x.
Proof. destruct x; cbn; reflexivity. Qed.

Lemma invert_un_FancyIndexType {r s ri ro} x : invert_FancyIndexType (@uninvert_FancyIndexType r s ri ro x) = x.
Proof.
  destruct x; destruct_head'_prod; destruct_head'_and; subst; cbn; reflexivity.
Qed.

Definition FancyIndexType_relation {r s ri ro} : relation (@FancyIndexType r s ri ro)
  := fun x y
     => match invert_FancyIndexType x, invert_FancyIndexType y with
        | inl (x, _), inl (y, _) => Tensor.eqf x y
        | inr x, inr y => x = y
        | inl _, _
        | inr _, _
          => False
        end.

#[export] Instance FancyIndexType_relation_Reflexive {r s ri ro} : Reflexive (@FancyIndexType_relation r s ri ro).
Proof. cbv -[tensor Tensor.eqf]; intros; destruct_head' (@FancyIndexType); reflexivity. Qed.
#[export] Instance FancyIndexType_relation_Symmetric {r s ri ro} : Symmetric (@FancyIndexType_relation r s ri ro).
Proof. hnf; cbv [FancyIndexType_relation]; intros; break_innermost_match; destruct_head'_prod; try assumption; symmetry; assumption. Qed.
#[export] Instance FancyIndexType_relation_Transitive {r s ri ro} : Transitive (@FancyIndexType_relation r s ri ro).
Proof. hnf; cbv [FancyIndexType_relation]; intros *; break_innermost_match; destruct_head'_prod; intuition auto; etransitivity; typeclasses eauto with core. Qed.

Module SliceIndex.
  Export SliceIndex.
  Module SliceIndexType.
    Export SliceIndexType.

    #[export] Instance slice_Proper {A ris ros ri ro transfer_shape_idxs slice_idxs idx s R}
      {slice_idxs_Proper : forall {s}, Proper (Tensor.eqfR R ==> Tensor.eqfR R) (@slice_idxs s)}
      : Proper (Tensor.eqfR R ==> Tensor.eqfR R)
          (@slice A ris ros ri ro transfer_shape_idxs (@slice_idxs) idx s).
    Proof.
      cbv [slice]; destruct idx; repeat intro; apply slice_idxs_Proper; try assumption; repeat intro; eauto.
    Qed.
  End SliceIndexType.
  Export (hints) SliceIndexType.

  #[export] Instance slice_Proper {A ri ro idxs s R} : Proper (Tensor.eqfR R ==> Tensor.eqfR R) (@slice A ri ro idxs s).
  Proof.
    induction idxs; cbn [slice]; repeat intro; eauto.
    apply SliceIndexType.slice_Proper; eauto.
  Qed.
End SliceIndex.
Export (hints) SliceIndex.

Module FancyIndex.
  Export FancyIndex.
  Module FancyIndexType.
    Export FancyIndexType.
    #[export] Instance broadcast_Proper {rb sb ri ro} : Proper (FancyIndexType_relation ==> Tensor.eqf) (@broadcast rb sb ri ro).
    Proof.
      cbv [broadcast].
      intros x y H idx.
      rewrite <- (un_invert_FancyIndexType x),  <- (un_invert_FancyIndexType y).
      cbv [FancyIndexType_relation] in H.
      break_innermost_match_hyps; destruct_head'_prod; destruct_head'_and; subst; destruct_head'_False; cbn [uninvert_FancyIndexType].
      all: try reflexivity.
      apply Tensor.map_Proper; try assumption; repeat intro; reflexivity.
    Qed.
  End FancyIndexType.
  Export (hints) FancyIndexType.

  Inductive t_relation {rb sb} : forall {ri ro}, relation (@t rb sb ri ro) :=
  | R_nil : t_relation nil nil
  | R_elipsis {r} : t_relation (elipsis (r:=r)) (elipsis (r:=r))
  | R_snoc {ris ros ri ro} : Proper (t_relation ==> FancyIndexType_relation ==> t_relation) (@snoc rb sb ris ros ri ro)
  .

  #[export] Instance t_relation_Reflexive {r s ri ro} : Reflexive (@t_relation r s ri ro).
  Proof. intro H; induction H; constructor; auto; reflexivity. Qed.
  #[export] Instance t_relation_Symmetric {r s ri ro} : Symmetric (@t_relation r s ri ro).
  Proof.
    intros x y H.
    induction H; constructor; eauto; symmetry; assumption.
  Qed.
  #[export] Instance t_relation_Transitive {r s ri ro} : Transitive (@t_relation r s ri ro).
  Proof.
    intros x y z H1 H2.
    induction H1; inversion H2; clear H2; inversion_sigma; subst.
    all: cbv [Rank] in *.
    all: repeat first [ progress cbn in *
                      | match goal with
                        | [ H : ?x = ?x :> nat |- _ ] => pose proof (UIP_refl_nat _ H); subst H
                        end ].
    all: constructor; eauto.
    etransitivity; typeclasses eauto with core.
  Qed.

  #[export] Instance broadcast_Proper {rb sb ri ro}
    : Proper (t_relation ==> Tensor.eqf) (@broadcast rb sb ri ro).
  Proof.
    intros ?? Hidxs.
    induction Hidxs; cbn [broadcast].
    all: try apply Tensor.map2_Proper; cbv [pointwise_relation]; intros.
    all: eauto.
    all: try apply FancyIndexType.broadcast_Proper.
    all: try reflexivity.
    all: try assumption.
  Qed.

  #[export] Instance slice_Proper {A rb sb ri ro}
    : Proper (t_relation ==> forall_relation (fun s => Tensor.eqf ==> Tensor.eqf_rank)) (@slice A rb sb ri ro).
  Proof.
    cbv [slice slice_ Tensor.map_dep]; repeat intro.
    apply Tensor.reshape_app_combine'_Proper_rank; repeat intro; trivial.
    let H := match goal with H : t_relation _ _ |- _ => H end in
    eapply broadcast_Proper in H;
    rewrite H.
    apply SliceIndex.slice_Proper; assumption.
  Qed.
End FancyIndex.
Export (hints) FancyIndex.
