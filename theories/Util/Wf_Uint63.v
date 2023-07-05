From Coq Require Import Uint63 ZArith Wellfounded Wf_Z Wf_nat.
Local Open Scope uint63_scope.

#[local] Coercion Uint63.to_Z : int >-> Z.
#[local] Coercion Z.to_nat : Z >-> nat.
#[local] Coercion is_true : bool >-> Sortclass.
Definition ltof {A} (f : A -> int) (a b : A) := f a <? f b.

Lemma well_founded_ltof {A f} : well_founded (@ltof A f).
Proof.
  unshelve eapply well_founded_lt_compat with (fun x:A => f x:nat); cbv [is_true ltof].
  intros *; rewrite Uint63.ltb_spec, Z2Nat.inj_lt by apply to_Z_bounded; trivial.
Qed.

Lemma lt_wf : well_founded ltb.
Proof.
  apply @well_founded_ltof with (f:=fun x => x).
Qed.
