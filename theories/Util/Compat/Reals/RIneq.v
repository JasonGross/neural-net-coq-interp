From Coq Require Import Reals RIneq.
#[local] Open Scope R_scope.

Lemma Rdiv_mult_distr : forall r1 r2 r3, r1 / (r2 * r3) = r1 / r2 / r3.
Proof. now unfold Rdiv; intros r1 r2 r3; rewrite Rinv_mult, Rmult_assoc. Qed.
