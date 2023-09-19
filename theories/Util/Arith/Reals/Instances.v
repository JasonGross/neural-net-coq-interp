From Coq Require Import RelationClasses Reals.

#[export] Instance Rle_Reflexive : Reflexive Rle | 10 := Rle_refl.
#[export] Instance Rge_Reflexive : Reflexive Rge | 10 := Rge_refl.
#[export] Instance Rlt_Irreflexive : Irreflexive Rlt | 10 := Rlt_irrefl.
#[export] Instance Rgt_Irreflexive : Irreflexive Rgt | 10 := Rgt_irrefl.
#[export] Instance Rlt_Asymmetric : Asymmetric Rlt | 10 := Rlt_asym.
#[export] Instance Rgt_Asymmetric : Asymmetric Rgt | 10 := Rgt_asym.
#[export] Instance Rle_Antisymmetric : Antisymmetric R eq Rle | 10 := Rle_antisym.
#[export] Instance Rge_Antisymmetric : Antisymmetric R eq Rge | 10 := Rge_antisym.
#[export] Instance Rlt_Transitive : Transitive Rlt | 10 := Rlt_trans.
#[export] Instance Rle_Transitive : Transitive Rle | 10 := Rle_trans.
#[export] Instance Rge_Transitive : Transitive Rge | 10 := Rge_trans.
#[export] Instance Rgt_Transitive : Transitive Rgt | 10 := Rgt_trans.
