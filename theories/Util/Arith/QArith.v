From Coq Require Import Lia Lqa Qround Qabs QArith ZArith Morphisms.
From NeuralNetInterp.Util Require Import Default.
From NeuralNetInterp.Util.Arith Require Import ZArith.
From NeuralNetInterp.Util.Tactics Require Import BreakMatch DestructHead.
Set Implicit Arguments.

(* XXX FIXME *)
Definition Qsqrt (v : Q) : Q := Qred (Qmake (Z.sqrt (Qnum v * Zpos (Qden v))) (Qden v)).

Definition pow_Z R (rI : R) (rmul : R -> R -> R) (rdiv : R -> R -> R) (x : R) (p : Z) : R
  := match p with
     | Zneg p => rdiv rI (@pow_N R rI rmul x (Npos p))
     | 0%Z => rI
     | Zpos p => @pow_N R rI rmul x (Npos p)
     end.

#[export] Instance subrelation_eq_Qeq : subrelation eq Qeq.
Proof. repeat intro; subst; reflexivity. Qed.

#[local] Coercion inject_Z : Z >-> Q.

Definition Qround (q : Q) : Z
  := let '(a, b) := (Qfloor q, Qceiling q) in
     let '(aerr, berr) := (q - a, b - q) in
     if Qeq_bool aerr berr
     then (* round to even *)
       if Z.even a then a else b
     else if Qle_bool aerr berr
          then a
          else b.

Lemma Qround_Z (z : Z) : Qround z = z.
Proof. cbv [Qround]; rewrite Qfloor_Z, Qceiling_Z; break_innermost_match; reflexivity. Qed.
Lemma Q_le_floor_ceiling x : (Qfloor x <= Qceiling x)%Z.
Proof. generalize (Qle_floor_ceiling x); rewrite Zle_Qle; auto. Qed.

Lemma Qeq_floor_ceiling x : Qfloor x = if Qeq_bool (Qceiling x) x then Qceiling x else (Qceiling x - 1)%Z.
Proof.
  destruct x as [n d]; cbn.
  cbv [Qceiling Qfloor Qeq_bool Qnum Qden inject_Z Qopp].
  match goal with |- context[Zeq_bool ?x ?y] => pose proof (Zeq_bool_if x y) end.
  destruct Zeq_bool; Z.to_euclidean_division_equations; nia.
Qed.
Lemma Qeq_ceiling_floor x : Qceiling x = if Qeq_bool (Qfloor x) x then Qfloor x else (Qfloor x + 1)%Z.
Proof.
  destruct x as [n d]; cbn.
  cbv [Qceiling Qfloor Qeq_bool Qnum Qden inject_Z Qopp].
  match goal with |- context[Zeq_bool ?x ?y] => pose proof (Zeq_bool_if x y) end.
  destruct Zeq_bool; Z.to_euclidean_division_equations; nia.
Qed.

Lemma Qfloor_ceiling_diff_max q : (0 <= Qceiling q - Qfloor q <= 1)%Z.
Proof.
  cbv [Qfloor Qceiling Qopp Qnum Qden]; destruct q as [n d].
  Z.to_euclidean_division_equations; nia.
Qed.

Lemma Qround_diff_max_cases q
  : (Qround q = Qfloor q /\ 0 <= q - Qround q /\ (q - Qround q < 0.5 \/ (Z.even (Qround q) = true /\ q - Qround q == 0.5)))
    \/ (Qround q = Qceiling q /\ 0 <= Qround q - q /\ (Qround q - q < 0.5 \/ (Z.even (Qround q) = true /\ Qround q - q == 0.5))).
Proof.
  cbv [Qround].
  generalize (Qle_bool_iff (q - Qfloor q) (Qceiling q - q)).
  generalize (@Q_le_floor_ceiling q); generalize (@Qlt_floor q); generalize (@Qceiling_lt q).
  generalize (@Qfloor_le q); generalize (@Qle_ceiling q).
  destruct (@Qfloor_ceiling_diff_max q).
  assert (Qfloor q + 1 == (Qfloor q + 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq]; lia).
  assert (Qceiling q - 1 == (Qceiling q - 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; lia).
  assert (Qceiling q - Qfloor q == (Qceiling q - Qfloor q)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
  assert (1 == 1%Z) by reflexivity.
  assert (0 == 0%Z) by reflexivity.
  rewrite Zle_Qle in *.
  destruct Qle_bool; break_innermost_match; intros; constructor;
    (split; [ reflexivity | ]).
  all: rewrite ?Qeq_bool_iff in *.
  all: split; try lra.
  all: lazymatch goal with |- ?x < ?y \/ _ => let H := fresh in assert (H : x < y \/ x == y) by lra; destruct H; [ left | right; split ] end.
  all: try lra.
  all: try assumption.
  all: rewrite Qeq_ceiling_floor in *.
  all: break_innermost_match; break_innermost_match_hyps.
  all: rewrite ?Qeq_bool_iff, ?Qeq_bool_neq in *.
  all: repeat match goal with H : Qeq_bool _ _ = false |- _ => apply Qeq_bool_neq in H end.
  all: try lra.
  all: assert (Qfloor q + 1 == (Qfloor q + 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
  all: try lra.
  all: let H := fresh in
       assert (H : Qfloor q == q - 0.5) by lra;
       rewrite H in *.
  all: try replace (Qfloor q + 1)%Z with (Z.succ (Qfloor q)) by lia; rewrite ?Z.even_succ, <- ?Z.negb_even.
  all: try (destruct Z.even; try reflexivity; congruence).
Qed.

Lemma Qround_diff_max q : Qabs (q - Qround q) <= 0.5.
Proof.
  destruct (@Qround_diff_max_cases q) as [[_ H]|[_ H]].
  { rewrite Qabs_pos by lra; lra. }
  { rewrite Qabs_neg by lra; lra. }
Qed.

Lemma Qround_resp_le : Proper (Qle ==> Z.le) Qround.
Proof.
  intros x y H.
  pose proof (@Qfloor_resp_le x y H); pose proof (@Qceiling_resp_le x y H).
  all: destruct (@Qround_diff_max_cases x); generalize dependent (Qround x); intros; subst.
  all: destruct (@Qround_diff_max_cases y); generalize dependent (Qround y); intros; subst.
  all: repeat (destruct_head'_and; destruct_head'_or); subst.
  all: rewrite Zle_Qle in *.
  all: try lra.
  all: assert (Qfloor y - Qceiling x == (Qfloor y - Qceiling x)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
  all: assert (0 == 0%Z) by reflexivity.
  all: assert (-1 == (-1)%Z) by reflexivity.
  all: try assert (Qceiling x < x + 0.5) by lra.
  all: try assert (y - 0.5 <= Qfloor y) by lra.
  all: try assert (y - x - 1 < Qfloor y - Qceiling x) by lra.
  all: assert (H' : 0 <= y - x < 1 \/ 1 < y - x \/ y == x + 1) by lra.
  all: destruct H' as [?|[?|?]]; try lra.
  all: try assert (-1 < Qfloor y - Qceiling x) by lra.
  all: try assert ((-1 < Qfloor y - Qceiling x)%Z) by (rewrite Zlt_Qlt; lra).
  all: try assert ((0 <= Qfloor y - Qceiling x)%Z) by lia.
  all: try rewrite Zle_Qle in *.
  all: try lra.
  { assert ((Qfloor y - Qceiling x)%Z == y - x - 1) by lra.
    assert ((Qfloor y - Qceiling x)%Z + 1 == y - x) by lra.
    assert ((Qfloor y - Qceiling x)%Z + 1 == (Qfloor y - Qceiling x + 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
    assert (0 <= (Qfloor y - Qceiling x + 1)%Z < 1) by lra.
    assert (0 <= (Qfloor y - Qceiling x + 1)%Z < 1)%Z by now rewrite Zle_Qle, Zlt_Qlt.
    assert (Qfloor y - Qceiling x + 1 = 0)%Z by lia.
    assert (Qfloor y = Qceiling x - 1)%Z by lia.
    all: try let H := fresh in assert (H : Qceiling x == x + 0.5) by lra.
    all: try let H := fresh in assert (H : Qfloor y == y - 0.5) by lra.
    assert (H' : (Qceiling x = Qfloor y + 1)%Z) by lia.
    rewrite H' in *.
    assert (Qfloor y + 1 == (Qfloor y + 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
    assert (y - 0.5 == x - 0.5) by lra.
    assert (x == y) by lra.
    replace (Z.even (Qfloor y + 1)%Z) with (Z.even (Z.succ (Qfloor y))) in * by (f_equal; lia).
    rewrite Z.even_succ in *.
    rewrite <- Z.negb_even in *.
    destruct Z.even; cbv [negb] in *; congruence. }
Qed.

Lemma Qle_floor_round (x : Q) : Qfloor x <= Qround x.
Proof. generalize (Qle_floor_ceiling x); cbv [Qround]; break_innermost_match; lra. Qed.
Lemma Qle_round_ceiling (x : Q) : Qround x <= Qceiling x.
Proof. generalize (Qle_floor_ceiling x); cbv [Qround]; break_innermost_match; lra. Qed.
Lemma Qround_comp x y : x == y -> Qround x = Qround y.
Proof.
  intro H.
  cbv [Qround]; rewrite (Qfloor_comp x y H), (Qceiling_comp x y H), !H.
  reflexivity.
Qed.
#[export] Instance Qround_comp_Proper : Proper (Qeq ==> eq) Qround.
Proof. repeat intro; apply Qround_comp; assumption. Qed.
Lemma Qceiling_lt' x : Qceiling x < x + 1.
Proof.
  generalize (Qceiling_lt x).
  assert (Qceiling x - 1 == (Qceiling x - 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
  lra.
Qed.
Lemma Qlt_floor' x : x - 1 < Qfloor x.
Proof.
  generalize (Qlt_floor x).
  assert (Qfloor x + 1 == (Qfloor x + 1)%Z) by (cbv [inject_Z Qplus Qnum Qden Qmult Qeq Qminus Qopp]; try lia).
  lra.
Qed.
Lemma Qround_lt (x : Q) : Qround x < x + 1.
Proof.
  generalize (Qceiling_lt' x) (Qle_floor_ceiling x); cbv [Qround]; break_innermost_match; try lra.
Qed.
Lemma Qlt_round (x : Q) : x - 1 < Qround x.
Proof.
  generalize (Qlt_floor' x) (Qle_floor_ceiling x); cbv [Qround]; break_innermost_match; try lra.
Qed.

Definition Qsplit_int_frac (q : Q) : Z * Q
  := let z := Qround q in
     (z, q - z).

Section exp.
  Context R (rI : R) (rmul : R -> R -> R) (rdiv : R -> R -> R) (radd : R -> R -> R).
  Local Notation state := (option R (* exp(x) *) * R (* fuel+1 as R *) * R (* x^fuel / fuel! as R *))%type
                            (only parsing).
  Section with_x.
    Context (x : R).
    Fixpoint exp_by_taylor_helper (fuel : nat) : state -> state
      := fun '(exp_x, succ_fuel, x_p_fuel_div_fact_fuel)
         => match fuel with
            | O => (exp_x, succ_fuel, x_p_fuel_div_fact_fuel)
            | S fuel
              => exp_by_taylor_helper
                   fuel
                   (match exp_x with
                    | Some exp_x => Some (radd exp_x x_p_fuel_div_fact_fuel)
                    | None => Some x_p_fuel_div_fact_fuel
                    end, radd rI succ_fuel, rdiv (rmul x_p_fuel_div_fact_fuel x) succ_fuel)
            end.
    (*      match fuel with
       | O => (rI, rI, rI)
       | S fuel
         => let '(exp_x, succ_fuel, x_p_fuel_div_fact_fuel) := @exp_by_taylor_helper fuel in
            (radd exp_x x_p_fuel_div_fact_fuel, radd rI succ_fuel, rdiv (rmul x_p_fuel_div_fact_fuel x) succ_fuel)
       end.*)

    Definition exp_by_taylor (fuel : nat) : R
      := let '(exp_x, _, _) := @exp_by_taylor_helper fuel (None, rI, rI) in
         match exp_x with
         | Some exp_x => exp_x
         | None => rI
         end.
  End with_x.

  Definition e : Q := 2.71828182846.
  Definition default_exp_precision : nat := 10.

  Definition exp
    (split_int_frac : R -> Z * R)
    (inject_Z : Z -> R)
    {expansion_terms : with_default "Taylor expansion terms" nat default_exp_precision}
    (x : R)
    : R
    := let '(z, x) := split_int_frac x in
       let exp_z := @pow_Z Q 1 Qmult Qdiv e z in
       rmul (rdiv (inject_Z (Qnum exp_z)) (inject_Z (Zpos (Qden exp_z))))
         (exp_by_taylor x expansion_terms).
End exp.

Definition Qexp {expansion_terms : with_default "Taylor expansion terms" nat default_exp_precision} (x : Q) : Q
  := Qred (@exp Q 1 Qmult Qdiv Qplus Qsplit_int_frac inject_Z expansion_terms x).

Definition Qlog2_floor (x : Q) : Z
  := (Z.log2 (Qnum x) - Z.log2 (Zpos (Qden x)))%Z.

Definition Qlog2_approx (x : Q) : Z
  := (Z.log2_round (Qnum x) - Z.log2_round (Zpos (Qden x)))%Z.

Lemma Qabs_alt q : Qabs q = if Qle_bool 0 q then q else -q.
Proof.
  destruct q as [n d]; cbv [Qle_bool Qabs Qnum Qden inject_Z Qopp]; break_innermost_match; f_equal.
  all: lia.
Qed.
(*
Definition Qlog2_approx (x : Q) : Z
  := (Z.log2_round (Qnum x) - Z.log2_round (Zpos (Qden x)))%Z.

Definition pow_Q R (rI : R) (rmul : R -> R -> R) (rdiv : R -> R -> R) (x : R) (p : Q) : R
  := let '(pz, pq) := Qsplit_int_frac p in
     let xpz := pow_Z x pz in
     let xpq := ??? in
     xpz * xpq.

(* XXX FIXME DO BETTER *)
HERE FIGURE OUT power of Q -> Q -> Q
Definition QpowerQ (b : Q)
*)
