From Coq Require Import QArith ZArith.
From NeuralNetInterp.Util Require Import Default.
From NeuralNetInterp.Util.Arith Require Import ZArith.
Set Implicit Arguments.

(* XXX FIXME *)
Definition Qsqrt (v : Q) : Q := Qred (Qmake (Z.sqrt (Qnum v * Zpos (Qden v))) (Qden v)).

Definition pow_Z R (rI : R) (rmul : R -> R -> R) (rdiv : R -> R -> R) (x : R) (p : Z) : R
  := match p with
     | Zneg p => rdiv rI (@pow_N R rI rmul x (Npos p))
     | 0%Z => rI
     | Zpos p => @pow_N R rI rmul x (Npos p)
     end.

#[local] Coercion inject_Z : Z >-> Q.

Definition Qfloor (q : Q) : Z := (Qnum q / (Zpos (Qden q)))%Z.
Definition Qceil (q : Q) : Z := (-((-Qnum q) / (Zpos (Qden q))))%Z.
Definition Qabs (q : Q) : Q := Qmake (Z.abs (Qnum q)) (Qden q).
Definition Qround (q : Q) : Z
  := let '(a, b) := (Qfloor q, Qceil q) in
     let '(aerr, berr) := (Qabs (a - q), Qabs (b - q)) in
     if Qle_bool aerr berr
     then a
     else b.

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

Definition Qlog2_approx (x : Q) : Z
  := (Z.log2_round (Qnum x) - Z.log2_round (Zpos (Qden x)))%Z.

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
