From Ltac2 Require Import Ltac1 Init Bool Option.
From NeuralNetInterp.Util.Tactics2 Require Import Constr.

Ltac is_uint63 x :=
  let err := idtac; fail 0 "Not a primitive uint63" in
  let f := ltac2:(x err
                  |- if Option.map_default is_uint63 false (Ltac1.to_constr x)
                     then ()
                     else Ltac1.run err) in
  f x err.
