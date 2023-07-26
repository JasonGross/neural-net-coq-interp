From Ltac2 Require Import Ltac1 Init Bool Option.
From NeuralNetInterp.Util.Tactics2 Require Import Constr.

Ltac is_array x :=
  let err := idtac; fail 0 "Not a primitive array" in
  let f := ltac2:(x err
                  |- if Option.map_default is_array false (Ltac1.to_constr x)
                     then ()
                     else Ltac1.run err) in
  f x err.
