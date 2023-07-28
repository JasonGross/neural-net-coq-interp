From Coq Require Import Floats Arith NArith ZArith QArith PArray Uint63 List.
Import ListNotations.
Class pointed T := point : T.
#[export] Instance default_Q : pointed Q := 0 : Q.
#[export] Instance default_nat : pointed nat := 0%nat.
#[export] Instance default_N : pointed N := 0%N.
#[export] Instance default_Z : pointed Z := 0%Z.
#[export] Instance default_int : pointed int := 0%int63.
#[export] Instance default_list {A} : pointed (list A) := [].
#[export] Instance default_array {A} {a : pointed A} : pointed (array A) := PArray.make 0 a.
#[export] Instance default_float : pointed float := 0%float.
#[export] Instance default_option {A} : pointed (option A) := None.
