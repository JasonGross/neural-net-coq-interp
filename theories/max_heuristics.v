From Coq Require Import String Floats Uint63 NArith QArith ZArith Lia List.
Local Open Scope float_scope.
Local Open Scope string_scope.
Local Open Scope list_scope.
Local Open Scope uint63_scope.
Import ListNotations.

#[local] Set Warnings Append "-inexact-float".

Module heuristics.
  Definition incorrect_results : list (list int)
    := [[ 3;  2];
        [ 6;  5];
        [ 7;  8];
        [13; 12];
        [15; 14];
        [17; 16];
        [22; 23];
        [22; 26];
        [24; 26];
        [25; 26];
        [30; 31];
        [34; 37];
        [36; 37];
        [36; 38];
        [41; 42];
        [41; 43];
        [42; 41];
        [48; 49];
        [50; 51]].
End heuristics.
