From Coq Require Import ZArith NArith Uint63 List PArray Wellfounded Lia.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63.
Local Open Scope list_scope.
Set Implicit Arguments.
Import ListNotations.
Open Scope uint63_scope.

#[local] Coercion is_true : bool >-> Sortclass.
#[local] Coercion Z.of_N : N >-> Z.
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.
#[local] Coercion N.of_nat : nat >-> N.
Fixpoint fill_array_of_list {A} (ls : list A) (start : int) (arr : array A) {struct ls} : array A
  := match ls with
     | [] => arr
     | x :: xs
       => fill_array_of_list xs (start+1) arr.[ start <- x ]
     end.
Definition array_of_list {A} (ls : list A) {default : pointed A} : array A
  := fill_array_of_list ls 0 (PArray.make (List.length ls) default).

Definition map_default {A B} {default : pointed B} (f : A -> B) (xs : array A) : array B.
Proof.
  refine (let len := PArray.length xs in
          let init := PArray.make len default in
          if (len =? 0)
          then init
          else
            Fix
              (Acc_intro_generator Uint63.size Wf_Uint63.lt_wf)
              (fun _ => _)
              (fun i cont res
               => let res := res.[i <- f xs.[i]] in
                  if Sumbool.sumbool_of_bool (i =? 0)
                  then res
                  else cont (i-1) _ res)
              (len-1)
              init).
  { abstract (
        hnf; rewrite eqb_false_spec, ltb_spec, sub_spec, to_Z_1 in *;
        lazymatch goal with
        | [ H : ?x <> ?y |- _ ]
          => specialize (fun H' => H (@to_Z_inj _ _ H'));
             rewrite ?to_Z_0 in H;
             destruct (to_Z_bounded x)
        end;
        rewrite Z.mod_small by lia; lia
      ). }
Defined.

Definition map {A B} (f : A -> B) (xs : array A) : array B
  := map_default (default:=f (PArray.default xs)) f xs.
