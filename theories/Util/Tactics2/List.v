From Ltac2 Require Import Ltac2 List.

Ltac2 diff (equal : 'a -> 'a -> bool) (ls1 : 'a list) (ls2 : 'a list) : 'a list
  := let (overlap, diff) := List.partition (fun a => List.mem equal a ls2) ls1 in
     diff.

(* Slow, but order-preserving *)
Ltac2 uniq (equal : 'a -> 'a -> bool) (ls : 'a list) : 'a list
  := let rec aux (xs : 'a list) (acc : 'a list) : 'a list
       := match xs with
          | [] => acc
          | x :: xs
            => if List.mem equal x acc
               then aux xs acc
               else aux xs (x :: acc)
          end in
     List.rev (aux ls []).
