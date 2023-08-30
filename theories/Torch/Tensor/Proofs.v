From Coq Require Import ZArith NArith Uint63 String.
From NeuralNetInterp.Util Require Import Default Pointed.
From NeuralNetInterp.Util.Arith Require Import Classes Instances.
From NeuralNetInterp.Torch Require Import Tensor.

#[local] Open Scope core_scope.

Lemma raw_get_arange {start stop step i}
  : raw_get (@arange start stop step) i = start + RawIndex.tl i * step.
Proof. reflexivity. Qed.

Lemma get_arange {start stop step i}
  : get (@arange start stop step) i = start + (Index.tl i mod (1 + (stop - start - 1) / step))%uint63 * step.
Proof. cbv [get]; rewrite raw_get_arange; reflexivity. Qed.
(*
Lemma raw_get_cartesian_exp_lax {s A defaultA t n i}
  : exists j, raw_get (@cartesian_exp s A defaultA t n) i = raw_get t j.
Proof.
  cbv [cartesian_exp raw_get].
  set (n' := Uint63.coer_int_N' n); clearbody n'; clear n; rename n' into n.
  induction n as [|n IH] using N.peano_ind.
  { destruct i as [[[] i] j].
    cbv [N.to_nat Shape.repeat cartesian_nprod Shape.Tuple.init reshape_all raw_get Shape.Tuple.init' RawIndex.unreshape RawIndex.snoc RawIndex.item RawIndex.tl RawIndex.unreshape' ntupleify ntupleify' Shape.nil Shape.Tuple.nth_default Shape.Tuple.to_list Shape.Tuple.to_list' List.nth_default].

  Search N.to
    Set Printing All.

  rename n into XXXX.
  clear n.
                                              ;l
Definition cartesian_exp {s A} {defaultA : pointed A} (t : tensor [s] A) (n : ShapeType) : tensor [(s^(n:N))%core; n] A
  := @cartesian_nprod n (Shape.repeat s n) A _ (Shape.Tuple.init (fun _ => t)).


Lemma raw_get_ntupleify' {r s A B2 B2 f
Fixpoint ntupleify' {r} : forall {s : Shape r} {A B1 B2},
    (B1 -> B2)
    -> Shape.fold_map (fun s => tensor [s] A) (fun x y => (y * x)%type) B1 s
    -> tensor s (Shape.fold_map (fun _ => A) (fun x y => (y * x)%type) B2 s)
  := match r with
     | O => fun s A B1 B2 f ts i => f ts
     | S r
       => fun s A B1 B2 f ts i
          => let f := (fun tab1 => let '(ta, b1) := (fst tab1, snd tab1) in (raw_get ta [RawIndex.tl i], f b1)) in
             @ntupleify' r (Shape.hd s) A (tensor [Shape.tl s] A * B1)%type (A * B2)%type f ts (RawIndex.hd i)
     end.


(* TODO: nary *)
Definition tupleify {s1 s2 A B} (t1 : tensor [s1] A) (t2 : tensor [s2] B) : tensor [s1; s2] (A * B)
  := fun '((tt, a), b) => (raw_get t1 [a], raw_get t2 [b]).
Definition cartesian_prod {s1 s2 A} (t1 : tensor [s1] A) (t2 : tensor [s2] A) : tensor [s1 * s2; 2] A
  := fun '((tt, idx), tuple_idx)
     => let '(a, b) := raw_get (reshape_all (tupleify t1 t2)) [idx] in
        nth_default a [a; b] (Z.to_nat (Uint63.to_Z (tuple_idx mod 2))).
Fixpoint ntupleify' {r} : forall {s : Shape r} {A B1 B2},
    (B1 -> B2)
    -> Shape.fold_map (fun s => tensor [s] A) (fun x y => (y * x)%type) B1 s
    -> tensor s (Shape.fold_map (fun _ => A) (fun x y => (y * x)%type) B2 s)
  := match r with
     | O => fun s A B1 B2 f ts i => f ts
     | S r
       => fun s A B1 B2 f ts i
          => let f := (fun tab1 => let '(ta, b1) := (fst tab1, snd tab1) in (raw_get ta [RawIndex.tl i], f b1)) in
             @ntupleify' r (Shape.hd s) A (tensor [Shape.tl s] A * B1)%type (A * B2)%type f ts (RawIndex.hd i)
     end.
Definition ntupleify {r} {s : Shape r} {A} (ts : Shape.tuple (fun s => tensor [s] A) s) : tensor s (Shape.tuple (fun _ => A) s)
  := ntupleify' (fun _tt => tt) ts.
Definition cartesian_nprod {r} {s : Shape r} {A} {defaultA : pointed A} (ts : Shape.tuple (fun s => tensor [s] A) s) : tensor [Shape.fold_map id mul 1 s; Uint63.of_Z r] A
  := fun '((tt, idx), tuple_idx)
     => let ts := raw_get (reshape_all (ntupleify ts)) [idx] in
        Shape.Tuple.nth_default point ts (Z.to_nat (Uint63.to_Z (tuple_idx mod Uint63.of_Z r))).
Definition cartesian_exp {s A} {defaultA : pointed A} (t : tensor [s] A) (n : ShapeType) : tensor [(s^(n:N))%core; n] A
  := @cartesian_nprod n (Shape.repeat s n) A _ (Shape.Tuple.init (fun _ => t)).
*)
