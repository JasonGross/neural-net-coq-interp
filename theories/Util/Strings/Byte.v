Require Import Coq.NArith.NArith.
Require Import Coq.Strings.Byte.

Local Open Scope bool_scope.
Local Open Scope N_scope.
Local Open Scope byte_scope.

(** Special characters *)

Example Null := "000".
Example Backspace := "008".
Example Tab := "009".
Example LF := "010".
Example NewPage := "012".
Example CR := "013".
Example Escape := "027".
Example NewLine := "010".

Local Coercion to_N : byte >-> N.

Definition is_upper (ch : byte) : bool
  := ("A" <=? ch) && (ch <=? "Z").
Definition is_lower (ch : byte) : bool
  := ("a" <=? ch) && (ch <=? "z").

Definition of_N_always (ch : N) : match of_N ch with Some _ => byte | _ => unit end
  := match of_N ch as n return match n with Some _ => _ | _ => _ end with Some ch => ch | _ => tt end.

Definition to_lower (ch : byte) : byte
  := if ("A" <=? ch) && (ch <=? "Z")
     then match of_N ("a" + ch - "A") with Some v => v | None => ch end
     else ch.
Definition to_upper (ch : byte) : byte
  := if ("a" <=? ch) && (ch <=? "z")
     then match of_N ("A" + ch - "a") with Some v => v | None => ch end
     else ch.

Definition is_whitespace (x : byte) : bool
  := ((x =? " ") || (x =? NewPage) || (x =? LF) || (x =? CR) || (x =? Tab))%byte%bool.
