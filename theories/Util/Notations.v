Reserved Notation "A ;; B" (at level 100, right associativity, format "'[v' A ;; '/' B ']'").
Reserved Notation "A ;;; B" (at level 100, right associativity, format "'[v' A ;;; '/' B ']'").
Reserved Notation "'olet' x .. y <-- X ; Y"
         (at level 100, X at next level, x binder, y binder, right associativity, format "'[v' 'olet'  x  ..  y  <--  X ; '/' Y ']'").
Reserved Notation "A <-- X ; B" (at level 100, X at next level, right associativity, format "'[v' A  <--  X ; '/' B ']'").
Reserved Notation "' A <-- X ; B" (at level 100, X at next level, A strict pattern, right associativity, format "'[v' ' A  <--  X ; '/' B ']'").
Reserved Notation "A <-- X ;; B" (at level 100, X at next level, right associativity, format "'[v' A  <--  X ;; '/' B ']'").
Reserved Notation "' A <-- X ;; B" (at level 100, X at next level, A strict pattern, right associativity, format "'[v' ' A  <--  X ;; '/' B ']'").

Reserved Infix "<?" (at level 70, no associativity).
Reserved Infix "<=?" (at level 70, no associativity).
Reserved Infix "≤?" (at level 70, no associativity).
Reserved Infix ">?" (at level 70, no associativity).
Reserved Infix ">=?" (at level 70, no associativity).
Reserved Infix "≤?" (at level 70, no associativity).
Reserved Infix "≥?" (at level 70, no associativity).
Reserved Infix "=?" (at level 70, no associativity).
Reserved Infix "!=" (at level 70, no associativity).
Reserved Infix "<>?" (at level 70, no associativity).
Reserved Infix "≠?" (at level 70, no associativity).
Reserved Infix "mod" (at level 40, no associativity).

Reserved Notation "c >>= f" (at level 50, left associativity).
Reserved Notation "f =<< c" (at level 51, right associativity).
