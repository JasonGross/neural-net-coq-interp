Reserved Notation "A ;; B" (at level 100, right associativity, format "'[v' A ;; '/' B ']'").
Reserved Notation "A ;;; B" (at level 100, right associativity, format "'[v' A ;;; '/' B ']'").
Reserved Notation "'olet' x .. y <-- X ; Y"
         (at level 100, X at next level, x binder, y binder, right associativity, format "'[v' 'olet'  x  ..  y  <--  X ; '/' Y ']'").
Reserved Notation "A <-- X ; B" (at level 100, X at next level, right associativity, format "'[v' A  <--  X ; '/' B ']'").
Reserved Notation "' A <-- X ; B" (at level 100, X at next level, A strict pattern, right associativity, format "'[v' ' A  <--  X ; '/' B ']'").
Reserved Notation "A <-- X ;; B" (at level 100, X at next level, right associativity, format "'[v' A  <--  X ;; '/' B ']'").
Reserved Notation "' A <-- X ;; B" (at level 100, X at next level, A strict pattern, right associativity, format "'[v' ' A  <--  X ;; '/' B ']'").

Reserved Infix "::'" (at level 59, left associativity).
Reserved Infix "++'" (at level 59, left associativity).
Reserved Infix "+'" (at level 48, left associativity).
Reserved Infix "-'" (at level 50, left associativity).
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
Reserved Infix "//" (at level 40, left associativity).
Reserved Notation "x ¹" (at level 1, no associativity).
Reserved Notation "x ²" (at level 1, no associativity).
Reserved Notation "x ³" (at level 1, no associativity).
Reserved Notation "x ⁴" (at level 1, no associativity).
Reserved Notation "x ⁵" (at level 1, no associativity).
Reserved Notation "x ⁶" (at level 1, no associativity).
Reserved Notation "x ⁷" (at level 1, no associativity).
Reserved Notation "x ⁸" (at level 1, no associativity).
Reserved Notation "x ⁹" (at level 1, no associativity).
Reserved Notation "√ x" (at level 5, right associativity, format "√ x").

Reserved Notation "c >>= f" (at level 50, left associativity).
Reserved Notation "f =<< c" (at level 51, right associativity).

Reserved Notation "\sum_ i F"
  (at level 41, F at level 41, i at level 0,
    right associativity,
    format "'[' \sum_ i '/ ' F ']'").
Reserved Notation "∑_ i F"
  (at level 41, F at level 41, i at level 0,
    right associativity,
    format "'[' ∑_ i '/ ' F ']'").
Reserved Notation "\sum_ ( i <- r ) F"
  (at level 41, F at level 41, i, r at level 50,
    format "'[' \sum_ ( i <- r ) '/ ' F ']'").
Reserved Notation "∑_ ( i <- r ) F"
  (at level 41, F at level 41, i, r at level 50,
    format "'[' ∑_ ( i <- r ) '/ ' F ']'").
Reserved Notation "\prod_ i F"
  (at level 36, F at level 36, i at level 0,
    format "'[' \prod_ i '/ ' F ']'").
Reserved Notation "∏_ i F"
  (at level 36, F at level 36, i at level 0,
    format "'[' ∏_ i '/ ' F ']'").
Reserved Notation "\prod_ ( i <- r ) F"
  (at level 36, F at level 36, i, r at level 50,
    format "'[' \prod_ ( i <- r ) '/ ' F ']'").
Reserved Notation "∏_ ( i <- r ) F"
  (at level 36, F at level 36, i, r at level 50,
    format "'[' ∏_ ( i <- r ) '/ ' F ']'").

Reserved Notation "\sum_ ( m <= i < n ) F"
  (at level 41, F at level 41, i, m, n at level 50,
    format "'[' \sum_ ( m <= i < n ) '/ ' F ']'").
Reserved Notation "\sum_ ( m ≤ i < n ) F"
  (at level 41, F at level 41, i, m, n at level 50,
    format "'[' \sum_ ( m ≤ i < n ) '/ ' F ']'").
Reserved Notation "∑_ ( m <= i < n ) F"
  (at level 41, F at level 41, i, m, n at level 50,
    format "'[' ∑_ ( m <= i < n ) '/ ' F ']'").
Reserved Notation "∑_ ( m ≤ i < n ) F"
  (at level 41, F at level 41, i, m, n at level 50,
    format "'[' ∑_ ( m ≤ i < n ) '/ ' F ']'").
Reserved Notation "\prod_ ( m <= i < n ) F"
  (at level 36, F at level 36, i, m, n at level 50,
    format "'[' \prod_ ( m <= i < n ) '/ ' F ']'").
Reserved Notation "\prod_ ( m ≤ i < n ) F"
  (at level 36, F at level 36, i, m, n at level 50,
    format "'[' \prod_ ( m ≤ i < n ) '/ ' F ']'").
Reserved Notation "∏_ ( m <= i < n ) F"
  (at level 36, F at level 36, i, m, n at level 50,
    format "'[' ∏_ ( m <= i < n ) '/ ' F ']'").
Reserved Notation "∏_ ( m ≤ i < n ) F"
  (at level 36, F at level 36, i, m, n at level 50,
    format "'[' ∏_ ( m ≤ i < n ) '/ ' F ']'").
