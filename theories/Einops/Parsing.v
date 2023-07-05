(* From einops + gpt-4 *)
From Coq Require Import String List Bool Arith NArith.
From NeuralNetInterp.Util Require Import Ascii String ErrorT Default Option List.
From RecordUpdate Require Import RecordSet.
Import StringSyntax ListNotations.
Import RecordSetNotations.
Open Scope string_scope.
Open Scope list_scope.
Open Scope error_scope.

Inductive Axis :=
| AnonymousAxis: N -> Axis
| NamedAxis: string -> Axis.

Definition is_anonymous (a : Axis) : bool :=
  match a with
  | AnonymousAxis _ => true
  | _ => false
  end.

Record ParsedExpression := {
    has_ellipsis : bool;
    has_ellipsis_parenthesized : option bool;
    has_non_unitary_anonymous_axes : bool;
    identifiers : list string;
    composition : list (list string + string);
  }.
#[export] Instance etaParsedExpression : Settable _ := settable! Build_ParsedExpression <has_ellipsis; has_ellipsis_parenthesized; has_non_unitary_anonymous_axes; identifiers; composition>.
Definition _ellipsis := "…". (* NB, this is a single unicode symbol *)

(*
Definition mkParsedExpression
  (expression : string) {allow_underscore : with_default bool false} {allow_duplicates : with_default bool false}
  : ErrorT string ParsedExpression.
  refine (self <- Success
               {| has_ellipsis := false
               ; has_ellipsis_parenthesized := None
               ; identifiers := []
               (* that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition *)
               ; has_non_unitary_anonymous_axes := false
               (* composition keeps structure of composite axes, see how different corner cases are handled in tests *)
               ; composition := [] |};
          '(expression, self)
          <- if String.contains 0 "." expression
             then if negb (String.contains 0 "..." expression)
                  then Error "Expression may contain dots only inside ellipsis (...)"
                  else if negb (String.count expression "..." =? 1)%nat || negb (String.count expression "." =? 3)%nat
                       then Error "Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor "
                       else let expression := String.replace "..." _ellipsis expression in
                            let self := self <| has_ellipsis := true |> in
                            Success (expression, self)
             else Success (expression, self);
          let bracket_group : option (list string) := None in
          let add_axis_name {state : ParsedExpression * option (list string)} (x : string) : ErrorT string (ParsedExpression * option (list string))
            := let '(self, bracket_group) := state in
               if List.existsb (fun y => x =? y)%string self.(identifiers)
               then if negb (allow_underscore && (x =? "_")) && negb allow_duplicates
                    then Error ("Indexing expression contains duplicate dimension " ++ x)
                    else if x =? _ellipsis
                         then let self := self <| identifiers ::= cons _ellipsis |> in
                              let '(self, braket_group)
                                := match bracket_group with
                                   | None
                                     => ((self <| composition ::= snoc (inr _ellipsis) |>
                                               <| has_ellipsis_parenthesized := false |>)
                                          , bracket_group)
                                   | Some bracket_group
                                     => ((self <| has_ellipsis_parenthesized := false |>)
                                          , Some (snoc bracket_group _ellipsis))
                                   end
                                   else self
                                        if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    # handling the case of anonymous axis of length 1
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass  # no need to think about 1s inside parenthesis
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise EinopsError('Invalid axis identifier: {}\n{}'.format(x, reason))
                if is_number:
                    x = AnonymousAxis(x)
                self.identifiers.add(x)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([x])
                else:
                    bracket_group.append(x)

          _).


            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True

_).
     ; bool = False

(* This function is a placeholder for actual implementation. In Python code it checks if the string is a valid identifier. *)
Definition is_valid_identifier (name : string) : bool :=
  negb (orb (string_dec name "_") (string_dec name "...")).

Definition parse_expression (expression : string) : ParsedExpression :=
  (* Coq doesn't have a way to change strings in-place, and
     doesn't support regular expressions as Python does, so this is a workaround. *)
  let expression := if string_dec expression "..."
                    then String "." (String "." (String "." ""))
                    else expression in
  (* This function is a placeholder for the actual implementation that parses the string *)
  let identifiers := parse_string_to_identifiers expression in
  let has_ellipsis := is_in_list (NamedAxis "...") identifiers in
  let has_non_unitary_anonymous_axes := existsb is_anonymous identifiers in
  parsed_exp has_ellipsis has_non_unitary_anonymous_axes identifiers.

(* This function is a placeholder for the actual implementation that parses the string *)
Fixpoint parse_string_to_identifiers (s : string) : list Axis :=
  (* This function should be implemented in a way that parses a string into a list of Axis based on your specific requirements *)
  (* This is a dummy implementation that just creates a list with a single NamedAxis *)
  [NamedAxis s].

*)
