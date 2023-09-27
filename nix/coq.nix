{ pkgs }:
with pkgs.coqPackages; [
  pkgs.ocaml
  pkgs.dune_3
  coq
  coq-lsp
  coq-record-update
  flocq
  interval
  mathcomp
  mathcomp-zify
  mathcomp-algebra-tactics
  mathcomp-analysis
]
