{ pkgs }:
with pkgs.coqPackages_8_17; [
  pkgs.ocaml
  pkgs.dune_3
  coq
  coq-lsp
  coq-record-update
  flocq
  interval
  vcfloat
  mathcomp
  mathcomp-zify
  mathcomp-algebra-tactics
  mathcomp-analysis
]
