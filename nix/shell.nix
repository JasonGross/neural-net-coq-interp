{ pkgs, text-editor }:
let coq-packages = import ./coq.nix { inherit pkgs; };
in pkgs.mkShell {
  name = "neural-net-coq-interp-development";
  buildInputs = coq-packages ++ text-editor;
}
