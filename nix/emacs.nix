{ pkgs, nix-doom-emacs }:
nix-doom-emacs.packages.${pkgs.system}.default.override {
  doomPrivateDir = ./doom.d;
}
