{
  description = "Work in progress formalization of mechanistic interpretability arguments on tiny neural nets";

  inputs = {
    nixpkgs-stable.url = "nixpkgs/nixos-23.05";
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    nix-doom-emacs = {
      url = "github:nix-community/nix-doom-emacs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs-stable, nixpkgs, flake-parts, nix-doom-emacs }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        ./nix
      ];
      systems = [ "x86_64-linux" ];
        # [ "aarch64-linux" "aarch64-darwin" "x86_64-darwin" "x86_64-linux" ];
    };
}
