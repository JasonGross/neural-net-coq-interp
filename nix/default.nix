{ self, inputs, ... }: {
  perSystem = { config, self', inputs', pkgs, system, ... }:
    let
      doom-emacs = with inputs;
        import ./emacs.nix { inherit pkgs nix-doom-emacs; };
      vscodium = import ./codium.nix { inherit pkgs; };
      shell = { text-editor ? [ ] }:
        import ./shell.nix { inherit pkgs text-editor; };
    in {
      _module.args.pkgs = import inputs.nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      devShells = {
        coq-no-ui = shell { };
        emacs = shell { text-editor = [ doom-emacs ]; };
        codium = shell { text-editor = [ vscodium ]; };
        coq = shell { text-editor = [ doom-emacs vscodium ]; };
      };
      packages.coq-neural-net-interp = pkgs.stdenv.mkDerivation {
        name = "coq-neural-net-interp-compile";
        buildInputs = (shell { }).buildInputs;
        src = ./..;
        buildPhase = ''
          patch -ruN _CoqProject < _CoqProjectDune.patch
          dune build
        '';
        installPhase = ''
          mkdir -p $out
          cp -r _build/* $out
        '';
      };
    };
}
