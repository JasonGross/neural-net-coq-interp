# neural-net-coq-interp

[![CI (Coq)](https://github.com/JasonGross/neural-net-coq-interp/actions/workflows/coq.yml/badge.svg?branch=main)](https://github.com/JasonGross/neural-net-coq-interp/actions/workflows/coq.yml)
[![CI (Python)](https://github.com/JasonGross/neural-net-coq-interp/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/JasonGross/neural-net-coq-interp/actions/workflows/python.yml)

Some experiments with doing NN interpretability in Coq

[Associated Colab notebook](https://colab.research.google.com/drive/1WdvPyO-bB6l-iWq8SYjiovHp5R3834wN?usp=sharing)

## Dev (`coq`) 

### `opam` / `coq_makefile`

Presumably normal (TODO: write instructions)

### `dune`

either run `patch` or tell your editor to use `_CoqProjectDune` instead of `_CoqProject`

``` sh
patch -ruN _CoqProject < _CoqProject.patch 
dune build
```

### `nix`

This can build `emacs` or `vscode` to choice of version with dependencies that does not interact with your machine's other installations of `emacs` or `vscode`. It'll do coq `8.17`, it'll take me 30-60 minutes to make a flag/option for `8.18` in the CLI, just ask. 

I'm less confident that the `vscode` build actually works, I last really test drove the nix code for that forever ago. 

You'll need to [`nixpkgs.config.allowUnfree = true;`](https://nixos.wiki/wiki/Unfree_Software) (or whatever) for `VST`.  

``` sh
# direnv, nix-direnv is the way I do emacs. Should be able to skip for vscode? 
echo "use flake .#coq-no-ui" > .envrc
# OPTIONAL, speedup sometimes once my CI server is running in new apartment 
cachix use quinn-dougherty
direnv allow
# enable flakes https://nixos.wiki/wiki/Flakes
nix develop .#emacs --command "emacs"  # downloads coq and libraries, just builds emacs
nix develop .#codium --command "codium"  # downloads coq and libraries, just builds codium
nix develop .#coq # downloads coq and libraries, builds both emacs and codium.
nix develop .#coq-no-ui # downloads coq and libraries with no text editor builds
```

If you really like the new `coq-lsp` workflow I'd have to think a lot about how to make a `nix` version competitive in terms of tight feedback cycle. I manually type `dune build` every time I need to switch to a downstream file, right now. 

## Dev (`python`)

### `pip`

See `training/requirements.txt`

### `conda` 

See `training/environment.yml`

### `nix`

TODO: `mach-nix`
