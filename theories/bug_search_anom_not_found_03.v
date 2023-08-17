(* -*- mode: coq; coq-prog-args: ("-emacs" "-w" "+implicit-core-hint-db,+implicits-in-term,+non-reversible-notation,+deprecated-intros-until-0,+deprecated-focus,+unused-intro-pattern,+variable-collision,+unexpected-implicit-declaration,+omega-is-deprecated,+deprecated-instantiate-syntax,+non-recursive,+undeclared-scope,+deprecated-hint-rewrite-without-locality,+deprecated-hint-without-locality,+deprecated-instance-without-locality,+deprecated-typeclasses-transparency-without-locality,-ltac2-missing-notation-var,unsupported-attributes" "-w" "-deprecated-native-compiler-option,-native-compiler-disabled" "-native-compiler" "ondemand" "-R" "theories" "NeuralNetInterp" "-top" "NeuralNetInterp.MaxOfTwoNumbers.Model.Instances") -*- *)
(* File reduced by coq-bug-minimizer from original input, then from 220 lines to 39 lines, then from 52 lines to 522 lines, then from 524 lines to 229 lines, then from 699 lines to 216 lines *)
(* coqc version 8.19+alpha compiled with OCaml 4.14.1
   coqtop version JasonGross-X1:/home/jgross/Downloads/coq/coq-master/_build/default,master (61ee398ed32f9334dd664ea8ed2697178e6e3844)
   Expected coqc runtime on this file: 1.991 sec *)
Require Coq.Init.Ltac.
Require Coq.Floats.Floats.
Require Coq.Numbers.Cyclic.Int63.Sint63.
Require Coq.Numbers.Cyclic.Int63.Uint63.
Require Coq.QArith.QArith.
Require Coq.micromega.Lia.
Require Coq.Lists.List.
Require Coq.Array.PArray.
Require Coq.Classes.Morphisms.
Require Coq.Classes.RelationClasses.
Require Coq.Strings.String.
Require NeuralNetInterp.Util.Default.
Require Coq.Arith.Arith.
Require Coq.NArith.NArith.
Require Coq.ZArith.ZArith.
Require NeuralNetInterp.Util.Pointed.
Require Coq.Bool.Bool.
Require Coq.Wellfounded.Wellfounded.
Require Coq.ZArith.Wf_Z.
Require Coq.Arith.Wf_nat.
Require Coq.Setoids.Setoid.
Require NeuralNetInterp.Util.Notations.
Require NeuralNetInterp.Util.Monad.
Require NeuralNetInterp.Util.Arith.Classes.
Require Coq.Reals.Reals.
Require Coq.PArith.PArith.
Require Coq.QArith.Qabs.
Require Coq.QArith.Qround.
Require Coq.micromega.Lqa.
Require NeuralNetInterp.Util.Arith.ZArith.
Require NeuralNetInterp.Util.Tactics.Head.
Require NeuralNetInterp.Util.Tactics.BreakMatch.
Require NeuralNetInterp.Util.Tactics.DestructHyps.
Require NeuralNetInterp.Util.Tactics.DestructHead.
Require NeuralNetInterp.Util.Arith.QArith.
Require NeuralNetInterp.Util.Arith.FloatArith.Definitions.
Require NeuralNetInterp.Util.Arith.Reals.Definitions.
Require NeuralNetInterp.Util.Arith.Instances.
Require NeuralNetInterp.Util.Wf_Uint63.
Require NeuralNetInterp.Util.Classes.
Require NeuralNetInterp.Util.ErrorT.
Require Coq.Relations.Relation_Definitions.
Require NeuralNetInterp.Util.PolymorphicOption.
Require NeuralNetInterp.Util.Slice.
Require NeuralNetInterp.Util.Bool.
Require NeuralNetInterp.Util.PArray.
Require NeuralNetInterp.Util.PArray.Instances.
Require NeuralNetInterp.Util.Relations.Relation_Definitions.Hetero.
Require Coq.Program.Basics.
Require Coq.Unicode.Utf8.
Require NeuralNetInterp.Util.Program.Basics.Dependent.
Require NeuralNetInterp.Util.Relations.Relation_Definitions.Dependent.
Require Coq.Program.Tactics.
Require NeuralNetInterp.Util.Classes.Morphisms.Dependent.
Require Coq.Lists.SetoidList.
Require Coq.Relations.Relations.
Require Coq.Classes.RelationPairs.
Require Coq.Classes.Init.
Require NeuralNetInterp.Util.Classes.RelationClasses.Hetero.
Require NeuralNetInterp.Util.Classes.RelationPairs.Hetero.
Require NeuralNetInterp.Util.Wf_Uint63.Instances.
Require NeuralNetInterp.Util.List.
Require NeuralNetInterp.Util.SolveProperEqRel.
Require NeuralNetInterp.Util.Option.
Require NeuralNetInterp.Util.List.Instances.NthError.
Require NeuralNetInterp.Util.Classes.RelationClasses.Dependent.
Require NeuralNetInterp.TransformerLens.HookedTransformer.Instances.
Require NeuralNetInterp.MaxOfTwoNumbers.Parameters.
Import Coq.Floats.Floats.
Import Coq.QArith.QArith.
Import Coq.Lists.List.
Import NeuralNetInterp.Util.Default.
Import NeuralNetInterp.Util.Pointed.
Import NeuralNetInterp.Util.Arith.Classes.
Import NeuralNetInterp.Util.Arith.Instances.
Import NeuralNetInterp.Util.Option.
Import NeuralNetInterp.Torch.Tensor.
Import NeuralNetInterp.TransformerLens.HookedTransformer.
Import NeuralNetInterp.TransformerLens.HookedTransformer.Config.
Import Instances.Truncating.
#[local] Open Scope core_scope.

#[local] Coercion Vector.of_list : list >-> Vector.t.

Notation tensor_of_list ls := (tensor_of_list ls) (only parsing).

Module Model (cfg : Config).

  Module Export Embed.
    Section __.
    End __.
  End Embed.

  Module Export Unembed.
    Section __.
    End __.
  End Unembed.

  Module Export PosEmbed.
    Section __.
    End __.
  End PosEmbed.

 
  Module Export Attention.
    Section __.
    End __.
  End Attention.

  Module Export TransformerBlock.
    Section __.
    End __.
  End TransformerBlock.

  Module Export HookedTransformer.
    Section __.
      Context {r} {batch : Shape r} {pos}
        (s := (batch ::' pos)%shape)
        (resid_shape := (s ::' cfg.d_model)%shape)
        {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
        {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
        {sqrtA : has_sqrt A} {expA : has_exp A}
        {use_checkpoint : with_default "use_checkpoint" bool true}.
Let coerA' (x : float) : A. exact (coer x). Defined.
      #[local] Coercion coerA' : float >-> A.
Let coer_ln_tensor : cfg.ln_tensor float -> cfg.ln_tensor A. exact (match cfg.normalization_type as nt return Config.ln_tensor_gen _ nt float -> Config.ln_tensor_gen _ nt A with
             | Some LN
             | Datatypes.None
               => fun x => x
             end). Defined.
      Definition coer_blocks_params
        := List.map
             (fun '((W_Q, W_K, W_V, W_O,
                      b_Q, b_K, b_V, b_O,
                      ln1_w, ln1_b) : cfg.block_params_type float)
              => ((W_Q:tensor _ A), (W_K:tensor _ A), (W_V:tensor _ A), (W_O:tensor _ A),
                   (b_Q:tensor _ A), (b_K:tensor _ A), (b_V:tensor _ A), (b_O:tensor _ A),
                   coer_ln_tensor ln1_w, coer_ln_tensor ln1_b)).
Local Definition masked_attn_scores (n : nat) (tokens : tensor s IndexType)
        : option (tensor (batch ::' cfg.n_heads ::' pos ::' pos) A). exact (HookedTransformer.HookedTransformer.masked_attn_scores
             (A:=A) (n_ctx:=cfg.n_ctx) (normalization_type:=cfg.normalization_type)cfg.eps
             cfg.W_E cfg.W_pos
             (coer_blocks_params cfg.blocks_params)
             n tokens). Defined.
    End __.
  End HookedTransformer.
End Model.
Import ListNotations.

Module cfg <: Config.
  Include Parameters.cfg.
  Parameter W_E : tensor [d_vocab; d_model] float.
  Parameter W_pos : tensor [n_ctx; d_model] float.
  Parameter W_U : tensor [d_model; d_vocab_out] float.
  Parameter b_U : tensor [d_vocab_out] float.
  Notation ln_tensor A := (ln_tensor_gen d_model normalization_type A).
  Parameter ln_final_w : ln_tensor float.
  Parameter ln_final_b : ln_tensor float.
  Notation block_params_type A := (block_params_type_gen n_heads d_model d_head normalization_type A).
  Parameter block_params : block_params_type float.
  Definition blocks_params := [block_params].
End cfg.
  Include Model cfg.

  Section with_batch.
    Context {r} {batch : Shape r} {pos}
      (s := (batch ::' pos)%shape)
      (resid_shape := (s ::' cfg.d_model)%shape)
      {return_per_token : with_default "return_per_token" bool false}
      {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
      (defaultA : pointed A := @coer _ _ coerZ point)
      {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
      {ltbA : has_ltb A}
      {oppA : has_opp A} {sqrtA : has_sqrt A} {expA : has_exp A} {lnA : has_ln A}
      {use_checkpoint : with_default "use_checkpoint" bool true}.
Definition masked_attn_scores (tokens : tensor s IndexType)
        : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A.
exact (Option.invert_Some
           (HookedTransformer.masked_attn_scores (A:=A) 0 tokens)).
Defined.
  End with_batch.
Import NeuralNetInterp.Torch.Tensor.Instances.
Import NeuralNetInterp.TransformerLens.HookedTransformer.Instances.
Import Dependent.ProperNotations.

  #[export] Instance masked_attn_scores_Proper_dep {r batch pos}
    : Dependent.Proper
        ((Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.const eq ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> (Dependent.idR ==> Dependent.idR)
           ==> Dependent.const (fun _ _ => True)
           ==> Dependent.const Tensor.eqf
           ==> Tensor.eqfR)
        (@masked_attn_scores r batch pos).
  Proof.
    cbv [masked_attn_scores].
    pose proof (@HookedTransformer.HookedTransformer.masked_attn_scores_Proper_dep) as H.
    repeat intro.
    repeat (let v := open_constr:(_) in specialize (H v)).
    move H at bottom.
    revert H.
    lazymatch goal with
    | [ |- ?R _ _ ?R'' ?x ?y -> ?R' (invert_Some ?x' ?i) (invert_Some ?y' ?i) ]
      => unify x x'; unify y y'; unify R'' R'; set (x'':=x); set (y'':=y);
         intro H;
         refine (@invert_Some_Proper_dep _ _ (Tensor.eqfR R') x y H i)
    end.
    Unshelve.
    Search HookedTransformer.block_params_type_genR.

