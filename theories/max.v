From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp Require Import max_parameters.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
Set Universe Polymorphism.
Unset Universe Minimization ToSet.
Set Polymorphic Inductive Cumulativity.
Import ListNotations.
Local Open Scope raw_tensor_scope.

(* Based on https://colab.research.google.com/drive/1N4iPEyBVuctveCA0Zre92SpfgH6nmHXY#scrollTo=Q1h45HnKi-43, Taking the minimum or maximum of two ints *)

(** Coq infra *)
#[local] Coercion Uint63.of_Z : Z >-> Uint63.int.

(** Hyperparameters *)
Definition N_LAYERS : nat := 1.
Definition N_HEADS : nat := 1.
Definition D_MODEL : nat := 32.
Definition D_HEAD : nat := 32.
(*Definition D_MLP = None*)

Definition D_VOCAB : nat := 64.

Notation tensor_of_list ls := (Tensor.PArray.abstract (Tensor.PArray.concretize (Tensor.of_list ls))) (only parsing).
Definition W_E : tensor _ _ := Eval cbv in tensor_of_list max_parameters.W_E.
Definition W_pos : tensor _ _ := Eval cbv in tensor_of_list max_parameters.W_pos.
Definition L0_attn_W_Q : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_Q.
Definition L0_attn_W_K : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_K.
Definition L0_attn_W_V : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_V.
Definition L0_attn_W_O : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_W_O.
Definition L0_attn_b_Q : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_Q.
Definition L0_attn_b_K : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_K.
Definition L0_attn_b_V : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_V.
Definition L0_attn_b_O : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_attn_b_O.
Definition L0_ln1_b : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_ln1_b.
Definition L0_ln1_w : tensor _ _ := Eval cbv in tensor_of_list max_parameters.L0_ln1_w.
Definition ln_final_b : tensor _ _ := Eval cbv in tensor_of_list max_parameters.ln_final_b.
Definition ln_final_w : tensor _ _ := Eval cbv in tensor_of_list max_parameters.ln_final_w.
Definition W_U : tensor _ _ := Eval cbv in tensor_of_list max_parameters.W_U.
Definition b_U : tensor _ _ := Eval cbv in tensor_of_list max_parameters.b_U.

#[local] Notation FLOAT := float (only parsing). (* or Q *)

Definition embed {r} {s : Shape r} (tokens : tensor IndexType s) : tensor FLOAT (s ::' Shape.tl (shape_of W_E))
  := (W_E.[tokens, :])%fancy_raw_tensor.

Definition pos_embed {r} {s : Shape (S r)} (tokens : tensor int s)
  (tokens_length := Shape.tl s) (* s[-1] *)
  (batch := Shape.droplastn 1 s) (* s[:-1] *)
  (d_model := Shape.tl (shape_of W_pos)) (* s[-1] *)
  : tensor FLOAT (batch ++' [tokens_length] ::' d_model)
  := repeat (W_pos.[:tokens_length, :]) batch.

Section layernorm.
  Context {r A} {s : Shape r} {d_model}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A} {sqrtA : has_sqrt A} {zeroA : has_zero A} {coerZ : has_coer Z A} {default : pointed A}
    (eps : A) (w b : tensor A [d_model]).

  Definition layernorm_linpart (x : tensor A (s ::' d_model))
    : tensor A (s ::' d_model)
    := (x - reduce_axis_m1 (keepdim:=true) mean x)%core.

  Definition layernorm_scale (x : tensor A (s ::' d_model))
    : tensor A (s ::' 1)
    := (√(reduce_axis_m1 (keepdim:=true) mean (x ²) + broadcast' eps))%core.

  Definition layernorm_rescale (x : tensor A (s ::' d_model))
                               (scale : tensor A (s ::' 1))
    : tensor A (s ::' d_model)
    := (x / scale)%core.

  Definition layernorm_postrescale (x : tensor A (s ::' d_model))
    : tensor A (s ::' d_model)
    := (x * broadcast w + broadcast b)%core.

  Definition layernorm (x : tensor A (s ::' d_model))
    : tensor A (s ::' d_model)
    := let x := PArray.checkpoint (layernorm_linpart x) in
       let scale := layernorm_scale x in
       let x := layernorm_rescale x scale in
       PArray.checkpoint (layernorm_postrescale x).
End layernorm.

Section ln.
  Context {r} {s : Shape r}
    (d_model := Uint63.of_Z cfg.d_model)
    (eps := cfg.eps).

  Section ln1.
    Context (w := L0_ln1_w) (b := L0_ln1_b).

    Definition ln1_linpart (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_linpart x.
    Definition ln1_scale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln1_rescale (x : tensor FLOAT (s ::' d_model)) (scale : tensor FLOAT (s ::' 1)) : tensor FLOAT (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln1_postrescale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln1 (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm eps w b x.
  End ln1.

  Section ln_final.
    Context (w := ln_final_w) (b := ln_final_b).

    Definition ln_final_linpart (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_linpart x.
    Definition ln_final_scale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_scale eps x.
    Definition ln_final_rescale (x : tensor FLOAT (s ::' d_model)) (scale : tensor FLOAT (s ::' 1)) : tensor FLOAT (s ::' d_model)
      := layernorm_rescale x scale.
    Definition ln_final_postrescale (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm_postrescale w b x.
    Definition ln_final (x : tensor FLOAT (s ::' d_model)) : tensor FLOAT (s ::' d_model)
      := layernorm eps w b x.
  End ln_final.
End ln.

Section Attention.
  Context {A r} {batch : Shape r}
    {sqrtA : has_sqrt A} {coerZ : has_coer Z A} {addA : has_add A} {zeroA : has_zero A} {mulA : has_mul A} {divA : has_div A} {expA : has_exp A} {defaultA : pointed A}
    {pos n_heads d_model d_head} {n_ctx:N}
    {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
    (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
    (b_Q b_K b_V : tensor A [n_heads; d_head])
    (b_O : tensor A [d_model])
    (IGNORE : A)
    (n_ctx' : int := Uint63.of_Z n_ctx)
    (attn_scale : A := √(coer (Uint63.to_Z d_head)))
    (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape)
    (query_input key_input value_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
    (mask : tensor bool [n_ctx'; n_ctx'] := to_bool (tril (A:=bool) (ones [n_ctx'; n_ctx']))).

  (*         if self.cfg.use_split_qkv_input:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        q = self.hook_q(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]*)
  Definition einsum_input
    (input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model)))
    (W : tensor A [n_heads; d_model; d_head])
    : tensor A ((batch ::' pos) ++' [n_heads; d_head])
    := Tensor.map'
         (if use_split_qkv_input return tensor A (maybe_n_heads use_split_qkv_input ::' d_model) -> tensor A [n_heads; d_head]
          then fun input => weaksauce_einsum {{{ {{ head_index d_model,
                                        head_index d_model d_head
                                        -> head_index d_head }}
                                    , input
                                    , W }}}
          else fun input => weaksauce_einsum {{{ {{ d_model,
                                        head_index d_model d_head
                                        -> head_index d_head }}
                                    , input
                                    , W }}})
         input.

  Definition q : tensor A (batch ++' [pos; n_heads; d_head])
    := PArray.checkpoint (einsum_input query_input W_Q + broadcast b_Q)%core.
  Definition k : tensor A (batch ++' [pos; n_heads; d_head])
    := PArray.checkpoint (einsum_input key_input W_K + broadcast b_K)%core.
  Definition v : tensor A (batch ++' [pos; n_heads; d_head])
    := PArray.checkpoint (einsum_input value_input W_V + broadcast b_V)%core.

  Definition attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
    := (let qk : tensor A (batch ++' [n_heads; pos; pos])
          := Tensor.map2'
               (fun q k : tensor A [pos; n_heads; d_head]
                => weaksauce_einsum
                     {{{ {{ query_pos head_index d_head ,
                               key_pos head_index d_head
                               -> head_index query_pos key_pos }}
                           , q
                           , k }}}
                  : tensor A [n_heads; pos; pos])
               q
               k in
        PArray.checkpoint (qk / broadcast' attn_scale))%core.

  Definition apply_causal_mask (attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos))
    : tensor A (batch ::' n_heads ::' pos ::' pos)
    := Tensor.map'
         (fun attn_scores : tensor A [pos; pos]
          => Tensor.where_ mask.[:pos,:pos] attn_scores (broadcast' IGNORE))
         attn_scores.

  Definition masked_attn_scores : tensor A (batch ::' n_heads ::' pos ::' pos)
    := apply_causal_mask attn_scores.

  Definition pattern : tensor A (batch ::' n_heads ::' pos ::' pos)
    := PArray.checkpoint (softmax_dim_m1 masked_attn_scores).

  Definition z : tensor A (batch ::' pos ::' n_heads ::' d_head)
    := PArray.checkpoint
         (Tensor.map2'
            (fun (v : tensor A [pos; n_heads; d_head])
                 (pattern : tensor A [n_heads; pos; pos])
             => weaksauce_einsum {{{ {{  key_pos head_index d_head,
                            head_index query_pos key_pos ->
                            query_pos head_index d_head }}
                        , v
                        , pattern }}}
               : tensor A [pos; n_heads; d_head])
            v
            pattern).

  Definition attn_out : tensor A (batch ::' pos ::' d_model)
    := (let out
          := Tensor.map'
               (fun z : tensor A [pos; n_heads; d_head]
                => weaksauce_einsum {{{ {{ pos head_index d_head,
                               head_index d_head d_model ->
                               pos d_model }}
                           , z
                           , W_O }}}
                  : tensor A [pos; d_model])
               z in
        PArray.checkpoint (out + broadcast b_O))%core.
End Attention.

Section Attention0.
  Context {r} {batch : Shape r}
    (IGNORE := -1e5)
    (query_input key_input value_input : tensor FLOAT (batch ++' [cfg.n_ctx; cfg.d_model])).

  Definition L0_attn_out : tensor FLOAT (batch ::' cfg.n_ctx ::' cfg.d_model)
    := attn_out
         (n_ctx:=cfg.n_ctx)
         L0_attn_W_Q L0_attn_W_K L0_attn_W_V L0_attn_W_O
         L0_attn_b_Q L0_attn_b_K L0_attn_b_V L0_attn_b_O
         IGNORE
         query_input key_input value_input.
End Attention0.

Section TransformerBlock.
  Context {A r} {batch : Shape r}
    {zeroA : has_zero A} {coerZ : has_coer Z A}
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
    {sqrtA : has_sqrt A} {expA : has_exp A}
    {default : pointed A}
    {pos n_heads d_model d_head} {n_ctx:N}
    {use_split_qkv_input : with_default "use_split_qkv_input" bool false}
    (W_Q W_K W_V W_O : tensor A [n_heads; d_model; d_head])
    (b_Q b_K b_V : tensor A [n_heads; d_head])
    (b_O : tensor A [d_model])
    (IGNORE : A)
    (eps : A)
    (ln1_w ln1_b ln2_w ln2_b : tensor A [d_model])
    (resid_pre : tensor A ((batch ::' pos) ++' [d_model]))
    (maybe_n_heads := fun b : bool => (if b return Shape (if b then _ else _) then [n_heads] else [])%shape).

  Definition add_head_dimension
    (resid_pre : tensor A ((batch ::' pos) ++' [d_model]))
    : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
    := if use_split_qkv_input return tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
       then Tensor.map'
              (fun resid_pre : tensor A [d_model]
               => Tensor.repeat resid_pre [n_heads]
                 : tensor A [n_heads; d_model])
              resid_pre
       else resid_pre.
  Definition query_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
    := add_head_dimension resid_pre.
  Definition key_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
    := add_head_dimension resid_pre.
  Definition value_input : tensor A ((batch ::' pos) ++' (maybe_n_heads use_split_qkv_input ::' d_model))
    := add_head_dimension resid_pre.

  Definition transformer_block_ln1 {r} {s : Shape r} (t : tensor A (s ::' d_model)) : tensor A (s ::' d_model)
    := layernorm eps ln1_w ln1_b t.
  Definition transformer_block_ln2 {r} {s : Shape r} (t : tensor A (s ::' d_model)) : tensor A (s ::' d_model)
    := layernorm eps ln2_w ln2_b t.

  Definition transformer_block_attn_only_out : tensor A (batch ++ [pos; d_model])
    := (let attn_out : tensor A (batch ++ [pos; d_model])
          := attn_out
               (n_ctx:=n_ctx)
               W_Q W_K W_V W_O
               b_Q b_K b_V b_O
               IGNORE
               (transformer_block_ln1 query_input)
               (transformer_block_ln1 key_input)
               (transformer_block_ln1 value_input) in
        resid_pre + attn_out)%core.
End TransformerBlock.

Section L0.
  Context {r} {batch : Shape r}
    (IGNORE := -1e5)
    (residual : tensor FLOAT (batch ++' [cfg.n_ctx; cfg.d_model])).

  Definition block0 : tensor FLOAT (batch ::' cfg.n_ctx ::' cfg.d_model)
    := transformer_block_attn_only_out
         (n_ctx:=cfg.n_ctx)
         L0_attn_W_Q L0_attn_W_K L0_attn_W_V L0_attn_W_O
         L0_attn_b_Q L0_attn_b_K L0_attn_b_V L0_attn_b_O
         IGNORE cfg.eps
         L0_ln1_w L0_ln1_b
         residual.
End L0.

Definition unembed {r} {s : Shape r} (residual : tensor FLOAT (s ::' cfg.n_ctx ::' cfg.d_model)) : tensor FLOAT (s ::' cfg.n_ctx ::' cfg.d_vocab_out)
  := (Tensor.map'
        (fun residual : tensor FLOAT [cfg.n_ctx; cfg.d_model]
         => weaksauce_einsum {{{ {{ pos d_model, d_model vocab -> pos vocab }}
                    , residual
                    , W_U }}}
           : tensor FLOAT [cfg.n_ctx; cfg.d_vocab_out])
        residual
      + broadcast b_U)%core.

Definition logits {r} {batch : Shape r} (tokens : tensor IndexType (batch ::' cfg.n_ctx))
  : tensor FLOAT (batch ::' cfg.n_ctx ::' cfg.d_vocab_out)
  := (let embed := embed tokens in
      let pos_embed := pos_embed tokens in
      let resid_shape := (batch ::' cfg.n_ctx ::' cfg.d_model)%shape in
      let residual : tensor FLOAT resid_shape := PArray.checkpoint (embed + pos_embed) in
      let residual : tensor FLOAT resid_shape := PArray.checkpoint (block0 residual) in
      let residual : tensor FLOAT resid_shape := PArray.checkpoint (ln_final residual) in
      let logits                          := PArray.checkpoint (unembed residual) in
      logits)%core.

Definition expected : tensor _ _ := Eval cbv in tensor_of_list [[11.4344;  0.5226;  3.3839;  1.9724;  4.5840;  0.6439;  3.9603;
           3.0340;  0.5467; -5.0662;  3.6980; -2.9019; -0.3635;  1.2298;
           4.1899; -1.8617;  1.8041;  1.8292;  3.8814;  1.5263; -0.3807;
           1.7364;  0.7279;  3.1928;  1.7333;  3.0208;  2.5443; -0.7108;
           0.9530; -2.2473; -0.9498; -2.4825;  0.0581;  3.0415;  0.8609;
           1.0091;  1.2683; -0.3473;  0.8327;  1.0073; -4.2868; -0.1874;
           1.8262;  0.7708;  3.0639;  2.7830;  3.9294; -1.5691; -4.2706;
          -0.5728;  2.8097;  0.7026;  3.4807;  0.1257; -3.0862; -5.0693;
          -0.7354; -2.0195; -4.0030; -0.7678; -2.5556; -4.7284; -3.5146;
          -5.1069];
         [ 6.0360; 11.5127;  4.9837;  4.2883; -1.8071;  3.2009;  4.2418;
           4.7591;  0.7054; -1.4242; -1.3333; -1.6746;  0.3838;  2.4033;
           3.9232;  3.2540;  0.3340; -3.2795;  4.9276;  1.1228;  2.5380;
           3.1367;  0.6326;  2.8105;  3.6972;  1.0036;  2.2720;  2.9813;
           1.5565;  0.4808;  3.2460; -3.2794; -4.5643;  3.1560;  2.5760;
          -1.0905; -0.1279; -0.5574; -1.7911;  1.7425; -2.4315; -6.5479;
          -3.5974; -1.0411; -0.4464;  0.3043;  3.2796;  2.8466; -0.4816;
          -1.8650;  3.2621; -2.8751; -0.6053;  2.9918; -1.3914; -6.3286;
           2.1076; -3.4658; -4.5172;  2.4994;  1.8171; -4.6102; -6.5373;
           1.7269]].
Time Definition result_arr : PArray.concrete_tensor _ _ := Eval vm_compute in PArray.concretize (logits (tensor_of_list [0; 1]%uint63)).
Compute FloatArith.PrimFloat.of_Q ((64*64)%Z * 0.3 / 60)%Q.
Compute PArray.concretize (logits (tensor_of_list [0; 2]%uint63)).
Definition result : tensor _ _ := PArray.abstract result_arr.
Compute PArray.concretize (result - expected)%core.
     = [| [| 11.434387031836676; 0.5225952099932295; 3.3839615308198665; 1.9723887274855556; 4.5840063309948658; 0.64385290801723039; 3.9602483380087743; 3.0339969681349053; 0.54666607504673503; -5.0662180322516335; 3.6979800582492062; -2.9019268688689768; -0.36348443193618468; 1.2297840763444583; 4.189857491356042; -1.8616910503821245; 1.804090211983326; 1.8292099701662567; 3.8813690053959768; 1.5262920988022219; -0.38072557083109465; 1.7363844416543817; 0.72784724183845517; 3.1928010802567672; 1.7333341503747564; 3.0208574099821242; 2.5442793427179553; -0.71076713961968063; 0.95301324929507791; -2.2472504997189153; -0.94977971836142461; -2.4825233318920228; 0.05807913103842096; 3.0415207967335487; 0.86088171076683107; 1.0090544553435303; 1.2682813303838136; -0.34728129319660128; 0.83264952656657487; 1.0073418196589397; -4.2868031462458358; -0.18743217937196893; 1.8262050705820332; 0.77080360675055126; 3.0638999117251253; 2.7829999418904188; 3.9294273712619519; -1.569034572864701; -4.2705754628179813; -0.57280493317185277; 2.8096841376280133; 0.70263283798417309; 3.4807142916345812; 0.12578831842769334; -3.0862327247845767; -5.0693239825969298; -0.73542478973588787; -2.019542098451649; -4.0029349556642133; -0.7678700145954146; -2.5555871045134904; -4.7283349691231136; -3.5146408535293845; -5.1069026943437343 |
          0 : float |]; [| 6.0360160792287969; 11.512676855279571; 4.9837538534347781; 4.2883503470396374; -1.8070936444021086; 3.2009103630435618; 4.2417619133695368; 4.7591326589726597; 0.70535258541459811; -1.4242060544427955; -1.333349936148857; -1.6746240601861251; 0.38384530563141156; 2.4032875969141934; 3.9231751531146357; 3.2539967672919414; 0.33398775746797815; -3.2795048004284282; 4.9276337190120616; 1.1227905011618451; 2.5379588444008108; 3.1366626154120354; 0.6325305731516675; 2.8104326649687406; 3.697226302709212; 1.0036645575970651; 2.2720114597247907; 2.9813322143054646; 1.5564877554718004; 0.48082360054450995; 3.2459437139521889; -3.2794239451063349; -4.5643368221776033; 3.1560178954026767; 2.5759588667793594; -1.090506705640119; -0.12788111123705706; -0.55740134375287398; -1.7911646630792972; 1.7425475380881124; -2.4314792009789867; -6.5479038953987692; -3.5973805570082895; -1.041050854628057; -0.44637160751175065; 0.30425237776553304; 3.2796284224670003; 2.846615225248923; -0.4816249794681694; -1.8650372765665457; 3.26210755147829; -2.8750565102352179; -0.60529706392899174; 2.9918399880007103; -1.3913747829982035; -6.3286350313135618; 2.1076734065113518; -3.4658154177561142; -4.5171855004796431; 2.4993281269426362; 1.8170869383488675; -4.6101297902501717; -6.5372888197882695; 1.7268999668675911 |
                        0 : float |] | [| | 0 : float |] : array float |]

Goal True.
  pose (PArray.concretize (logits (tensor_of_list [0; 1]%uint63))) as v.
  cbv beta delta [logits PArray.checkpoint] in v.
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.
  repeat match (eval cbv delta [v] in v) with
         | context V[let x := ?val in @?k x]
           => lazymatch val with
              | context[PArray.concretize ?val] => fail 1 val
              | _ => idtac
              end;
              let V := context V[match val with x => k x end] in
              change V in (value of v); cbv beta in v
         end.
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.
  repeat match (eval cbv delta [v] in v) with
         | context V[let x := ?val in @?k x]
           => lazymatch val with
              | context[PArray.concretize ?val] => fail 1 val
              | _ => idtac
              end;
              let V := context V[match val with x => k x end] in
              change V in (value of v); cbv beta in v
         end.
  set (k := PArray.concretize _) in (value of v) at 2.
  lazymatch (eval cbv delta [k] in k) with
  | PArray.concretize (block0 ?x)
    => set (k0 := x) in (value of k); set (k1 := block0 k0) in (value of k)
  end.
  cbv beta delta [block0] in k1.
  cbv beta zeta in k1.
  cbv beta delta [transformer_block_attn_only_out] in k1.
  cbv beta zeta in k1.
  Timeout 10 Time vm_compute in k.
  cbv beta delta [PArray.checkpoint] in k3.
  set (k_tmp := PArray.concretize _) in (value of k3).
  Time vm_compute in k_tmp.
  subst k_tmp.
  lazymatch (eval cbv delta [k1] in k1) with
  | (_ + ?x)%core
    => set (k2 := x) in (value of k1)
  end.
  clear -k2.
  cbv beta delta [attn_out] in k2.
  cbv beta zeta in k2.
  cbv beta delta [PArray.checkpoint] in k2.
  set (k_tmp := PArray.concretize _) in (value of k2).
  (*Set NativeCompute Profiling.
  Set NativeCompute Timing.*)
  Time vm_compute in k_tmp.
  subst k_tmp.
  lazymatch (eval cbv delta [k2] in k2) with
  | PArray.checkpoint (map' ?f ?x + ?y)%core
    => set (k3 := x) in (value of k2);
       set (k4 := y) in (value of k2)
  end.
  cbv [z] in k3.
  cbv beta delta [PArray.checkpoint] in k3.
  set (k_tmp := PArray.concretize _) in (value of k3).
  Time vm_compute in k_tmp.
  subst k_tmp.
  HERE
  cbv beta delta [z] in k3.
  cbv beta zeta in k3.
  let k := k3 in
  lazymatch (eval cbv delta [k] in k) with
  | PArray.checkpoint (map2' ?f ?x ?y)%core
    => let k1 := fresh "k" in
       let k2 := fresh "k" in
       set (k1 := x) in (value of k);
       set (k2 := y) in (value of k)
  end.
  cbv beta delta [v] in *.
  cbv beta zeta in k1.
  cbv beta delta [value_input key_input query_input add_head_dimension] in *.
  cbv beta zeta iota in k1, k2.
  cbv beta delta [transformer_block_ln1] in *.
  cbv beta zeta iota in k1, k2.
  set (lnv := layernorm _ _ _ _) in *.
  cbv beta delta [layernorm pattern] in *.
  cbv beta iota zeta in lnv.
  cbv beta delta [PArray.checkpoint] in lnv.
  set (k_tmp := PArray.concretize _) in (value of lnv).
  Time vm_compute in k_tmp.
  subst k_tmp.
  cbv beta delta [PArray.checkpoint] in k3.
  set (k_tmp := PArray.concretize _) in (value of k3).
  Time vm_compute in k_tmp.
  subst k_tmp.
  cbv beta delta [PArray.checkpoint] in k1.
  set (k_tmp := PArray.concretize _) in (value of k1).
  Time vm_compute in k_tmp.
  subst k_tmp.
  cbv beta zeta in k2.
  cbv beta delta [PArray.checkpoint] in k2.
  set (k_tmp := PArray.concretize _) in (value of k2).
  Time vm_compute in k_tmp.
  subst k_tmp.

  set (
  cbv [einsum_input] in k1.
  Timeout 5 Compute PArray.concretize k3.
               (*
  cbv beta delta [Shape.hd Shape.nil Shape.tl Shape.snoc fst snd Shape.cons Shape.app Nat.radd Tensor.raw_get] in k1.

  cbv beta iota zeta in k1.
  Set Printing All.

  cbn [fst snd] in k1.


  pose
  Timeout 5 Compute PArray.concretize lnv.
  pose (lnvc := PArray.concretize lnv).
  vm_compute in lnvc.
  vm_compute in lnv.
  clear -lnv.
  set (lnlv := layernorm_linpart _) in *.
  cbv beta delta [layernorm_linpart] in *.
  set (mv := reduce_axis_m1 _ _) in *.
  cbv beta delta [reduce_axis_m1] in *.
  clear -mv.
  cbv beta iota in mv.
  set (mv' := reduce_axis_m1' _ _) in *.
  clear -mv'.
  cbv beta delta [reduce_axis_m1'] in *.
  vm_compute Shape.tl in mv'.
  set (k_tmp := reshape_snoc_split) in *.
  cbv in k_tmp; subst k_tmp.
  cbv beta iota in mv'.
  cbv beta delta [map] in mv'.
  pose (PArray.concretize mv') as mv'c.
  vm_compute in mv'c.
  cbv [PArray.concretize Shape.snoc init_default] in mv'c.
  set (k_tmp := _ <=? max_length) in *.
  vm_compute in k_tmp; subst k_tmp; cbv beta iota in mv'c.
  vm_compute make in mv'c.
  cbv -[mv'] in mv'c.
  subst mv'.
  cbv beta iota zeta in mv'c.
  cbv [mean] in mv'c.
  vm_compute inject_Z_coer in mv'c.
  cbv [raw_get] in mv'c.
  cbv [RawIndex.snoc RawIndex.nil] in mv'c.
  cbv [sum] in mv'c.
  cbv [map_reduce] in mv'c.
  Timeout 5 cbv -[add Q_has_add k0 Qdiv] in mv'c.
  Time repeat (time (set (k0v := k0 _) in (value of mv'c) at 1;
                     timeout 5 vm_compute in k0v; subst k0v)).
  Time vm_compute in mv'c.
.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  set (k0v := k0 _) in (value of mv'c) at 1.
  timeout 5 vm_compute in k0v; subst k0v.
  do 10 (set (k0v := k0 _) in (value of mv'c) at 1;
          timeout 5 vm_compute in k0v; subst k0v).
  do 10 (set (k0v := k0 _) in (value of mv'c) at 1;
          timeout 5 vm_compute in k0v; subst k0v).
  do 10 (set (k0v := k0 _) in (value of mv'c) at 1;
          timeout 5 vm_compute in k0v; subst k0v).
  clear -k0v.
  set (val := (_ + _)%core) in (value of k0) at 1.
  cbv [ltb] in *.
  vm_compute RawIndex.repeat in k0.
  vm_compute Shape.snoc in k0.
  vm_compute PArray.abstract in k0.
  clearbody val.
  vm_compute in val.
  vm_compute RawIndex in *.

  vm_compute in k0v.
(*
  generalize dependent (_ + _)
  cbv in k0.
  cbv [RawIndex
  vm_compute in k0v.
  cbv in k0v.
  Timeout 5 vm_compute in k0v.
  vm_compute in k0v.
  subst k0.
  Set Printing All.
  cbv in mv'c.
  vm_compute

  cbv -
  cbv [reshape_snoc_split] in mv'.
  cbv [reshape_snoc_split] in mv'.

  Timeout 5 Compute PArray.concretize mv'.
  cbv beta delta [reduce_axis_m1'] in *.

  Timeout 5 Compute PArray.concretize mv.
  let k := k1 in
  let k1 := fresh "k" in
  let k2 := fresh "k" in
  lazymatch (eval cbv delta [k] in k) with
  | (map2' ?f ?x ?y)%core
    => set (k1 := x) in (value of k);
       set (k2 := y) in (value of k)
  end.



  set (k2 := attn_out _ _ _
  Time native_compute in k.
  Time vm_compute in k.
  subst k.

  cbv
  set (k := PArray.concretize _) in (value of v) at 2.
  Time vm_compute in k.
  subst k.

Time Timeout 5 Compute PArray.concretize (logits (tensor_of_list [0; 1]%uint63)).
Compute PArray.concrete_tensor

Compute PArray.concretize (embed (tensor_of_list [0; 1]%uint63)).
Compute PArray.concretize (pos_embed (tensor_of_list [[0; 1]]%uint63) : tensor FLOAT [1; cfg.n_ctx; cfg.d_model]).
*)
