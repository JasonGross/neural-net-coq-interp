From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Morphisms RelationClasses.
From NeuralNetInterp.Util Require Import Default Pointed PArray List Notations Arith.Classes Arith.Instances Bool Option.
From NeuralNetInterp.Util Require Nat Wf_Uint63.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer.
From NeuralNetInterp.MaxOfTwoNumbers Require Import Parameters.
Import Util.Nat.Notations.
Import Util.Wf_Uint63.LoopNotation.
Import Util.Wf_Uint63.
Import Util.Wf_Uint63.Reduction.
Import Arith.Instances.Truncating.
Local Open Scope float_scope.
Local Open Scope list_scope.
Set Implicit Arguments.
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
Definition W_E : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.W_E.
Definition W_pos : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.W_pos.
Definition L0_attn_W_Q : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_Q.
Definition L0_attn_W_K : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_K.
Definition L0_attn_W_V : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_V.
Definition L0_attn_W_O : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_W_O.
Definition L0_attn_b_Q : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_Q.
Definition L0_attn_b_K : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_K.
Definition L0_attn_b_V : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_V.
Definition L0_attn_b_O : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_attn_b_O.
Definition L0_ln1_b : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_ln1_b.
Definition L0_ln1_w : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.L0_ln1_w.
Definition ln_final_b : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.ln_final_b.
Definition ln_final_w : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.ln_final_w.
Definition W_U : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.W_U.
Definition b_U : tensor _ _ := Eval cbv in tensor_of_list MaxOfTwoNumbers.Parameters.b_U.

Definition all_tokens : tensor [(cfg.d_vocab ^ cfg.n_ctx)%core : N; 2] RawIndexType
  := let all_toks := Tensor.arange (start:=0) (Uint63.of_Z cfg.d_vocab) in
     PArray.checkpoint (Tensor.cartesian_prod all_toks all_toks).

Section with_batch.
  Context {r} {batch : Shape r} {pos}
    (s := (batch ::' pos)%shape)
    (resid_shape := (s ::' cfg.d_model)%shape)
    {return_per_token : with_default "return_per_token" bool false}
    {A} {coer_float : has_coer float A} {coerZ : has_coer Z A}
    {zeroA : has_zero A} {oneA : has_one A}
    (defaultA : pointed A := @coer _ _ coerZ point)
    {addA : has_add A} {subA : has_sub A} {mulA : has_mul A} {divA : has_div A}
    {ltbA : has_ltb A}
    {oppA : has_opp A} {sqrtA : has_sqrt A} {expA : has_exp A} {lnA : has_ln A}
    {use_checkpoint : with_default "use_checkpoint" bool true}.
  #[local] Existing Instance defaultA.
  #[local] Notation checkpoint x := (if use_checkpoint then PArray.checkpoint x else x%tensor).

  Let coer_tensor_float {r s} (x : @tensor r s float) : @tensor r s A
      := Tensor.map coer x.
  Let coerA' (x : float) : A := coer x.
  #[local] Set Warnings Append "-uniform-inheritance,-ambiguous-paths".
  #[local] Coercion coer_tensor_float : tensor >-> tensor.
  #[local] Set Warnings Append "uniform-inheritance,ambiguous-paths".
  #[local] Coercion coerA' : float >-> A.

  Definition embed (tokens : tensor s IndexType) : tensor resid_shape A
    := HookedTransformer.embed (A:=A) W_E tokens.

  Definition pos_embed (tokens : tensor s IndexType) : tensor resid_shape A
    := HookedTransformer.pos_embed (A:=A) (n_ctx:=cfg.n_ctx) W_pos tokens.

  Definition ln_final (resid : tensor resid_shape A) : tensor resid_shape A
    := HookedTransformer.ln_final (A:=A) cfg.eps ln_final_w ln_final_b resid.

  Definition unembed (resid : tensor resid_shape A) : tensor (s ::' cfg.d_vocab_out) A
    := HookedTransformer.unembed (A:=A) W_U b_U resid.

  Definition blocks_params : list _
    := [(L0_attn_W_Q:tensor _ A, L0_attn_W_K:tensor _ A, L0_attn_W_V:tensor _ A, L0_attn_W_O:tensor _ A,
          L0_attn_b_Q:tensor _ A, L0_attn_b_K:tensor _ A, L0_attn_b_V:tensor _ A,
          L0_attn_b_O:tensor _ A,
          L0_ln1_w:tensor _ A, L0_ln1_b:tensor _ A)].

  Definition logits (tokens : tensor s IndexType) : tensor (s ::' cfg.d_vocab_out) A
    := HookedTransformer.logits
         (A:=A)
         (n_ctx:=cfg.n_ctx)
         cfg.eps
         W_E
         W_pos

         blocks_params

         ln_final_w ln_final_b

         W_U b_U

         tokens.

  (** convenience *)
  Definition masked_attn_scores (tokens : tensor s IndexType)
    : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A
    := Option.invert_Some
         (HookedTransformer.HookedTransformer.masked_attn_scores
            (A:=A)
            (n_ctx:=cfg.n_ctx)
            cfg.eps
            W_E
            W_pos

            blocks_params

            0

            tokens).

  Definition attn_pattern (tokens : tensor s IndexType)
    : tensor (batch ::' cfg.n_heads ::' pos ::' pos) A
    := Option.invert_Some
         (HookedTransformer.HookedTransformer.attn_pattern
            (A:=A)
            (n_ctx:=cfg.n_ctx)
            cfg.eps
            W_E
            W_pos

            blocks_params

            0

            tokens).

  Definition loss_fn
    (logits : tensor (s ::' cfg.d_vocab_out) A)
    (tokens : tensor s IndexType)
  : tensor (if return_per_token return Shape (if return_per_token then _ else _) then Shape.squeeze batch else []) A
  := (let logits : tensor (batch ::' _) A
        := checkpoint (logits.[…, -1, :]) in
      let true_maximum : tensor (batch ::' 1) IndexType
        := reduce_axis_m1 (keepdim:=true) Reduction.max tokens in
      let log_probs
        := log_softmax_dim_m1 logits in
      let correct_log_probs
        := checkpoint (gather_dim_m1 log_probs true_maximum) in
      if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) A)
      then -Tensor.squeeze correct_log_probs
      else -Tensor.mean correct_log_probs)%core.

  Definition acc_fn
    (logits : tensor (s ::' cfg.d_vocab_out) A)
    (tokens : tensor s IndexType)
    : tensor (if return_per_token return Shape (if return_per_token then _ else _) then batch else []) A
    := (let pred_logits : tensor (batch ::' _) A
          := checkpoint (logits.[…, -1, :]) in
        let pred_tokens : tensor batch IndexType
          := reduce_axis_m1 (keepdim:=false) Reduction.argmax pred_logits in
        let true_maximum : tensor batch IndexType
          := reduce_axis_m1 (keepdim:=false) Reduction.max tokens in
        let res : tensor _ A
          := checkpoint (Tensor.of_bool (Tensor.map2 eqb pred_tokens true_maximum)) in
        if return_per_token return (tensor (if return_per_token return Shape (if return_per_token then _ else _) then _ else _) A)
        then res
        else Tensor.mean res)%core.
End with_batch.

Notation model := logits (only parsing).

Definition logits_all_tokens : tensor _ float
  := logits all_tokens.

(*
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
Compute FloatArith.Definitions.PrimFloat.of_Q ((64*64)%Z * 0.3 / 60)%Q.
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
Compute PArray.concretize (pos_embed (tensor_of_list [[0; 1]]%uint63) : tensor [1; cfg.n_ctx; cfg.d_model] FLOAT).
*)
*)*)
