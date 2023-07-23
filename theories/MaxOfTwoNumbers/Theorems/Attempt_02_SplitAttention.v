From Coq Require Import Floats Sint63 Uint63 QArith Lia List PArray Derive.
From NeuralNetInterp.Torch Require Import Tensor Einsum Slicing.
From NeuralNetInterp.Util Require Import Pointed Wf_Uint63 Wf_Uint63.Instances SolveProperEqRel.
From NeuralNetInterp.Util.Arith Require Import Classes Instances FloatArith.Definitions.
From NeuralNetInterp.Torch Require Import Tensor.Instances Slicing.Instances.
From NeuralNetInterp.TransformerLens Require Import HookedTransformer HookedTransformer.Instances.
From NeuralNetInterp.MaxOfTwoNumbers Require Import Parameters Model Heuristics TheoremStatement Model.Instances.
Import LoopNotation.
(*From NeuralNetInterp.MaxOfTwoNumbers.Computed Require Import AllLogits.*)
Local Open Scope uint63_scope.
Local Open Scope core_scope.

Local Ltac let_bind_1 _ :=
  match goal with
  | [ |- context G[let n := ?v in @?f n] ]
    => let n' := fresh n in
       set (n' := v); let G := context G[f n'] in change G; cbv beta
  end.

Local Ltac let_bind_hyp _ :=
  match goal with
  | [ H := context G[let n := ?v in @?f n] |- _ ]
    => let n' := fresh n in
       set (n' := v) in (value of H); let G := context G[f n'] in change G in (value of H); cbv beta in H
  end.

Local Ltac let_bind _ := repeat first [ progress cbv beta iota in * | let_bind_1 () | let_bind_hyp () ].

Local Ltac let_bind_subst_shape _ :=
  let_bind ();
  repeat match goal with H : Shape _ |- _ => subst H end;
  repeat match goal with H : forall b : bool, Shape _ |- _ => subst H end.

#[export] Existing Instance reflexive_eq_dom_reflexive.
#[local] Hint Constants Opaque : typeclass_instances.
#[local] Typeclasses Transparent
  relation fst snd
  Classes.max Instances.Uint63.max has_default_max_leb
  Classes.leb Instances.Uint63.leb
  Classes.one
  Shape ShapeType Shape.nil Shape.snoc Shape.app Shape.broadcast2 Shape.broadcast3 Shape.map Shape.map2 Shape.map3 Shape.hd Shape.tl Shape.ones Shape.repeat Shape.ShapeType.one
  RawIndex RawIndexType RawIndex.nil RawIndex.snoc RawIndex.app RawIndex.hd RawIndex.tl
  Index IndexType Index.nil Index.snoc Index.app Index.hd Index.tl
  SliceIndex.transfer_shape SliceIndex.SliceIndexType.transfer_shape Slice.start Slice.stop Slice.step Slice.Concrete.length Slice.norm_concretize SliceIndex.SliceIndexType.transfer_shape_single_index
  inject_int int_has_one
  Rank Nat.radd
  Classes.add Classes.div Classes.sub Classes.mul
  has_add_with has_div_by has_sub_with has_mul_with
  tensor tensor_of_rank
  Tensor.eqfR
  raw_get
.
#[export] Hint Transparent PrimInt63.leb : typeclass_instances.
#[local] Set Keyed Unification.
Local Ltac Proper_Tensor_eqf_t_step _
  := match goal with
     | [ |- Proper (_ ==> _) (fun x => _) ]
       => intros ???
     | [ |- Tensor.eqfR _ (fun x => _) (fun y => _) ]
       => intro
     | [ |- ?R (match ?x with _ => _ end) (match ?y with _ => _ end) ]
       => tryif constr_eq x y
         then destruct x eqn:?; subst
         else destruct x eqn:?, y eqn:?; subst
     | [ H : Tensor.eqfR _ ?x ?y |- _ ]
       => move H at top;
          first [ is_var x; revert dependent x; intros x H; setoid_rewrite H; clear H x; intros
                | is_var y; revert dependent y; intros y H; setoid_rewrite <- H; clear H y; intros ]
     | [ |- ?R ?x ?x ] => reflexivity
     end.
Local Ltac Proper_Tensor_eqf_t _ := repeat Proper_Tensor_eqf_t_step ().
#[export] Hint Extern 1 (Proper (_ ==> _) (fun _ => _)) => progress Proper_Tensor_eqf_t () : typeclass_instances.
Theorem good_accuracy : TheoremStatement.Accuracy.best (* (abs (real_accuracy - expected_accuracy) <? error)%float = true *).
Proof.
  cbv [real_accuracy].
  cbv beta iota delta [acc_fn]; let_bind_subst_shape ().
  cbv beta iota delta [logits] in *; let_bind_subst_shape ().
  cbv beta iota delta [HookedTransformer.logits] in *; let_bind_subst_shape ().
  cbv beta iota delta [blocks_params] in *.
  cbv beta iota delta [HookedTransformer.blocks_cps fold_right HookedTransformer.blocks List.map] in *; let_bind_subst_shape ().
  vm_compute Shape.tl in *.
  vm_compute of_Z in *.
  vm_compute SliceIndex.transfer_shape in *.
  vm_compute Shape.app in *.
  vm_compute Shape.broadcast2 in *.
  cbv beta iota delta [TransformerBlock.attn_only_out] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.attn_out] in *; let_bind_subst_shape ().
  cbv beta iota delta [Attention.z] in *; let_bind_subst_shape ().
  set (v := Attention.v _ _ _) in *.
  set (pattern := Attention.pattern _ _ _ _ _ _) in *.
  cbv beta iota delta [HookedTransformer.unembed] in *; let_bind_subst_shape ().
  cbv beta iota delta [Unembed.forward] in *; let_bind_subst_shape ().
  cbv beta iota delta [all_tokens] in true_maximum; let_bind_subst_shape ().
  set (all_toks_c := PArray.checkpoint _) in (value of true_maximum).
  set (out' := PArray.checkpoint _) in (value of out).
  Local Ltac setoid_rewrite_in_body R lem H :=
    let rewrite_or_error lem :=
      let rew _ := (rewrite lem) (* first [ rewrite lem | setoid_rewrite lem ] *) in
      tryif rew ()
      then idtac
      else (match goal with
            | [ |- ?G ]
              => let lemo := open_constr:(lem) in
                 let T := type of lemo in
                 idtac "Could not rewrite" lem ":" T "in goal" G
            end;
            rew ()) in
    let ty := open_constr:(_) in
    let H' := fresh H in
    rename H into H';
    evar (H : ty);
    let lemH := fresh in
    assert (lemH : R H' H) by (subst H'; rewrite_or_error lem; subst H; reflexivity);
    let rec do_rewrite _
      := lazymatch goal with
         | [ H'' := context[H'] |- _ ]
           => setoid_rewrite_in_body R lemH H'';
              do_rewrite ()
         | _ => idtac
         end in
    do_rewrite ();
    lazymatch goal with
    | [ |- context[H'] ] => rewrite_or_error lemH
    | _ => idtac
    end;
    clear lemH; clear H'.
  Tactic Notation "setoid_rewrite" "(" uconstr(R) ")" uconstr(lem) "in" "(" "value" "of" hyp(H) ")" := setoid_rewrite_in_body R lem H.
  do 8 lazymatch goal with
  | [ H := PArray.checkpoint _ |- _ ]
    => setoid_rewrite (Tensor.eqf) Tensor.PArray.checkpoint_correct_eqf in (value of H)
    end.
  cbv beta iota delta [of_bool map map2] in res.
  move true_maximum at bottom.
  move all_toks_c at bottom.
  move all_toks at bottom.
  cbv beta iota delta [reduce_axis_m1 reduce_axis_m1' reshape_snoc_split RawIndex.curry_radd RawIndex.combine_radd map RawIndex] in true_maximum.
  cbv beta iota delta [Reduction.max max has_default_max_leb leb] in true_maximum.
  cbv -[PrimInt63.leb all_toks_c] in true_maximum.
  move out' at bottom.
  cbv -[map2' raw_get v pattern RawIndex.snoc RawIndex.nil] in out'.
  unfold Reduction.sum in (value of out) at 1.
  cbv -[map2' Reduction.sum L0_attn_W_O out'] in out.
  cbv [mean Reduction.mean reduce_axis_m1 reduce_axis_m1' map item SliceIndex.slice raw_get Truncating.coer_Z_float Shape.reshape' Shape.reduce Shape.tl snd reshape_m1 reshape_snoc_split].
  vm_compute of_Z.
  vm_compute PrimFloat.of_Z.
  cbv [RawIndex.unreshape RawIndex.unreshape' RawIndex.curry_radd RawIndex.combine_radd RawIndex.item RawIndex.snoc RawIndex.tl RawIndex.nil snd Shape.tl].
  cbv [expected_accuracy].
  cbv [error].
  vm_compute expected_accuracy.
  subst all_toks_c.
  cbv beta iota delta [cartesian_prod] in true_maximum.
  cbv beta iota delta [reshape_m1 modulo PrimInt63.mod Uint63.to_Z Uint63.to_Z_rec Uint63.size Uint63.is_even Uint63.is_zero Uint63.land Uint63.eqb Uint63.compare PrimInt63.lsr Z.double Z.succ_double Z.to_nat nth_default nth_error Pos.to_nat Pos.iter_op] in true_maximum.
  cbv [raw_get] in true_maximum.
  vm_compute PrimInt63.add in true_maximum.
  cbv [tupleify] in true_maximum.
  cbv [RawIndex.unreshape] in true_maximum.
  cbv [RawIndex.unreshape'] in true_maximum.
  cbv [RawIndex.item RawIndex.tl RawIndex.snoc RawIndex.nil fst snd Shape.hd RawIndex.hd Shape.tl Shape.snoc Shape.nil] in true_maximum.
  move all_toks at bottom.
  vm_compute of_Z in all_toks.
  cbv [arange] in all_toks.
  subst all_toks.
  cbv [RawIndex.item RawIndex.tl raw_get] in true_maximum.
  cbn [snd] in true_maximum.
  cbn -[all_toks] in true_maximum'.
  cbv [Uint63.to_Z Uint63.to_Z_red] in true_maximum'.

  HERE?
https://coq.zulipchat.com/#narrow/stream/237977-Coq-users/topic/Working.20with.20primitive.20floats/near/377735862


  cbv [RawInde
  cbv [
  cbv [tupleify] in
  cbn [
  cbv -[all_toks] in true_maximum'.
  subst true_maximum.
  cbv
  cbv


  cbv beta iota delta [reduce_axis_m1 reduce_axis_m1' reshape_snoc_split RawIndex.curry_radd RawIndex.combine_radd map RawIndex] in pred_tokens.
  Typeclasses eauto := debug.
  #[export] Instance : Params (@Tensor.eqfR) 4 := {}.
  #[export] Instance : Params (@Tensor.map2) 6 := {}.
  #[export] Instance : Params (@Tensor.of_bool) 5 := {}.
  #[export] Instance : Params (@SliceIndex.slice) 5 := {}.
  #[export] Instance : Params (@map') 9 := {}.
  try lazymatch goal with
  | [ H := PArray.checkpoint _ |- _ ]
    => setoid_rewrite (Tensor.eqf) Tensor.PArray.checkpoint_correct_eqf in (value of H)
  end.
  Unshelve.
  2: { clear.
       exact _.
       generalize (@Tensor.map'_Proper_2).
       generalize (@map'); intro map'.
       repeat lazymatch goal with
              | [ |- context[map' ?x] ]
         => intro H; specialize (H x); revert H; generalize (map' x); clear map'; intro map'; assert_fails typeclasses eauto
              end.
       Set Typeclasses Debug Verbosity 2.
       intro H.
       try exact _.
       intro H; specialize (H eq eq).

       Set Typeclasses Debug.
       try typeclasses eauto.
       exact _.
       Unshelve.
       clear.
       lazymatch goal with
         | [ H : Tensor.eqfR _ ?x ?y |- _ ]
           => move H at top;
              is_var x;
              revert dependent x; intros x H; setoid_rewrite H
       end.
              first [ is_var x; revert dependent x; intros x H; setoid_rewrite H; clear x; intros
                    | is_var y; revert dependent y; intros y H; setoid_rewrite <- H; clear y; intros ]
       end.
       setoid_rewrite H.
       reflexivity.
       change (x ?i) with (raw_get x i).
       change (y ?i) with (raw_get y i).
       setoid_rewrite H.
       Set Printing All.
       intros ?? H.

       Typeclasses eauto := debug.
       Set Typeclasses Debug Verbosity 2.

       try exact _.
       assert ((@Shape.snoc
                                                 (S (S O))
                                                 (@Shape.snoc
                                                    (S O)
                                                    (@Shape.snoc O Shape.nil 0x1000)
                                                    0x2) 0x40) =
(@Shape.broadcast2 (S (S (S O)))
   (@Shape.app (S (S O)) (S O)
      (@Shape.snoc (S O) (@Shape.snoc O Shape.nil 0x1000) 0x2)
      (@Shape.snoc O Shape.nil 0x40))
   (@Shape.app (S (S O)) (S O) (@Shape.ones (S (S O)))
      (@Shape.snoc O Shape.nil 0x40)))).
       cbv [relation fst snd
  Classes.max Instances.Uint63.max has_default_max_leb
  Classes.leb Instances.Uint63.leb
  Classes.one
  Shape ShapeType Shape.nil Shape.snoc Shape.app Shape.broadcast2 Shape.broadcast3 Shape.map Shape.map2 Shape.map3 Shape.hd Shape.tl Shape.ones Shape.repeat Shape.ShapeType.one int_has_one
  RawIndex RawIndexType RawIndex.nil RawIndex.snoc RawIndex.app RawIndex.hd RawIndex.tl
  Index IndexType Index.nil Index.snoc Index.app Index.hd Index.tl
  SliceIndex.transfer_shape SliceIndex.SliceIndexType.transfer_shape Slice.start Slice.stop Slice.step Slice.Concrete.length Slice.norm_concretize SliceIndex.SliceIndexType.transfer_shape_single_index
  inject_int
  Rank Nat.radd
  Classes.add Classes.div Classes.sub Classes.mul
  has_add_with has_div_by has_sub_with has_mul_with
  tensor].
       cbv [PrimInt63.leb].
       cbv [one].
      vm_compute.
  2: { generalize (@Tensor.map'_Proper_2).
       clear.
       generalize (@map'); intro map'.
       repeat lazymatch goal with
       | [ |- context[map' ?x] ]
         => intro H; specialize (H x); revert H; generalize (map' x); clear map'; intro map'; assert_fails typeclasses eauto
              end.
       intro H; specialize (H eq eq).
       apply H.
       exact _.
       exact _.
       exact _.
       | [ H : context[Proper _ (?f
       Set Printing Implicit.

       (*Set Typeclasses Debug Verbosity 2.
       Set Debug "tactic-unification".*)
       try typeclasses eauto.

  let ty := open_constr:(_) in
  evar (res' : ty);
  let lem := open_constr:(Tensor.eqf res res') in
  unshelve erewrite (_ : lem);
  [ subst res res'; rewrite Tensor.PArray.checkpoint_correct_eqf; reflexivity | clear res; rename res' into res ].
  let res' := open_constr:(_) in
  setoid_replace res with res'.
  subst res.
  rewrite
  setoid_rewrite PArray.checkpoint_correct.
  cbv [item mean].
    cbv [reduce_axis_m1 reduce_axis_m1' reshape_snoc_split map RawIndex.curry_radd reshape_m1].
  cbv [raw_get].
  cbv [RawIndex.unreshape].
  cbv [RawIndex.item].
  cbv [RawIndex.tl].
  cbv [RawIndex.combine_radd].
  vm_compute of_Z.
  cbv [RawIndex.unreshape'].
  cbv [Shape.tl].
  cbn [snd].
  cbv [Shape.snoc].
  cbv [RawIndex.snoc].
  cbv [RawIndex.nil].
  cbn [fst snd].
  cbv [Reduction.mean].
  vm_compute Truncating.coer_Z_float.

  rewrite FloatAxioms.ltb_spec, FloatAxioms.abs_spec, FloatAxioms.sub_spec, FloatAxioms.div_spec.
  vm_compute (Prim2SF error).
  vm_compute (Prim2SF expected_accuracy).
  cbv [SFltb SFabs SFcompare SF64sub].
  cbv [
  Search PrimFloat.of_Z.
  Search SFltb.
  Print SFltb.
  Search PrimFloat.abs.



  Set Debug "Cbv".
  cbv -[PrimFloat.ltb PrimFloat.abs PrimFloat.sub Reduction.mean res expected_accuracy error].


  Set Printing
  cbv [
  set (mean_res := item (mean res)).

  let T := open_constr:(_) in
  evar (mean_res' : T);
  replace mean_res with mean_res'; revgoals; [ symmetry | ].
  { subst mean_res mean_res'.

    apply Tensor.item_Proper.
    apply Tensor.mean_Proper.
    subst res.
    apply Tensor.PArray.checkpoint_Proper.
    apply Tensor.of_bool_Proper.
    apply map2_Pr
  Typeclasses eauto := debug.
  setoid_replace (P
  rewrite PArray.checkpoint_correct.


  cbn [Shape.tl Shape.snoc] in *.
  cbv

  rewrite <- computed_accuracy_eq.
  vm_compute; reflexivity.
Qed.
