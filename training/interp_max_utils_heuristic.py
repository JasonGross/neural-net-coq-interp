# %%
import math
from interp_max_utils import EU_PU_PVOU, all_EVOU
from max_of_n import acc_fn, loss_fn
from training_utils import compute_all_tokens
import torch
from einops import rearrange
from torchtyping import TensorType
from transformer_lens import HookedTransformer
from typing import Iterable, List, Optional
import numpy as np
import gc
# %%
def make_local_tqdm(tqdm):
    if tqdm is None:
        return lambda arg, **kwargs: arg
    else:
        return tqdm

# %%
@torch.no_grad()
def compute_heuristic_independence_attention_copying(model: HookedTransformer, min_gap: int = 1, tqdm=None) -> List[TensorType["batch"]]:
    """
    Assuming that attention paid to the non-max tokens is independent of the copying behavior on non-max tokens which are at least min_gap away, and also these are independent of the attention pattern on positional embeddings, computes the logit outputs, grouped by gap
    Returns: List[Tensor[batch, d_vocab_out]] (indices are gap)
    Complexity:
    """
    local_tqdm = make_local_tqdm(tqdm)
    n_ctx, d_vocab, d_vocab_out = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out

    all_tokens = compute_all_tokens(model=model)
    assert all_tokens.shape == (d_vocab**n_ctx, n_ctx), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits, cache = model.run_with_cache(all_tokens)
    predicted_logits = predicted_logits[:,-1,:].detach().cpu()
    assert predicted_logits.shape == (d_vocab**n_ctx, d_vocab_out), f"predicted_logits.shape = {predicted_logits.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"
    # pattern is post softmax, scores is presoftmax
    attention_patterns = cache['blocks.0.attn.hook_pattern']
    assert attention_patterns.shape == (d_vocab**n_ctx, 1, n_ctx, n_ctx), f"attention_pattern.shape = {attention_patterns.shape} != {(d_vocab**n_ctx, 1, n_ctx, n_ctx)} = (d_vocab**n_ctx, n_heads, n_ctx_q, n_ctx_k)"
    attention_patterns = attention_patterns[:,0,-1,:]
    EVOU = all_EVOU(model)
    assert EVOU.shape == (d_vocab, d_vocab_out), f"EVOU.shape = {EVOU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    EU_PU_PVOU_logits = EU_PU_PVOU(model, attention_patterns)
    assert EU_PU_PVOU_logits.shape == (d_vocab**n_ctx, d_vocab, d_vocab_out), f"EU_PU_PVOU_logits.shape = {EU_PU_PVOU_logits.shape} != {(d_vocab**n_ctx, d_vocab, d_vocab_out)} = (d_vocab**n_ctx, d_vocab, d_vocab_out)"
    copying_behaviors_for_attention = [[] for _ in range(attention_patterns.shape[0])]

    for i, (input_seq, pattern) in local_tqdm(enumerate(zip(all_tokens, attention_patterns)), total=all_tokens.shape[0]):
        true_max = input_seq.max().item()
        for s in torch.cartesian_prod(*([(torch.arange(true_max - min_gap + 1) if tok <= true_max - min_gap else torch.tensor([tok]))
                                         for tok in input_seq[:-1]]
                                         + [torch.tensor([input_seq[-1]])])):
            copying_behaviors_for_attention[i].append(pattern @ EVOU[s])
        copying_behaviors_for_attention[i] = torch.stack(copying_behaviors_for_attention[i])

    results = [[] for _ in range(d_vocab)]
    for i, (input_seq, copying_behavior) in local_tqdm(enumerate(zip(all_tokens, copying_behaviors_for_attention)), total=all_tokens.shape[0]):
        true_max = input_seq.max().item()
        gap = true_max - input_seq[input_seq != true_max].max().item() if any(input_seq != true_max) else 0
        positional_impact = EU_PU_PVOU_logits[:,input_seq[-1],:]
        assert positional_impact.shape == (d_vocab**n_ctx, d_vocab_out), f"positional_impact.shape = {positional_impact.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"
        positional_impact = positional_impact - positional_impact[:,true_max][:,None]
        assert len(copying_behavior.shape) == 2 and copying_behavior.shape[1] == d_vocab_out, f"copying_behavior.shape = {copying_behavior.shape} != (_, {d_vocab_out}) = (_, d_vocab_out)"
        copying_behavior = copying_behavior - copying_behavior[:,true_max][:,None]
        # remove the true_max column
        positional_impact = positional_impact[:,torch.arange(positional_impact.shape[1]) != true_max]
        copying_behavior = copying_behavior[:,torch.arange(copying_behavior.shape[1]) != true_max]
        assert positional_impact.shape == (d_vocab**n_ctx, d_vocab_out - 1), f"positional_impact.shape = {positional_impact.shape} != {(d_vocab**n_ctx, d_vocab_out - 1)} = (d_vocab**n_ctx, d_vocab_out - 1)"
        assert len(copying_behavior.shape) == 2 and copying_behavior.shape[1] == d_vocab_out - 1, f"copying_behavior.shape = {copying_behavior.shape} != (_, {d_vocab_out - 1}) = (_, d_vocab_out - 1)"
        # # move max up front
        # positional_impact = torch.cat([torch.zeros_like(positional_impact[:,0:1]), positional_impact], dim=1)
        # copying_behavior = torch.cat([torch.zeros_like(copying_behavior[:,0:1]), copying_behavior], dim=1)
        # assert positional_impact.shape == (d_vocab**n_ctx, d_vocab_out), f"positional_impact.shape = {positional_impact.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"
        # assert len(copying_behavior.shape) == 2 and copying_behavior.shape[1] == d_vocab_out, f"copying_behavior.shape = {copying_behavior.shape} != (_, {d_vocab_out}) = (_, d_vocab_out)"
        results[gap].append(rearrange(positional_impact[:,None,:] + copying_behavior[None,:,:],
                                      'batch_pos_impact batch_copying out -> (batch_pos_impact batch_copying) out'))
        # del positional_impact
        # del copying_behavior
    gc.collect()
    results = [torch.cat(result) for result in local_tqdm(results)]
    gc.collect()
    return results
# %%
@torch.no_grad()
def compute_loss_from_centered_results(results: Iterable[TensorType["batch", "d_vocab_minus_one"]], tqdm=None, total_count: Optional[int] = None):
    local_tqdm = make_local_tqdm(tqdm)
    results = list(results)
    if total_count is None: total_count = sum(result.shape[0] for result in results)
    res = 0
    for result in local_tqdm(results):
        # result.to('cuda' if torch.cuda.is_available() else 'cpu')
        res -= (1 + result.exp().sum(dim=-1)).log().sum().item()
    return -res / total_count

    # # add a 0 to the front of each result
    # full_results = ( for result in results)
    # log_probs = (result.log_softmax(dim=-1) for result in full_results)
    # correct_log_probs = (log_prob[:, 0] for log_prob in log_probs)
    # return -torch.cat(tuple(correct_log_probs)).mean()

# %%
@torch.no_grad()
def print_independence_attention_copying_stats(model: HookedTransformer, min_gap: int = 1, tqdm=None, results: Optional[TensorType["batch"]]=None):
    if results is None: return print_independence_attention_copying_stats(model, min_gap=min_gap, tqdm=tqdm, results=compute_heuristic_independence_attention_copying(model, min_gap=min_gap, tqdm=tqdm))
    gc.collect()
    d_vocab, n_ctx = model.cfg.d_vocab, model.cfg.n_ctx
    print(f"Assume that attention paid to the non-max tokens is independent of the copying behavior on non-max tokens which are at least {min_gap} away")
    bad_result_count_per_gap = [(r.max(dim=-1).values >= 0).sum() for r in results]
    bad_result_count = sum(bad_result_count_per_gap)
    total_count = sum(r.shape[0] for r in results)
    bad_result_rate = bad_result_count / total_count
    real_total_count = d_vocab ** n_ctx
    heuristic_loss = compute_loss_from_centered_results(results, tqdm=tqdm)
    all_tokens = compute_all_tokens(model=model)
    assert all_tokens.shape == (d_vocab**n_ctx, n_ctx), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    logits = model(all_tokens)
    real_loss = loss_fn(logits, all_tokens)
    real_acc = acc_fn(logits, all_tokens)
    print(f"Then we expect that in {bad_result_rate}% of cases ({bad_result_rate * real_total_count} out of {real_total_count}), the model will return an incorrect result")
    print(f"(The true accuracy is {real_acc} (1 - acc = {1 - real_acc}), i.e., {int(np.round((1 - real_acc) * real_total_count))} wrong out of {real_total_count})")
    print(f"The best possible heuristic loss is {heuristic_loss} (true loss: {real_loss})")
    print(f"The worst-case logit gap is {-max(r.max() for r in results)}")
    all_tokens_gaps = all_tokens.max(dim=-1, keepdim=True).values - all_tokens
    # replace 0s with float('inf')
    all_tokens_gaps[all_tokens_gaps == 0] = d_vocab + 1
    all_tokens_gaps = all_tokens_gaps.min(dim=-1).values
    # replace inf with 0
    all_tokens_gaps[all_tokens_gaps == d_vocab + 1] = 0
    bad_gap_loss_contrib = 0
    good_gap_loss_contrib = 0
    if bad_result_count > 0:
        print("Broken out by gap:")
        for gap, result_for_gap in enumerate(results):
            print(f"Gap {gap}:")
            print(f"Loss on gap {gap} is {compute_loss_from_centered_results([result_for_gap], tqdm=None)}")
            loss_contrib = compute_loss_from_centered_results([result_for_gap], tqdm=None, total_count=total_count)
            print(f"Loss contribution on gap {gap} is {loss_contrib}")
            if bad_result_count_per_gap[gap] > 0:
                gap_bad_result_count = bad_result_count_per_gap[gap]
                gap_total_count = result_for_gap.shape[0]
                gap_bad_result_rate = gap_bad_result_count / gap_total_count
                gap_real_total_count = (all_tokens_gaps == gap).sum().item()
                print(f"We expect that in {gap_bad_result_rate}% of cases ({gap_bad_result_rate * gap_real_total_count} out of {gap_real_total_count}), the model will return an incorrect result")
                print(f"The worst-case logit gap is {-result_for_gap.max()}")
                bad_gap_loss_contrib += loss_contrib
            else:
                good_gap_loss_contrib += loss_contrib
    print(f"Bad gap loss contribution is {bad_gap_loss_contrib}")
    print(f"Good gap loss contribution is {good_gap_loss_contrib}")
