# NOTE that this file is synchronized with theories/TrainingComputations/interp_max_utils.v
# In[ ]:
from typing import Any, Dict, Optional, Union
from torchtyping import TensorType
from enum import Enum, verify, UNIQUE, CONTINUOUS
import enum
import itertools
from analysis_utils import make_local_tqdm
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
import plotly.express as px
from analysis_utils import summarize
from training_utils import compute_all_tokens, generate_all_sequences
import math

# In[ ]:
def complexity_of(f):
    lines = (line.split(':') for line in f.__doc__.split('\n'))
    lines = (line for line in lines if line[0].lower().strip().startswith('complexity'))
    lines = (':'.join(line[1:]).strip() if line[0].lower().strip() == 'complexity' else ':'.join(line).strip()[len('complexity'):].strip()
             for line in lines)
    return '\n'.join(lines)

# In[ ]:
@torch.no_grad()
def logit_delta_of_results(all_tokens: TensorType["batch", "n_ctx"], predicted_logits: TensorType["batch", "d_vocab_out"], renderer=None, histogram_all_incorrect_logit_differences: bool = False, return_summary: bool = False, hist_args={}) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max.
    """
    (batch, n_ctx), (_batch, d_vocab_out) = all_tokens.shape, predicted_logits.shape
    assert predicted_logits.shape == (batch, d_vocab_out), f"predicted_logits.shape = {predicted_logits.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"

    # Extract statistics for each row
    # Use values in all_tokens as indices to gather correct logits
    indices_of_max = all_tokens.max(dim=-1, keepdim=True).values
    assert indices_of_max.shape == (batch, 1), f"indices_of_max.shape = {indices_of_max.shape} != {(batch, 1)} = (batch, 1)"
    correct_logits = torch.gather(predicted_logits, -1, indices_of_max)
    assert correct_logits.shape == (batch, 1), f"correct_logits.shape = {correct_logits.shape} != {(batch, 1)} = (batch, 1)"
    logits_above_correct = correct_logits - predicted_logits
    assert logits_above_correct.shape == (batch, d_vocab_out), f"logits_above_correct.shape = {logits_above_correct.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"
    # replace correct logit indices with large number so that they don't get picked up by the min
    logits_above_correct[torch.arange(logits_above_correct.shape[0]), indices_of_max.squeeze()] = float('inf')
    min_incorrect_logit = logits_above_correct.min(dim=-1).values
    assert min_incorrect_logit.shape == (batch,), f"min_incorrect_logit.shape = {min_incorrect_logit.shape} != {(batch,)} = (batch,)"

    if histogram_all_incorrect_logit_differences:
        all_incorrect_logits = logits_above_correct[logits_above_correct != float('inf')]
        summarize(all_incorrect_logits, name='all incorrect logit differences', histogram=True, hist_args=hist_args, renderer=renderer)

    if return_summary:
        return summarize(min_incorrect_logit, name='min(correct logit - incorrect logit)', renderer=renderer, histogram=True)

    else:
        return min_incorrect_logit.min().item()


# In[ ]:
@torch.no_grad()
def logit_delta(model: HookedTransformer, renderer=None, histogram_all_incorrect_logit_differences: bool = False, return_summary: bool = False, hist_args={}) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max.
    Complexity: O(d_vocab^n_ctx * fwd_pass)
    Complexity: fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    Complexity: n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out, model.cfg.d_model

    all_tokens = compute_all_tokens(model=model)
    assert all_tokens.shape == (d_vocab**n_ctx, n_ctx), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits = model(all_tokens)[:,-1,:].detach().cpu()
    assert predicted_logits.shape == (d_vocab**n_ctx, d_vocab_out), f"predicted_logits.shape = {predicted_logits.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"

    return logit_delta_of_results(all_tokens=all_tokens, predicted_logits=predicted_logits, renderer=renderer, histogram_all_incorrect_logit_differences=histogram_all_incorrect_logit_differences, return_summary=return_summary, hist_args=hist_args)

# In[ ]:
@torch.no_grad()
def compute_gap(all_tokens: TensorType["batch", "n_ctx"]) -> TensorType["batch"]:
    """
    computes the gap between the max token and the second max token in each row of all_tokens
    """
    maxv = all_tokens.max(dim=-1, keepdim=True).values
    all_but_maxv = all_tokens.clone()
    all_but_maxv[all_but_maxv == maxv] = -all_tokens.max().item()
    second_maxv = all_but_maxv.max(dim=-1, keepdim=True).values
    second_maxv[second_maxv < 0] = maxv[second_maxv < 0]
    return (maxv - second_maxv)[:, 0]

# In[ ]:
@torch.no_grad()
def all_tokens_small_gap(model: HookedTransformer, max_min_gap: int = 1) -> TensorType["batch", "n_ctx"]:
    """
    All sequences of tokens with the constraint that some token z in the sequence satisfies true_max - max_min_gap <= z < true_max
    Complexity: O(d_vocab ^ (n_ctx - 1) * (max_min_gap * 2 + 1))
    """
    n_ctx, d_vocab = model.cfg.n_ctx, model.cfg.d_vocab

    all_tokens_after_start = generate_all_sequences(n_digits=d_vocab, sequence_length=n_ctx - 1)
    all_tokens_after_start_max = all_tokens_after_start.max(dim=-1, keepdim=True).values
    all_tokens_after_start_max_minf = all_tokens_after_start.clone()
    all_tokens_after_start_max_minf[all_tokens_after_start_max_minf == all_tokens_after_start_max] = -max_min_gap - 1
    all_tokens_after_start_second_max = all_tokens_after_start_max_minf.max(dim=-1, keepdim=True).values
    first_token_max = all_tokens_after_start_max + max_min_gap + 1
    gap_already_present = all_tokens_after_start_second_max >= all_tokens_after_start_max - max_min_gap
    first_token_upper_min = all_tokens_after_start_max + gap_already_present.long()
    first_token_min = torch.zeros_like(first_token_max)
    first_token_min[~gap_already_present] = all_tokens_after_start_max[~gap_already_present] - max_min_gap
    first_token_min[first_token_min < 0] = 0
    first_token_max[first_token_max >= d_vocab] = d_vocab
    first_token_upper_min[first_token_upper_min >= d_vocab] = d_vocab
    assert first_token_max.shape == (d_vocab**(n_ctx - 1), 1), f"first_token_max.shape = {first_token_max.shape} != {(d_vocab**(n_ctx - 1), 1)} = (d_vocab**(n_ctx - 1), 1)"
    assert first_token_upper_min.shape == (d_vocab**(n_ctx - 1), 1), f"first_token_upper_min.shape = {first_token_upper_min.shape} != {(n_ctx, 1)} = (d_vocab**(n_ctx - 1), 1)"
    assert all_tokens_after_start_max.shape == (d_vocab**(n_ctx - 1), 1), f"all_tokens_after_start_max.shape = {all_tokens_after_start_max.shape} != {(d_vocab**(n_ctx - 1), 1)} = (d_vocab**(n_ctx - 1), 1)"
    assert first_token_min.shape == (d_vocab**(n_ctx - 1), 1), f"first_token_min.shape = {first_token_min.shape} != {(d_vocab**(n_ctx - 1), 1)} = (d_vocab**(n_ctx - 1), 1)"
    first_token_max, first_token_upper_min, all_tokens_after_start_max, first_token_min = first_token_max[:, 0], first_token_upper_min[:, 0], all_tokens_after_start_max[:, 0], first_token_min[:, 0]
    first_token_ranges = [torch.cat([torch.arange(lower, mid), torch.arange(lower_big, upper)]) for lower, mid, lower_big, upper in zip(first_token_min, all_tokens_after_start_max, first_token_upper_min, first_token_max)]
    all_tokens_with_small_gap = torch.cat([torch.cartesian_prod(first_tokens, *rest_tokens[:, None]) for first_tokens, rest_tokens in zip(first_token_ranges, all_tokens_after_start)])

    return all_tokens_with_small_gap

# In[ ]:
@torch.no_grad()
def logit_delta_small_gap_exhaustive(model: HookedTransformer, max_min_gap: int = 1, renderer=None, histogram_all_incorrect_logit_differences: bool = False, return_summary: bool = False, hist_args={}) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max, with the constraint that some token z in the sequence satisfies true_max - max_min_gap <= z < true_max
    Complexity: O(d_vocab ^ (n_ctx - 1) * (max_min_gap * 2 + 1) * fwd_pass)
    Complexity: fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    Complexity: n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out, model.cfg.d_model

    all_tokens = all_tokens_small_gap(model, max_min_gap=max_min_gap)
    assert len(all_tokens.shape) == 2 and all_tokens.shape[1] == n_ctx, f"all_tokens.shape = {all_tokens.shape} != (_, {n_ctx}) = (_, n_ctx)"
    predicted_logits = model(all_tokens)[:,-1,:].detach().cpu()
    assert len(predicted_logits.shape) == 2 and predicted_logits.shape[1] == d_vocab_out, f"predicted_logits.shape = {predicted_logits.shape} != (_, {d_vocab_out}) = (_, d_vocab_out)"

    return logit_delta_of_results(all_tokens=all_tokens, predicted_logits=predicted_logits, renderer=renderer, histogram_all_incorrect_logit_differences=histogram_all_incorrect_logit_differences, return_summary=return_summary, hist_args=hist_args)

# In[ ]:
@torch.no_grad()
def logit_delta_by_gap(model: HookedTransformer, renderer=None, histogram_all_incorrect_logit_differences: bool = False, return_summary: bool = False, hist_args={}) -> Dict[int, Union[float, Dict[str, Any]]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max, with the constraint that all non-max tokens in the sequence are strictly more than gap away from the true max, indexed by gap
    Complexity: O(d_vocab ^ n_ctx * fwd_pass)
    Complexity: fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    Complexity: n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out, model.cfg.d_model

    all_tokens = compute_all_tokens(model=model)
    assert all_tokens.shape == (d_vocab**n_ctx, n_ctx), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits = model(all_tokens)[:,-1,:].detach().cpu()
    assert predicted_logits.shape == (all_tokens.shape[0], d_vocab_out), f"predicted_logits.shape = {predicted_logits.shape} != {(all_tokens.shape[0], d_vocab_out)} = (all_tokens.shape[0], d_vocab_out)"
    gaps = compute_gap(all_tokens)
    assert gaps.shape == (all_tokens.shape[0],), f"gaps.shape = {gaps.shape} != {(all_tokens.shape[0],)} = (all_tokens.shape[0],)"
    return {gap: logit_delta_of_results(all_tokens=all_tokens[gaps == gap, :], predicted_logits=predicted_logits[gaps == gap, :], renderer=renderer, histogram_all_incorrect_logit_differences=histogram_all_incorrect_logit_differences, return_summary=return_summary, hist_args=hist_args)
            for gap in range(d_vocab)}

# In[ ]:
@torch.no_grad()
def EU_PU(model: HookedTransformer, renderer=None, pos: int = -1) -> TensorType["d_vocab_q", "d_vocab_out"]:
    """
    Calculates logits from just the EU and PU paths in position pos.
    Complexity: O(d_vocab^2 * d_model)
    Return shape: (d_vocab, d_vocab_out) (indexed by query token)
    """
    W_E, W_pos, W_U = model.W_E, model.W_pos, model.W_U
    d_model, n_ctx, d_vocab, d_vocab_out = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_U.shape == (d_model, d_vocab_out)

    result = (W_E + W_pos[pos][None, :]) @ W_U
    assert result.shape == (d_vocab, d_vocab_out)

    return result

# In[ ]:
@torch.no_grad()
def all_attention_scores(model: HookedTransformer) -> TensorType["n_ctx_k", "d_vocab_q", "d_vocab_k"]:
    """
    Returns pre-softmax attention of shape (n_ctx_k, d_vocab_q, d_vocab_k)
    Complexity: O(d_vocab * d_head^2 * d_model * n_ctx)
    """
    W_E, W_pos, W_Q, W_K = model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, n_ctx, d_vocab, d_head = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_head
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_Q.shape == (1, 1, d_model, d_head)
    assert W_K.shape == (1, 1, d_model, d_head)

    last_resid = (W_E + W_pos[-1]) # (d_vocab, d_model). Rows = possible residual streams.
    assert last_resid.shape == (d_vocab, d_model), f"last_resid.shape = {last_resid.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"
    key_tok_resid = (W_E + W_pos[:, None, :]) # (n_ctx, d_vocab, d_model). Dim 1 = possible residual streams.
    assert key_tok_resid.shape == (n_ctx, d_vocab, d_model), f"key_tok_resid.shape = {key_tok_resid.shape} != {(n_ctx, d_vocab, d_model)} = (n_ctx, d_vocab, d_model)"
    q = last_resid @ W_Q[0, 0, :, :] # (d_vocab, d_head).
    assert q.shape == (d_vocab, d_head), f"q.shape = {q.shape} != {(d_vocab, d_head)} = (d_vocab, d_head)"
    k = einsum('n_ctx d_vocab d_head, d_head d_model_k -> n_ctx d_model_k d_vocab', key_tok_resid, W_K[0, 0, :, :])
    assert k.shape == (n_ctx, d_head, d_vocab), f"k.shape = {k.shape} != {(n_ctx, d_head, d_vocab)} = (n_ctx, d_head, d_vocab)"
    x_scores = einsum('d_vocab_q d_head, n_ctx d_head d_vocab_k -> n_ctx d_vocab_q d_vocab_k', q, k)
    assert x_scores.shape == (n_ctx, d_vocab, d_vocab), f"x_scores.shape = {x_scores.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    # x_scores[pos, qt, kt] is the score from query token qt to key token kt at position pos

    return x_scores

# In[ ]:
@torch.no_grad()
def all_EVOU(model: HookedTransformer) -> TensorType["d_vocab", "d_vocab_out"]:
    """
    Returns all OV results, ignoring position, of shape (d_vocab, d_vocab_out)
    Complexity: O(d_vocab * (d_model^2 * d_head + d_head^2 * d_model + d_model^2 * d_vocab_out)) ~ O(d_vocab^2 * d_model^2)
    """
    W_E, W_O, W_V, W_U = model.W_E, model.W_O, model.W_V, model.W_U
    d_model, d_vocab, d_head, d_vocab_out = model.cfg.d_model, model.cfg.d_vocab, model.cfg.d_head, model.cfg.d_vocab_out
    assert W_E.shape == (d_vocab, d_model)
    assert W_O.shape == (1, 1, d_model, d_head)
    assert W_V.shape == (1, 1, d_model, d_head)
    assert W_U.shape == (d_model, d_vocab_out)

    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    assert EVOU.shape == (d_vocab, d_vocab_out), f"EVOU.shape = {EVOU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    return EVOU


# In[ ]:
@torch.no_grad()
def all_PVOU(model: HookedTransformer) -> TensorType["n_ctx", "d_vocab_out"]:
    """
    Returns all OV results, position only, of shape (n_ctx, d_vocab_out)
    Complexity: O(n_ctx * (d_model^2 * d_head + d_head^2 * d_model + d_model^2 * d_vocab_out)) ~ O(n_ctx * d_vocab * d_model^2)
    """
    W_pos, W_O, W_V, W_U = model.W_pos, model.W_O, model.W_V, model.W_U
    d_model, n_ctx, d_head, d_vocab_out = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_head, model.cfg.d_vocab_out
    assert W_pos.shape == (n_ctx, d_model)
    assert W_O.shape == (1, 1, d_model, d_head)
    assert W_V.shape == (1, 1, d_model, d_head)
    assert W_U.shape == (d_model, d_vocab_out)

    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (n_ctx, d_vocab_out). PVOU[i, j] is how copying at position i affects logit j.
    assert PVOU.shape == (n_ctx, d_vocab_out), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    return PVOU


# In[ ]:
@torch.no_grad()
def find_all_d_attention_scores(model: HookedTransformer, min_gap: int = 1) -> Union[TensorType["d_vocab_q", "d_vocab_k"], TensorType["d_vocab_q", "n_ctx_max", "n_ctx_non_max", "d_vocab_k_max", "d_vocab_k_nonmax"]]:
    """
    If input tokens are x, y, with x - y > min_gap, the minimum values of
    score(x) - score(y).

    Complexity: O(d_vocab * d_model^2 * n_ctx + d_vocab^min(3,n_ctx) * n_ctx^min(2,n_ctx-1))
    Returns: d_attention_score indexed by
        if n_ctx <= 2:
            (d_vocab_q, d_vocab_k)
        if n_ctx > 2:
            (d_vocab_q, n_ctx_max, n_ctx_non_max, d_vocab_k_max, d_vocab_k_nonmax)
    """
    n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    x_scores = all_attention_scores(model)
    assert x_scores.shape == (n_ctx, d_vocab, d_vocab), f"x_scores.shape = {x_scores.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    # x_scores[pos, qt, kt] is the score from query token qt to key token kt at position pos

    if n_ctx <= 2:
        # when there are only two cases, it must be the case that either the max is in the query slot, or the non-max is in the query slot
        scores = torch.zeros((d_vocab, d_vocab)) + float('inf')
        for q_tok in range(d_vocab):
            for k_tok in range(d_vocab):
                if math.abs(k_tok - q_tok) >= min_gap:
                    # q_tok is always in the last position
                    scores[q_tok, k_tok] = (x_scores[0, q_tok, k_tok].item() - x_scores[-1, q_tok, q_tok].item()) * np.sign(k_tok-q_tok)
    else:
        # when there are more than two cases, we need to consider all cases
        scores = torch.zeros((d_vocab, n_ctx, n_ctx, d_vocab, d_vocab)) + float('inf')
        for q_tok in range(d_vocab):
            for pos_of_max in range(n_ctx):
                for k_tok_max in range(d_vocab):
                    if pos_of_max == n_ctx - 1 and k_tok_max != q_tok: continue
                    for pos_of_non_max in range(n_ctx):
                        if pos_of_max == pos_of_non_max: continue
                        for k_tok_non_max in range(k_tok_max - (min_gap - 1)):
                            if pos_of_non_max == n_ctx - 1 and k_tok_non_max != q_tok: continue
                            scores[q_tok, pos_of_max, pos_of_non_max, k_tok_max, k_tok_non_max] = x_scores[pos_of_max, q_tok, k_tok_max].item() - x_scores[pos_of_non_max, q_tok, k_tok_non_max].item()

    return scores


# In[ ]:
@torch.no_grad()
def find_min_d_attention_score(model: HookedTransformer, min_gap: int = 1, reduce_over_query=False) -> Union[float, TensorType["d_vocab_q"]]:
    """
    If input tokens are x, y, with x - y > min_gap, the minimum value of
    score(x) - score(y).

    Complexity: O(d_vocab * d_model^2 * n_ctx + d_vocab^min(3,n_ctx) * n_ctx^min(2,n_ctx-1))
    Returns: float if reduce_over_query else torch.Tensor[d_vocab] (indexed by query token)
    """
    scores = find_all_d_attention_scores(model, min_gap=min_gap)
    while len(scores.shape) != 1:
        scores = scores.min(dim=-1).values
    if reduce_over_query:
        scores = scores.min(dim=0).values.item()
    return scores

# In[ ]:
@torch.no_grad()
def EU_PU_PVOU(model: HookedTransformer, attention_pattern: TensorType["batch", "n_ctx"]) -> TensorType["batch", "d_vocab_q", "d_vocab_out"]:
    """
    Calculates logits from EU, PU, and the positional part of the OV path for a given batch of attentions
    attention_pattern: (batch, n_ctx) # post softmax
    Returns: (batch, d_vocab_q, d_vocab_out)
    Complexity: O(d_vocab^2 * d_model + d_vocab^2 * d_model^2 + batch * n_ctx * d_vocab_out + batch * d_vocab^2)
    """
    n_ctx, d_vocab, d_vocab_out = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out
    batch, _ = attention_pattern.shape
    assert attention_pattern.shape == (batch, n_ctx), f"attention_post_softmax.shape = {attention_pattern.shape} != {(batch, n_ctx)} = (batch, n_ctx)"
    EUPU = EU_PU(model)
    assert EUPU.shape == (d_vocab, d_vocab_out), f"EUPU.shape = {EUPU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    PVOU = all_PVOU(model)
    assert PVOU.shape == (n_ctx, d_vocab_out), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    PVOU_scaled = attention_pattern @ PVOU
    assert PVOU_scaled.shape == (batch, d_vocab_out), f"PVOU_scaled.shape = {PVOU_scaled.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"
    result = EUPU[None, :, :] + PVOU_scaled[:, None, :]
    assert result.shape == (batch, d_vocab, d_vocab_out), f"result.shape = {result.shape} != {(batch, d_vocab, d_vocab_out)} = (batch, d_vocab, d_vocab_out)"

    return result

# In[ ]:
# @verify(UNIQUE, CONTINUOUS)
# class TokenType(Enum):
#     EXACT = enum.auto() # max, or within gap
#     BELOW_GAP = enum.auto()

# In[ ]:
@torch.no_grad()
def worst_PVOU_for(model: HookedTransformer, query_tok: int, max_tok: int, non_max_output_tok: int,
                   PVOU: Optional[TensorType["n_ctx", "d_vocab_out"]] = None,
                   attention_score_map: Optional[TensorType["n_ctx_k", "d_vocab_q", "d_vocab_k"]] = None,
                   optimize_max_query_comparison=True) -> TensorType["d_vocab_out"]:
    """
    Returns the PVOU with the worst (largest) value of PVOU[non_max_output_tok] - PVOU[max_tok], across all possible attention scalings for the query token and for token values <= max_tok.
    Complexity: O(PVOU + attention_score_map + n_ctx^2)
    Complexity: ~ O(n_ctx * d_vocab * d_model^2 (from PVOU) + d_vocab * d_head^2 * d_model * n_ctx (from attention_score_map) + n_ctx * log(n_ctx) (sorting) + n_ctx^2)
    Complexity: (for n_ctx=2) O(POVU + attention_score_map + n_ctx)
    N.B. Clever caching could reduce n_ctx^2 to n_ctx, leaving n_ctx log(n_ctx) from sorting as the dominant factor
    N.B. If optimize_max_query_comparison is set, and n_ctx is 2, then whenever query_tok != max_tok we know exactly what the sequence is and can just compute the attention
    """
    assert max_tok >= query_tok, f"max_tok = {max_tok} < {query_tok} = query_tok"
    n_ctx, d_vocab_out, d_vocab = model.cfg.n_ctx, model.cfg.d_vocab_out, model.cfg.d_vocab
    if PVOU is None: PVOU = all_PVOU(model)
    assert PVOU.shape == (n_ctx, d_vocab_out), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    if attention_score_map is None: attention_score_map = all_attention_scores(model)
    assert attention_score_map.shape == (n_ctx, d_vocab, d_vocab), f"attention_scores.shape = {attention_score_map.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    worst_attention_score = torch.zeros((n_ctx,))
    worst_attention_score[-1] = attention_score_map[-1, query_tok, query_tok]
    if n_ctx == 2 and optimize_max_query_comparison and query_tok != max_tok:
        worst_attention_score[0] = attention_score_map[0, query_tok, max_tok]
    else:
        # compute the min and max attention scores for each position and query token where the key token is <= max_tok
        min_attention_scores, max_attention_scores = attention_score_map[:-1, query_tok, :max_tok+1].min(dim=-1).values, attention_score_map[:-1, query_tok, :max_tok+1].max(dim=-1).values
        assert min_attention_scores.shape == (n_ctx-1,), f"min_attention_scores.shape = {min_attention_scores.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        assert max_attention_scores.shape == (n_ctx-1,), f"max_attention_scores.shape = {max_attention_scores.shape} != {(n_ctx-1,)} = (n_ctx-1,)"
        worst_attention_score[:-1] = min_attention_scores
        d_PVOU = PVOU[:, non_max_output_tok] - PVOU[:, max_tok]
        assert d_PVOU.shape == (n_ctx,), f"d_PVOU.shape = {d_PVOU.shape} != {(n_ctx,)} = (n_ctx,)"
        # sort d_PVOU in descending order
        _, d_PVOU_idxs = d_PVOU[:-1].sort(descending=True)
        for i in d_PVOU_idxs:
            # compare d_PVOU weighted by softmax of worst_attention_score for worst_attention_score[i] in (min_attention_scores[i], max_attention_scores[i])
            # set worst_attention_score[i] to whichever one is worse (more positive)
            # print(d_PVOU.shape, worst_attention_score.softmax(dim=-1).shape)
            min_d_PVOU = worst_attention_score.softmax(dim=-1) @ d_PVOU
            worst_attention_score[i] = max_attention_scores[i]
            max_d_PVOU = worst_attention_score.softmax(dim=-1) @ d_PVOU
            if min_d_PVOU > max_d_PVOU: worst_attention_score[i] = min_attention_scores[i]
            # print(i, min_attention_scores[i], worst_attention_score[i], max_attention_scores[i], min_d_PVOU, max_d_PVOU, d_PVOU[i])
    # return the PVOU for the worst_attention_score
    return worst_attention_score.softmax(dim=-1) @ PVOU

# In[ ]:
@torch.no_grad()
def worst_PVOU_gap_for(model: HookedTransformer, query_tok: int, max_tok: int, non_max_output_tok: int,
                       PVOU: Optional[TensorType["n_ctx", "d_vocab_out"]] = None,
                       attention_score_map: Optional[TensorType["n_ctx_k", "d_vocab_q", "d_vocab_k"]] = None) -> float:
    worst_PVOU = worst_PVOU_for(model, query_tok, max_tok, non_max_output_tok, PVOU=PVOU, attention_score_map=attention_score_map)
    return worst_PVOU[non_max_output_tok] - worst_PVOU[max_tok]

# In[ ]:
@torch.no_grad()
def all_worst_PVOU(model: HookedTransformer, tqdm=None) -> TensorType["d_vocab_q", "d_vocab_max", "d_vocab_out"]:
    """
    Returns the mixture of PVOUs with the worst (largest) value of PVOU[non_max_output_tok] - PVOU[max_tok], across all possible attention scalings for the query token and for token values <= max_tok.
    Complexity: O(PVOU + attention_score_map + n_ctx^2 * d_vocab^3)
    Complexity: ~ O(n_ctx * d_vocab * d_model^2 (from PVOU) + d_vocab * d_head^2 * d_model * n_ctx (from attention_score_map) + (n_ctx * log(n_ctx) (sorting) + n_ctx^2) * d_vocab^3)
    Complexity: (for n_ctx=2) O(PVOU + attention_score_map + n_ctx * d_vocab^3)
    N.B. Clever caching could reduce n_ctx^2 to n_ctx, leaving n_ctx log(n_ctx) * d_vocab^3 from sorting as the dominant factor.
    N.B. for max_of_{two,three}, this is maybe? worse than exhaustive enumeration (oops)
    """
    local_tqdm = make_local_tqdm(tqdm)
    n_ctx, d_vocab_out, d_vocab = model.cfg.n_ctx, model.cfg.d_vocab_out, model.cfg.d_vocab
    PVOU = all_PVOU(model)
    assert PVOU.shape == (n_ctx, d_vocab_out), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    attention_score_map = all_attention_scores(model)
    assert attention_score_map.shape == (n_ctx, d_vocab, d_vocab), f"attention_scores.shape = {attention_score_map.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    result = torch.zeros((d_vocab, d_vocab, d_vocab_out)) + float('nan')
    for non_max_tok, query_tok in local_tqdm(itertools.product(range(d_vocab), repeat=2), total=d_vocab**2):
        for max_tok in range(query_tok, d_vocab):
            result[query_tok, max_tok, non_max_tok] = worst_PVOU_gap_for(model, query_tok, max_tok, non_max_tok, PVOU=PVOU, attention_score_map=attention_score_map)

    return result

