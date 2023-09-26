# NOTE that this file is synchronized with theories/TrainingComputations/interp_max_utils.v
# In[ ]:
from typing import Any, Dict, Optional, Union
from torchtyping import TensorType
from enum import Enum, verify, UNIQUE, CONTINUOUS
import enum
import einops
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
from training_utils import compute_all_tokens
import math

# In[ ]:

def logit_delta(model: HookedTransformer, renderer=None, histogram_all_incorrect_logit_differences=False, return_summary=False) -> Union[float, Dict[str, Any]]:
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max.
    Complexity: O(d_vocab^n_ctx * fwd_pass)
    fwd_pass = O(n_ctx * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_hidden * 2 + n_ctx * d_hidden^2 + n_ctx * d_model^2 * d_hidden + n_ctx * d_hidden^2 * d_model + n_ctx * d_model + n_ctx * d_model^2 * d_vocab)
    n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """
    n_ctx, d_vocab, d_vocab_out, d_model = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out, model.cfg.d_model

    all_tokens = compute_all_tokens(model=model)
    assert all_tokens.shape == (d_vocab**n_ctx, n_ctx), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits = model(all_tokens)[:,-1,:].detach().cpu()
    assert predicted_logits.shape == (d_vocab**n_ctx, d_vocab_out), f"predicted_logits.shape = {predicted_logits.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"

    # Extract statistics for each row
    # Use values in all_tokens as indices to gather correct logits
    indices_of_max = all_tokens.max(dim=-1, keepdim=True).values
    assert indices_of_max.shape == (d_vocab**n_ctx, 1), f"indices_of_max.shape = {indices_of_max.shape} != {(d_vocab**n_ctx, 1)} = (d_vocab**n_ctx, 1)"
    correct_logits = torch.gather(predicted_logits, -1, indices_of_max)
    assert correct_logits.shape == (d_vocab**n_ctx, 1), f"correct_logits.shape = {correct_logits.shape} != {(d_vocab**n_ctx, 1)} = (d_vocab**n_ctx, 1)"
    logits_above_correct = correct_logits - predicted_logits
    assert logits_above_correct.shape == (d_vocab**n_ctx, d_vocab_out), f"logits_above_correct.shape = {logits_above_correct.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"
    # replace correct logit indices with large number so that they don't get picked up by the min
    logits_above_correct[torch.arange(logits_above_correct.shape[0]), indices_of_max.squeeze()] = float('inf')
    min_incorrect_logit = logits_above_correct.min(dim=-1).values
    assert min_incorrect_logit.shape == (d_vocab**n_ctx,), f"min_incorrect_logit.shape = {min_incorrect_logit.shape} != {(d_vocab**n_ctx,)} = (d_vocab**n_ctx,)"

    if histogram_all_incorrect_logit_differences:
        all_incorrect_logits = logits_above_correct[logits_above_correct != float('inf')]
        summarize(all_incorrect_logits, name='all incorrect logit differences', histogram=True, renderer=renderer)

    if return_summary:
        return summarize(min_incorrect_logit, name='min(correct logit - incorrect logit)', renderer=renderer, histogram=True)

    else:
        return min_incorrect_logit.min().item()


# In[ ]:
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
def EU_PU_PVOU(model: HookedTransformer, attention_post_softmax: TensorType["batch", "n_ctx"]) -> TensorType["btach", "d_vocab_q", "d_vocab_out"]:
    """
    Calculates logits from EU, PU, and the positional part of the OV path for a given batch of attentions
    attention_post_softmax: (batch, n_ctx)
    Returns: (batch, d_vocab_q, d_vocab_out)
    Complexity: O(d_vocab^2 * d_model + d_vocab^2 * d_model^2 + batch * n_ctx * d_vocab_out + batch * d_vocab^2)
    """
    n_ctx, d_vocab, d_vocab_out = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out
    batch, _ = attention_post_softmax.shape
    assert attention_post_softmax.shape == (batch, n_ctx), f"attention_post_softmax.shape = {attention_post_softmax.shape} != {(batch, n_ctx)} = (batch, n_ctx)"
    EUPU = EU_PU(model)
    assert EUPU.shape == (d_vocab, d_vocab_out), f"EUPU.shape = {EUPU.shape} != {(d_vocab, d_vocab_out)} = (d_vocab, d_vocab_out)"
    PVOU = all_PVOU(model)
    assert PVOU.shape == (n_ctx, d_vocab_out), f"PVOU.shape = {PVOU.shape} != {(n_ctx, d_vocab_out)} = (n_ctx, d_vocab_out)"
    PVOU_scaled = attention_post_softmax @ PVOU
    assert PVOU_scaled.shape == (batch, d_vocab_out), f"PVOU_scaled.shape = {PVOU_scaled.shape} != {(batch, d_vocab_out)} = (batch, d_vocab_out)"
    result = EUPU[None, :, :] + PVOU_scaled[:, None, :]
    assert result.shape == (batch, d_vocab, d_vocab_out), f"result.shape = {result.shape} != {(batch, d_vocab, d_vocab_out)} = (batch, d_vocab, d_vocab_out)"

    return result

# In[ ]:
@verify(UNIQUE, CONTINUOUS)
class TokenType(Enum):
    EXACT = enum.auto() # max, or within gap
    BELOW_GAP = enum.auto()

# In[ ]:
def compute_heuristic_independence_attention_copying(model: HookedTransformer, min_gap: int = 1) -> Dict[int, TensorType["batch", "d_vocab_out"]]:
    """
    Assuming that attention paid to the non-max tokens is independent of the copying behavior on non-max tokens which are at least min_gap away, computes the logit outputs, grouped by gap
    Returns: Dict[gap, Tensor[batch, d_vocab_out]]
    Complexity:
    """
    n_ctx, d_vocab, d_vocab_out, d_model = model.cfg.n_ctx, model.cfg.d_vocab, model.cfg.d_vocab_out, model.cfg.d_model

    all_tokens = compute_all_tokens(model=model)
    assert all_tokens.shape == (d_vocab**n_ctx, n_ctx), f"all_tokens.shape = {all_tokens.shape} != {(d_vocab**n_ctx, n_ctx)} = (d_vocab**n_ctx, n_ctx)"
    predicted_logits, cache = model.run_with_cache(all_tokens)
    predicted_logits = predicted_logits[:,-1,:].detach().cpu()
    assert predicted_logits.shape == (d_vocab**n_ctx, d_vocab_out), f"predicted_logits.shape = {predicted_logits.shape} != {(d_vocab**n_ctx, d_vocab_out)} = (d_vocab**n_ctx, d_vocab_out)"

    return cache

# In[ ]:
