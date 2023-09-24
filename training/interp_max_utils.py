# NOTE that this file is synchronized with theories/TrainingComputations/interp_max_utils.v
# In[ ]:
from typing import Any, Dict, Optional, Union
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
    fwd_pass = (n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
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
def EU_PU(model: HookedTransformer, renderer=None, pos: int = -1) -> torch.Tensor:
    """
    Calculates logits from just the EU and PU paths in position pos.
    Complexity: O(d_vocab^2 * d_model)
    Return shape: (d_vocab,) (indexed by query token)
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
def find_min_d_attention_score(model: HookedTransformer, min_gap: int = 1, reduce_over_query=False) -> Union[float, torch.Tensor]:
    """
    If input tokens are x, y, with x - y > min_gap, the minimum value of
    score(x) - score(y).

    Complexity: O(d_vocab * d_model^2 * n_ctx + d_vocab^min(3,n_ctx) * n_ctx^min(2,n_ctx-1))
    Returns: float if reduce_over_query else torch.Tensor[d_vocab] (indexed by query token)
    """
    W_E, W_pos, W_Q, W_K = model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_Q.shape == (1, 1, d_model, d_model)
    assert W_K.shape == (1, 1, d_model, d_model)

    last_resid = (W_E + W_pos[-1]) # (d_vocab, d_model). Rows = possible residual streams.
    assert last_resid.shape == (d_vocab, d_model), f"last_resid.shape = {last_resid.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"
    key_tok_resid = (W_E + W_pos[:, None, :]) # (n_ctx, d_vocab, d_model). Dim 1 = possible residual streams.
    assert key_tok_resid.shape == (n_ctx, d_vocab, d_model), f"key_tok_resid.shape = {key_tok_resid.shape} != {(n_ctx, d_vocab, d_model)} = (n_ctx, d_vocab, d_model)"
    q = last_resid @ W_Q[0, 0, :, :] # (d_vocab, d_model).
    assert q.shape == (d_vocab, d_model), f"q.shape = {q.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"
    k = einsum(key_tok_resid, W_K[0, 0, :, :], 'n_ctx d_vocab d_model, d_model d_model_k -> n_ctx d_model_k d_vocab')
    assert k.shape == (n_ctx, d_model, d_vocab), f"k.shape = {k.shape} != {(n_ctx, d_model, d_vocab)} = (n_ctx, d_model, d_vocab)"
    x_scores = einsum(q, k, 'd_vocab_q d_model, n_ctx d_model d_vocab_k -> n_ctx d_vocab_q d_vocab_k')
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
        scores = scores.min(dim=-1).values
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
        while len(scores.shape) != 1:
            scores = scores.min(dim=-1).values
    if reduce_over_query:
        scores = scores.min(dim=0).values.item()

    return scores