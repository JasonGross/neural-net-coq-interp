# %%
import torch
import torch.nn as nn
import numpy as np
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
import circuitsvis as cv
from einops import reduce, repeat, rearrange, einsum
from pathlib import Path
from IPython import get_ipython

from coq_export_utils import strify
from analysis_utils import line, summarize, plot_QK_cosine_similarity, \
    analyze_svd, calculate_OV_of_pos_embed, calculate_attn, calculate_attn_by_pos, \
    calculate_copying, calculate_copying_with_pos, calculate_embed_and_pos_embed_overlap, \
    calculate_rowwise_embed_and_pos_embed_overlap, \
    calculate_embed_overlap, calculate_pos_embed_overlap, check_monotonicity, \
    plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from training_utils import compute_all_tokens, make_generator_from_data

import os, sys
from importlib import reload
import matplotlib.pyplot as plt


# %%

if __name__ == '__main__':
    PTH_BASE_PATH = Path(os.getcwd())
    PTH_BASE_PATH = PTH_BASE_PATH / 'trained-models'
    SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-two-simpler.pth'
    # SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-n-2023-09-01_01-30-10.pth'

    N_CTX = 2
    # N_CTX = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 123

    simpler_cfg = HookedTransformerConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_head=32,
        # n_ctx=2,
        n_ctx=N_CTX,
        d_vocab=64,
        seed=SEED,
        device=DEVICE,
        attn_only=True,
        normalization_type=None,
    )
    model = HookedTransformer(simpler_cfg, move_to_device=False)

    cached_data = torch.load(SIMPLER_MODEL_PTH_PATH)
    model.load_state_dict(cached_data['model'])
# %%

def min_effect_of_EU_PU(model) -> torch.Tensor:
    """
    Calculate the maximum negative effect of the EU and PU paths on the output.
    Complexity: O(d_vocab^2 * n_ctx * d_model)
    Return shape: (q_token,)
    """
    W_E, W_pos, W_U = model.W_E, model.W_pos, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_U.shape == (d_model, d_vocab)

    # The logit effect of token x and position p is given by the vector:
    #   logits(x, p) = (W_E[x] + W_pos[p]) @ W_U
    max_logit_deltas = torch.zeros((d_vocab, n_ctx))
    for x in range(d_vocab): # query token
        for p in range(n_ctx):
            logit_deltas = (W_E[x] + W_pos[p]) @ W_U # (d_vocab,)
            max_logit_deltas[x, p] = logit_deltas.max() - logit_deltas.min()

    result = -max_logit_deltas.max(dim=1).values # (q_token,)
    print(f"EU and PU paths have min effect of {result.min():.2f}")
    return result

if __name__ == '__main__':
    eu_pu = min_effect_of_EU_PU(model)

# %%


def find_d_score_coeff(model) -> float:
    """
    If input tokens are x, y, with x>y, finds the coefficient c such that
    score(x) - score(y) >= c * (x-y).

    Complexity: O(d_vocab * d_model^2 * n_ctx + d_vocab^2 * d_model * n_ctx)
    """
    W_E, W_pos, W_Q, W_K = model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_Q.shape == (1, 1, d_model, d_model)
    assert W_K.shape == (1, 1, d_model, d_model)

    points = []
    # We have two cases, x in position 0 and x in position 1.
    last_resid = (W_E + W_pos[-1]) # (d_vocab, d_model). Rows = possible residual streams.
    key_tok_resid = (W_E + W_pos[:, None, :]) # (n_ctx, d_model, d_vocab). Dim 1 = possible residual streams.
    q = last_resid @ W_Q[0, 0, :, :] # (d_vocab, d_model).
    # print(f"{key_tok_resid.shape=}")
    # print(f"{W_K.shape=}")
    k = einsum(key_tok_resid, W_K[0, 0, :, :], 'n_ctx d_vocab d_model, d_model d_model_k -> n_ctx d_model_k d_vocab')
    # k = key_tok_resid @ W_K[0, 0, :, :] # (n_ctx, d_model, d_vocab).
    x_scores = einsum(q, k, 'd_vocab_q d_model, n_ctx d_model d_vocab_k -> n_ctx d_vocab_q d_vocab_k')
    # print(f"{x_scores.shape=}")
    score_coeffs = torch.zeros((n_ctx - 1, d_vocab, d_vocab)) + 1000.
    global all_attn_scores
    all_attn_scores = x_scores
    # x_scores[pos, qt, kt] is the score from query token qt to key token kt at position pos.
    # q_tok is always in the last position; k_tok can be anywhere before it.
    for q_tok in range(d_vocab):
        for k_tok in range(d_vocab):
            for pos_of_max in range(n_ctx - 1):
                if k_tok != q_tok:
                    score_coeffs[pos_of_max, q_tok, k_tok] = (x_scores[pos_of_max, q_tok, k_tok].item() - x_scores[-1, q_tok, q_tok].item())/(k_tok-q_tok)

    # result = 0
    # print(f"Score coefficient: {result:.2f}")
    return score_coeffs.min(dim=2).values.min(dim=0).values # (d_vocab,)

if __name__ == '__main__':
    score_coeff = find_d_score_coeff(model)

# %%

def find_d_EVOU_PVOUx(model) -> float:
    """
    When x is maximum, the minimum logit effect of copying the correct residual stream.

    Complexity: O(d_vocab * d_model^2 + d_vocab^2 * d_model + ...)
    """
    W_E, W_pos, W_V, W_O, W_U = model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    assert W_U.shape == (d_model, d_vocab)

    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (n_ctx, d_vocab)

    # Worst case over all x of (effect on x - effect on y) where y != x. (could do y < x)
    EVOU_without_diag = EVOU - EVOU.diag().diag() * EVOU.max()
    min_EVOU_effect = (EVOU.diag() - EVOU_without_diag.max(dim=1).values)

    # Worst case over all positions of (effect on x - effect on y) where y <= x.
    PVOU_cummax = PVOU.cummax(dim=1).values # (n_ctx, d_vocab)
    min_PVOU_effect = (PVOU - PVOU_cummax).min(dim=0).values # (d_vocab,)

    # To improve this bound we take into account x-dependence of EVOU and PVOU.
    result = (min_EVOU_effect + min_PVOU_effect).min()
    print(f"Correct copying effect from:")
    print(f"EVOU: {min_EVOU_effect.min().item():.2f}, PVOU: {min_PVOU_effect.min().item():.2f}")
    print(f"Total: {result.item():.2f}")
    return result

if __name__ == '__main__':
    find_d_EVOU_PVOUx(model)
# %%
def find_d_EVOU_PVOUy(model) -> float:
    """
    When x is maximum, the minimum logit effect of copying the incorrect residual stream.
    Basically the max amount that copying y increases z more than x where z < x and y < x.
    """
    W_E, W_pos, W_V, W_O, W_U = model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    assert W_U.shape == (d_model, d_vocab)

    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    EVOU.names = ('qtok', 'ktok')
    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (n_ctx, d_vocab)

    # Our reasoning is simpler than for find_d_EVOU_PVOUx: just the largest logit delta from each query token
    EVOU_neg_range = -EVOU.max(dim='ktok').values + EVOU.min(dim='ktok').values # (d_vocab,) for each query token
    # Case 1: y = x - 1 e.g. 37, 38. We want EVOU[37, 38] - max_j EVOU[37, j].
    # EVOU_delta_case_1 = torch.diff(EVOU, dim=1).min(dim='ktok').values # (d_vocab,)
    EVOU_y_yp1 = torch.cat((EVOU.rename(None).diag(1), torch.tensor((1000,)))) # (d_vocab,)
    EVOU_y_smaller = EVOU.cummax(dim='ktok').values.diagonal() # (d_vocab,)
    EVOU_delta_case_1 = (EVOU_y_yp1[:, None] - EVOU_y_smaller) # (d_vocab, d_vocab)

    # Worst case over all positions of (effect on x - effect on y) where y <= x.
    PVOU_cummax_reverse = PVOU.flip(dims=(1,)).cummax(dim=1).values.flip(dims=(1,))
    min_PVOU_effect_case_2 = (PVOU - PVOU_cummax_reverse).min(dim=0).values # (d_vocab,)


    result_case_1 = (EVOU_delta_case_1 + min_PVOU_effect_case_2).min()
    result_case_2 = (EVOU_neg_range + min_PVOU_effect_case_2).min()
    print(f"Incorrect copying effect:")
    print(f"Case 1: {result_case_1.item():.2f}, Case 2: {result_case_2.item():.2f}")
    # result_case_1, result_case_2
    return result_case_1, result_case_2

if __name__ == '__main__':
    find_d_EVOU_PVOUy(model)

# %%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def slack(model):
    """
    Compute the minimum value of logit(x)-logit(y) when x > y.
    If this is >0, the model gets 100% accuracy.
    """

d_EU_PU = min_effect_of_EU_PU(model)
d_score_coeff = find_d_score_coeff(model)
worst_case_attn_pattern = torch.zeros((model.cfg.d_vocab, model.cfg.n_ctx))
worst_case_attn_pattern[:, 0] = d_score_coeff
worst_case_attn_pattern = torch.softmax(worst_case_attn_pattern, dim=1)[0]
print(f"Worst case attention weight for x: {worst_case_attn_pattern.min().item():.3f}")
d_EOVU_POVUx = find_d_EVOU_PVOUx(model)
d_EOVU_POVUy_c1, d_EOVU_POVUy_c2 = find_d_EVOU_PVOUy(model)

d_attn_out_U_case_1 = sigmoid(d_score_coeff*2) * d_EOVU_POVUx + (1 - sigmoid(d_score_coeff*2)) * d_EOVU_POVUy_c1
d_attn_out_U_case_2 = sigmoid(d_score_coeff * 2) * d_EOVU_POVUx + (1 - sigmoid(d_score_coeff * 2)) * d_EOVU_POVUy_c2
d_attn_out_U = torch.minimum(d_attn_out_U_case_1, d_attn_out_U_case_2)

result = (d_EU_PU + d_attn_out_U).min().item() # min over query token
print(f"Total model slack: {result:.2f}")
print(f"Model {'is' if result > 0 else 'is not'} proven 100% accurate.")

if __name__ == '__main__':
    slack(model)

# %%
# Bar graph of d_attn_out_U_case_1
plt.bar(range(64), d_attn_out_U_case_1.detach().numpy())
print(f"argmin={d_attn_out_U_case_1.argmin().item()}")
plt.show()
# %%
# Bar graph of d_attn_out_U_case_2
plt.bar(range(64), d_attn_out_U_case_2.detach().numpy())
print(f"argmin={d_attn_out_U_case_2.argmin().item()}")
plt.show()
# %%
# Bar graph of d_score_coeff
plt.bar(range(64), d_score_coeff.detach().numpy())
print(f"argmin={d_score_coeff.argmin().item()}")
plt.show()
# %%

score_coeff
# %%
all_attn_scores.shape

# 2d heatmap
plt.imshow(all_attn_scores[-1].detach().numpy())
plt.xlabel("key token")
plt.ylabel("query token")
# %%

d_38_37 = all_attn_scores[-1, 38, 38] - all_attn_scores[0, 38, 37]

# %%

# EVOU[37, 38] - EVOU[37, :].max()
# %%
