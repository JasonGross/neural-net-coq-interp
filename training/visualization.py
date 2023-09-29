# %%

# %%
import torch
import torch.nn as nn
import numpy as np
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
import circuitsvis as cv
from einops import einsum
from pathlib import Path
from IPython import get_ipython
import matplotlib.pyplot as plt
import seaborn as sns

from coq_export_utils import strify
from analysis_utils import line, summarize, plot_QK_cosine_similarity, \
    analyze_svd, calculate_OV_of_pos_embed, calculate_attn, calculate_attn_by_pos, \
    calculate_copying, calculate_copying_with_pos, calculate_embed_and_pos_embed_overlap, \
    calculate_rowwise_embed_and_pos_embed_overlap, \
    calculate_embed_overlap, calculate_pos_embed_overlap, check_monotonicity, \
    plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from interp_max_utils import logit_delta
from training_utils import compute_all_tokens, make_testset_trainset, make_generator_from_data

import os, sys
from importlib import reload

# %%

if __name__ == '__main__':
    PTH_BASE_PATH = Path(os.getcwd())
    PTH_BASE_PATH = PTH_BASE_PATH / 'trained-models'
    # SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-two-simpler.pth'
    # SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-n.pth'
    SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-n-2023-09-01_01-30-10.pth'

    # N_CTX = 2
    N_CTX = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 123

    simpler_cfg = HookedTransformerConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        d_head=32,
        n_ctx=N_CTX,
        d_vocab=64,
        seed=SEED,
        device=DEVICE,
        attn_only=True,
        normalization_type=None,
    )
    model = HookedTransformer(simpler_cfg, move_to_device=False).cpu()

    cached_data = torch.load(SIMPLER_MODEL_PTH_PATH)
    model.load_state_dict(cached_data['model'])

# %%

input = torch.tensor([17, 12, 19, 17, 10])

def visualize_on_input(model, input):
    W_E, W_pos, W_V, W_O, W_U = [arr.detach().numpy() for arr in (model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U)]
    input_numpy = input.detach().numpy()
    # def visualize(model, input):
    logits, cache = model.run_with_cache(input)
    topk = torch.topk(logits[0, -1], k=6, dim=-1)
    topk_vals, topk_idxs = topk.values.detach().numpy(), topk.indices.detach().numpy()

    true_max = input.max().item()

    # Make 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Input: {input_numpy}, True max: {true_max}')

    xlabels_10 = [f'{t}@{i}' for i, t in enumerate(input_numpy)]

    # Bottom right: topk_vals as bar graph
    bar_colors_topk = ['green' if i == true_max else 'red' if i in input else 'grey' for i in topk_idxs]
    sns.barplot(y=topk_idxs, x=topk_vals, ax=axs[1, 1], palette=bar_colors_topk, order=topk_idxs, orient='h')
    axs[1, 1].set_title('Top k logits')

    # Top left: Attention pattern as bar graph
    attn = cache['attn', 0][0,0,-1].detach().numpy()
    bar_colors_attn = ['green' if input[i]== true_max else 'red' for i in range(len(attn))]
    sns.barplot(x=xlabels_10, y=attn, ax=axs[0, 0],palette=bar_colors_attn)
    axs[0, 0].set_title('Attention weights')

    # Bottom left: Heatmap of attention contributions ((E+P)VOU) to top k logits, NOT weighted by attention
    resid_pre = cache['resid_pre', 0][0] # (n_ctx, d_model)
    contribs = (resid_pre @ W_V @ W_O @ W_U)[0, 0].detach().numpy() # (n_ctx, d_vocab)
    contribs_topk = contribs[:, topk_idxs].T # (k, n_ctx)
    sns.heatmap(contribs_topk, ax=axs[1, 0], cmap='PuOr', center=0, yticklabels=topk_idxs, xticklabels=xlabels_10)
    axs[1, 0].set_title('Attention contributions to top k logits')
    axs[1, 0].set_ylabel('To logit')
    axs[1, 0].set_xlabel('From token')

    # Top right: source of difference between logit of true max and highest wrong logit
    # Includes attention contributions, EU, and PU
    highest_wrong = topk_idxs[0] if topk_idxs[0] != true_max else topk_idxs[1]
    eu_contrib = (W_E[input[-1], :] @ (W_U[:, true_max] - W_U[:, highest_wrong])).item()
    pu_contrib = (W_pos[-1, :] @ (W_U[:, true_max] - W_U[:, highest_wrong])).item()
    attn_contribs = attn * (contribs[:, true_max] - contribs[:, highest_wrong])
    bar_colors_diff = ['blue', 'yellow'] + ['green' if i == true_max else 'red' if i==highest_wrong else 'black' for i in input_numpy]
    sns.barplot(x=['EU', 'PU'] + xlabels_10, y=[eu_contrib, pu_contrib] + list(attn_contribs), ax=axs[0, 1],
                palette=bar_colors_diff)
    axs[0, 1].set_title(f'Sources of Δ between true max and highest wrong {highest_wrong}')

    fig.show()

visualize_on_input(model, torch.tensor([39, 39, 39, 39, 42]))
# %%


for test_case in [
        # [ 4,  5, 15, 12,  4],
        # [37, 37, 38,  4, 19],
        # [35, 39,  3, 39, 42],
        # [17, 12, 19, 17, 10],
        # [38, 37, 25, 37, 19],
        # [30, 35, 39, 39, 42],
        # [40, 24, 37, 37, 25],
        # [31, 33, 30, 30, 25],
        # [35, 39,  3, 39, 42],
        [17, 12, 19, 17, 10],
        [38, 37, 25, 37, 19],
        [30, 35, 39, 39, 42],
        [40, 24, 37, 37, 25],
        [31, 33, 30, 30, 25],
        [35, 37, 40, 37, 32],
        [39, 24, 36, 39, 42],
        [47, 48, 47, 47, 23],
        [37, 37, 40, 35, 25],
        [16, 32, 33, 32, 31],
        [ 4,  4,  6,  4,  0],
        # [ 8,  1, 15, 12,  4],
        # [ 2,  1, 15, 12,  9],
        # [12, 13, 12, 12, 11],
        # [ 5, 40, 37, 37, 32],
        # [37, 14, 38, 37, 19],
        # [38, 22, 37, 37,  5],
        # [47, 48, 37, 47, 23],
        # [37, 37, 24, 38, 19],
        # [37, 40, 32, 37, 25],
        # [ 6, 30, 33, 30, 25],
        # [38, 37, 23, 37, 19],
        # [35, 35,  9, 38, 24],
        # [12,  7, 15,  9,  8],
        # [ 6, 17, 19, 17, 10],
        # [27, 17, 33, 27, 26],
        # [38, 29, 37, 37,  5],
        # [35, 39, 20, 39, 42],
        # [ 2,  6, 15, 12,  8],
        # [19, 27, 33, 27, 26],
        # [31, 31, 32, 31, 20],
        # [ 3,  3,  5,  2,  0],
        # [ 4,  2,  5,  4,  1],
        # [12, 12, 15,  7, 11],
        # [35, 35, 42, 11, 24],
        # [11,  0, 14, 11,  9],
        # [40, 37, 37, 11, 32],
        # [32, 31, 31, 26,  7],
        # [ 3, 37, 40, 37, 32],
        # [19, 37, 38, 37,  5],
        # [35, 35,  4, 38, 24],
        # [37, 22, 38, 37,  5],
        # [33, 32, 32, 32, 24],
        # [52, 52, 52, 53, 15], 
        # [37, 37, 37, 37, 25], 
        # [40, 40, 40, 40, 25], 
        # [35, 35, 35, 35, 25], [40, 37, 37, 35, 25], 
        # [37, 37, 40, 35, 25], 
        # [35, 37, 40, 37, 25], 
        # [37, 40, 35, 37, 25]
        ]: 
    visualize_on_input(model, torch.tensor(test_case))
# %%

def plot_weight_info(model):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Left: Attention pattern heatmap
    attn_weights = model.W_E @ model.W_Q[0,0] @ model.W_K[0,0].T @ model.W_E.T
    sns.heatmap(attn_weights.detach().numpy(), cmap='PuOr', center=0, ax=axs[0])
    axs[0].set_title('Attention pattern')
    axs[0].set_xlabel('To token')
    axs[0].set_ylabel('From token')

    # Right: EVOU
    copying_weights = model.W_E @ model.W_V[0,0] @ model.W_O[0,0] @ model.W_U
    sns.heatmap(copying_weights.detach().numpy(), cmap='PuOr', center=0, ax=axs[1])
    axs[1].set_title('EVOU')
    axs[1].set_xlabel('To token')
    axs[1].set_ylabel('From token')


plot_weight_info(model)

# %%

# TODO make EVOU histogram plots somehow
# evou = all_EVOU(model).detach()
# # add axis names
# evou = evou.rename("qt", "kt")
# evou_diag = torch.diagonal(evou)[:, None].rename_("qt", 'kt')
# devou = evou - evou_diag

# # %%
# # histogram of diagonal, with text showing min, med, max
# plt.hist(evou_diag)
# plt.title("Histogram of diagonal of EVOU")
# plt.text(evou_diag.min(), 10, f"min: {evou_diag.min():.2f}\nmed: {np.median(evou_diag):.2f}\nmax: {evou_diag.max():.2f}")
# plt.show()

# %%
copying_weights = model.W_E @ model.W_V[0,0] @ model.W_O[0,0] @ model.W_U
copying_weights.topk

# %%
