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

from analysis_utils import line, summarize, plot_QK_cosine_similarity, \
    analyze_svd, calculate_OV_of_pos_embed, calculate_attn, calculate_attn_by_pos, \
    calculate_copying, calculate_copying_with_pos, calculate_embed_and_pos_embed_overlap, \
    calculate_rowwise_embed_and_pos_embed_overlap, \
    calculate_embed_overlap, calculate_pos_embed_overlap, check_monotonicity, \
    plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity

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
    SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'neural-net-coq-interp-max-5-epochs-50000-5-epochs-50000-2023-09-13_20-47-42.pth'

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

for test_case in [[37, 37, 38,  4, 19]]: 
    print(test_case)
    for i, v in enumerate(model(torch.tensor([test_case]))[0,-1,:]):
        print(f"{i}:{v.item()}")

# %%

DEVICE = 'cpu'
model = model.to(DEVICE)
test_cases = torch.randint(0, 64, size=(10000000, 5)).to(DEVICE)
predictions = model(test_cases)[:,-1,:]
true_max_idxs = test_cases.max(dim=-1).values
prediction_max_idxs = predictions.max(dim=-1).indices

wrong_tests = test_cases[true_max_idxs != prediction_max_idxs]
print(wrong_tests)

        
# %%

for test_case in [[ 4,  5, 15, 12,  4],
        [37, 37, 38,  4, 19],
        [35, 39,  3, 39, 42],
        [17, 12, 19, 17, 10],
        [38, 37, 25, 37, 19],
        [30, 35, 39, 39, 42],
        [40, 24, 37, 37, 25],
        [31, 33, 30, 30, 25],[35, 39,  3, 39, 42],
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
        [ 8,  1, 15, 12,  4],
        [ 2,  1, 15, 12,  9],
        [12, 13, 12, 12, 11],
        [ 5, 40, 37, 37, 32],
        [37, 14, 38, 37, 19],
        [38, 22, 37, 37,  5],
        [47, 48, 37, 47, 23],
        [37, 37, 24, 38, 19],
        [37, 40, 32, 37, 25],
        [ 6, 30, 33, 30, 25],
        [38, 37, 23, 37, 19],
        [35, 35,  9, 38, 24],
        [12,  7, 15,  9,  8],
        [ 6, 17, 19, 17, 10],
        [27, 17, 33, 27, 26],
        [38, 29, 37, 37,  5],
        [35, 39, 20, 39, 42],
        [ 2,  6, 15, 12,  8],
        [19, 27, 33, 27, 26],
        [31, 31, 32, 31, 20],
        [ 3,  3,  5,  2,  0],
        [ 4,  2,  5,  4,  1],
        [12, 12, 15,  7, 11],
        [35, 35, 42, 11, 24],
        [11,  0, 14, 11,  9],
        [40, 37, 37, 11, 32],
        [32, 31, 31, 26,  7],
        [ 3, 37, 40, 37, 32],
        [19, 37, 38, 37,  5],
        [35, 35,  4, 38, 24],
        [37, 22, 38, 37,  5],
        [33, 32, 32, 32, 24],
        [52, 52, 52, 53, 15], 
        [37, 37, 37, 37, 25], 
        [40, 40, 40, 40, 25], 
        [35, 35, 35, 35, 25], [40, 37, 37, 35, 25], 
        [37, 37, 40, 35, 25], 
        [35, 37, 40, 37, 25], 
        [37, 40, 35, 37, 25]]: 
    test_case = torch.tensor([test_case])
    true_max_idx = test_case.max().item()
    predicted_max = model(test_case)[0,-1,:].argmax().item()
    print(f"{test_case}: max: {true_max_idx}, predicted: {predicted_max}")
    

# %%
gres = calculate_copying(model)


# %%
for test_case in [[35, 37, 40, 37, 32],
        [39, 24, 36, 39, 42],
        [47, 48, 47, 47, 23],
        [37, 37, 40, 35, 25],
        [16, 32, 33, 32, 31],
        [ 4,  4,  6,  4,  0],
        [ 8,  1, 15, 12,  4],
        [ 2,  1, 15, 12,  9],
        [12, 13, 12, 12, 11],
        [ 5, 40, 37, 37, 32],
        [37, 14, 38, 37, 19],
        [38, 22, 37, 37,  5],
        [47, 48, 37, 47, 23],
        [37, 37, 24, 38, 19],
        [37, 40, 32, 37, 25],
        [ 6, 30, 33, 30, 25],
        [38, 37, 23, 37, 19],
        [35, 35,  9, 38, 24],
        [12,  7, 15,  9,  8],
        [ 6, 17, 19, 17, 10],
        [27, 17, 33, 27, 26],
        [38, 29, 37, 37,  5],
        [35, 39, 20, 39, 42],
        [ 2,  6, 15, 12,  8],
        [19, 27, 33, 27, 26],
        [31, 31, 32, 31, 20],
        [ 3,  3,  5,  2,  0],
        [ 4,  2,  5,  4,  1],
        [12, 12, 15,  7, 11],
        [35, 35, 42, 11, 24],
        [11,  0, 14, 11,  9],
        [40, 37, 37, 11, 32],
        [32, 31, 31, 26,  7],
        [ 3, 37, 40, 37, 32],
        [19, 37, 38, 37,  5],
        [35, 35,  4, 38, 24],
        [37, 22, 38, 37,  5],
        [33, 32, 32, 32, 24],
        [52, 52, 52, 53, 15],
        [39,  7, 39, 39, 42],
        [ 7, 18, 19, 18, 10],
        [37, 36, 40, 35, 24]]: 
    test_case = torch.tensor([test_case])
    true_max_idx = test_case.max().item()
    predicted_max = model(test_case)[0,-1,:].argmax().item()
    print(f"{test_case}: max: {true_max_idx}, predicted: {predicted_max}")
    

# %%
#[19, 27, 33, 27, 26]
# %%
