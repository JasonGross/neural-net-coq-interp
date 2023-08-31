#!/usr/bin/env python
# coding: utf-8

# # How A Transformer Takes Max (For Formalization in Proof Assistants)

# # Setup
# (No need to read)

# In[ ]:


TRAIN_MODEL_IF_CANT_LOAD = True # @param
ALWAYS_TRAIN_MODEL = False # @param
OVERWRITE_DATA = False # @param
SAVE_IN_GOOGLE_DRIVE = True # @param


# In[ ]:


# get_ipython().run_line_magic('pip', 'install transformer_lens scipy scikit-learn einops circuitsvis kaleido # git+https://github.com/neelnanda-io/neel-plotly.git')


# In[ ]:


import einops
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from analysis_utils import imshow, line
from coq_export_utils import strify
from analysis_utils import pm_range, summarize
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model
from training import compute_all_tokens, get_data, large_data_gen, make_generator_from_data
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
import circuitsvis as cv
from fancy_einsum import einsum
import dataclasses
from pathlib import Path
from typing import Optional

import os, sys

from scipy.optimize import curve_fit

from training.analysis_utils import calculate_OV_of_pos_embed, calculate_embed_and_pos_embed_overlap, calculate_embed_overlap, calculate_pos_embed_overlap, compute_slack


# In[ ]:


try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook")
    # from IPython import get_ipython

    # ipython = get_ipython()
    # # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    # ipython.magic("load_ext autoreload")
    # ipython.magic("autoreload 2")


# In[ ]:


# # Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
# import plotly.io as pio
# if IN_COLAB:
#     pio.renderers.default = "colab"
# else:
#     pio.renderers.default = "notebook_connected"
# print(f"Using renderer: {pio.renderers.default}")


# In[ ]:


print(pio.renderers)
print(pio.renderers.default)


# In[ ]:

# convert model to format for Coq
# In[ ]:


# In[ ]:


# where we save the model
if IN_COLAB:
    if SAVE_IN_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive/')
        PTH_BASE_PATH = Path('/content/drive/MyDrive/Colab Notebooks/')
    else:
        PTH_BASE_PATH = Path("/workspace/_scratch/")
else:
    PTH_BASE_PATH = Path(os.getcwd())

PTH_BASE_PATH = PTH_BASE_PATH / 'transformer-takes-max'

if not os.path.exists(PTH_BASE_PATH):
    os.makedirs(PTH_BASE_PATH)


# Summarizing data

# In[ ]:


# # Introduction
# 
# This Colab tackles one of [Neel Nanda's 200 open problems in mechanistic interpretability](https://www.alignmentforum.org/s/yivyHaCAmMJ3CqSyj/p/ejtFsvyhRkMofKAFy), by exploring how a transformer takes the max over a list.
# 
# I used a one layer attention-only transformer with a single head and found that its head learns to copy the largest value when at the last position. This is similiar to what the the sorting head discovered by [Bagiński and Kolly](https://github.com/MatthewBaggins/one-attention-head-is-all-you-need/) (B & K) does when encountering the mid-token (expcept, they are sorting lists in increasing order and hence the mid-token pays attention the lowest value).
# 
# The first few sections deal with setup, data-generation, and model training. Skip to [Interpretability](#scrollTo=D0RLiGYW-ZQK) to read the main analysis.
# 
# Somewhat surprisingly, for very short sequence lengths the model learns to also predict the max for subsequences, though this behavior gets less pronounced [for longer sequences](#scrollTo=OvcWD6LgBZy7).
# 
# To tell which token is the last in the input sequence the model needs the positional embedding. Hence, when [removing the positional embedding](#scrollTo=wUGXj6S_pP_X), the model always learns to take the max over every subsequence (even for longer input lengths), even if it is only trained to take the max over the whole sequence.

# # Hyperparameters

# In[ ]:


N_LAYERS = 1
N_HEADS = 1
D_MODEL = 32
D_HEAD = 32
D_MLP = None

D_VOCAB = 64


# # Constants

# In[ ]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123


# In[ ]:


MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-two.pth'
SIMPLER_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-two-simpler.pth'


# Verify that loss is low and accuracy is high for appropriate logits.

# In[ ]:


tokens = torch.randint(0, 64, (32, 2))
# tokens = torch.hstack((tokens, torch.max(tokens, dim=1)[0].unsqueeze(-1)))


# In[ ]:


logits = torch.rand((32, 2, 64))
logits[list(range(32)),-1,torch.max(tokens, dim=1)[0]] = 10.


# In[ ]:


loss_fn(logits, tokens)


# In[ ]:


acc_fn(logits, tokens)


# # Simpler Model Setup

# A simple one-layer attention only model with a context length of 2, optimized for provability by removing norms, etc.

# In[ ]:


simpler_cfg = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=2,
    d_vocab=D_VOCAB,
    seed=SEED,
    device=DEVICE,
    attn_only=True,
    normalization_type=None,
)
simpler_model = HookedTransformer(simpler_cfg, move_to_device=True)


# Disable the biases, as we don't need them for this task and it makes things easier to interpret.

# In[ ]:


for name, param in simpler_model.named_parameters():
    if "b_" in name:
        param.requires_grad = False


# # Simpler Training

# ## Training Loop

# We train for 500 epochs with 10 batches of 128 tokens per epoch. (This is somewhat arbitrary as the intention was mostly to get a good model quickly.)

# In[ ]:


# In[ ]:


ALWAYS_TRAIN_MODEL=False
OVERWRITE_DATA=False


# In[ ]:


TRAIN_MODEL = ALWAYS_TRAIN_MODEL
if not ALWAYS_TRAIN_MODEL:
    try:
        cached_data = torch.load(SIMPLER_MODEL_PTH_PATH)
        simpler_model.load_state_dict(cached_data['model'])
        #model_checkpoints = cached_data["checkpoints"]
        #checkpoint_epochs = cached_data["checkpoint_epochs"]
        #test_losses = cached_data['test_losses']
        simpler_train_losses = cached_data['train_losses']
        #train_indices = cached_data["train_indices"]
        #test_indices = cached_data["test_indices"]
    except Exception as e:
        print(e)
        TRAIN_MODEL = TRAIN_MODEL_IF_CANT_LOAD


# In[ ]:


if TRAIN_MODEL:
    simpler_train_losses = train_model(simpler_model, n_epochs=1500, batch_size=128, sequence_length=2, force_adjacent=True)


# In[ ]:


if TRAIN_MODEL:
    data = {
                "model":simpler_model.state_dict(),
                "config": simpler_model.cfg,
                "train_losses": simpler_train_losses,
            }
    if OVERWRITE_DATA or not os.path.exists(SIMPLER_MODEL_PTH_PATH):
        torch.save(
            data,
            SIMPLER_MODEL_PTH_PATH)
    else:
        print(f'WARNING: Not overwriting {SIMPLER_MODEL_PTH_PATH} because it already exists.')
        ext = 0
        while os.path.exists(f"{SIMPLER_MODEL_PTH_PATH}.{ext}"):
            ext += 1
        torch.save(
            data,
            f"{SIMPLER_MODEL_PTH_PATH}.{ext}")
        print(f'WARNING: Wrote to {SIMPLER_MODEL_PTH_PATH}.{ext} instead.')


# As we can see accuracy is high and loss is low.

# In[ ]:


line(simpler_train_losses, xaxis="Epoch", yaxis="Loss")


EXPORT_TO_COQ = True
if EXPORT_TO_COQ:
    coq_export_params(simpler_model)

# In[ ]:


all_integers = compute_all_tokens(simpler_model)
all_integers.shape
#model(torch.tensor([0, 1]))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'all_integers_result = simpler_model(all_integers)\n')


# In[ ]:


print(f"loss: {loss_fn(all_integers_result, all_integers)}")
print(f"acc: {acc_fn(all_integers_result, all_integers)}")


# In[ ]:


all_integers_ans = all_integers_result[:,-1]
ans = all_integers_ans.argmax(dim=-1)
expected = all_integers.max(dim=-1).values
alt_expected = all_integers.min(dim=-1).values
correct_idxs = (ans == expected)
very_wrong_idxs = ~((ans == expected) | (ans == alt_expected))
print(all_integers[~correct_idxs], very_wrong_idxs.sum())
# In[ ]:


compute_slack(simpler_model)#, renderer='png')


# In[ ]:


calculate_embed_overlap(simpler_model)


# In[ ]:


calculate_pos_embed_overlap(simpler_model)


# In[ ]:


calculate_embed_and_pos_embed_overlap(simpler_model)


# In[ ]:


calculate_OV_of_pos_embed(simpler_model)


# ## Copying: W_E @ W_V @ W_O @ W_U

# In[ ]:


def calculate_copying(model: HookedTransformer, renderer=None):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_E @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    res_diag = res.diag()
    res_off_diagonal = res[torch.eye(d_vocab) == 0]
    centered = -res + res.diag()[:, None]
    nonzero_centered = centered[torch.eye(d_vocab) == 0]
    imshow(res, title='W_E @ W_V @ W_O @ W_U', renderer=renderer,
           xaxis="logit affected", yaxis="input token")
    imshow(centered, title='copying.diag()[:,None] - copying', renderer=renderer)
    line(res.diag(), title='copying.diag()', renderer=renderer)
    # take svd of res
    u, s, v = torch.svd(res)
    # plot singular values
    line(s, title='singular values of copying', renderer=renderer)
    # plot u, v
    imshow(u, title='u', renderer=renderer)
    imshow(v, title='v.T', renderer=renderer)

    # 1. We already have u, s, and v from torch.svd(res)
    u1 = u[:, 0]
    v1 = v[0, :]

    # 2. Fit linear models to u1 and v1
    # Fit for u's first column
    x_vals_u = np.arange(d_vocab)
    y_vals_u = u[:, 0].numpy()
    popt_u, _ = curve_fit(linear_func, x_vals_u, y_vals_u)

    # Fit for v's first row
    x_vals_v = np.arange(d_vocab)
    y_vals_v = v[:, 0].numpy()
    popt_v, _ = curve_fit(linear_func, x_vals_v, y_vals_v)

    # Plot u's column against its linear fit
    plt.figure()
    plt.scatter(x_vals_u, y_vals_u, alpha=0.5, label='Data')
    plt.plot(x_vals_u, linear_func(x_vals_u, *popt_u), 'r-', label=f'u: y = {popt_u[0]:.4f}x + {popt_u[1]:.4f}')
    plt.title("First Column of u vs Linear Fit")
    plt.legend()
    plt.show()

    # Plot residuals for u
    plt.figure()
    residuals_u = y_vals_u - linear_func(x_vals_u, *popt_u)
    plt.scatter(x_vals_u, residuals_u, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals of u's First Column Fit")
    plt.show()

    # Plot v's row against its linear fit
    plt.figure()
    plt.scatter(x_vals_v, y_vals_v, alpha=0.5, label='Data')
    plt.plot(x_vals_v, linear_func(x_vals_v, *popt_v), 'r-', label=f'v: y = {popt_v[0]:.4f}x + {popt_v[1]:.4f}')
    plt.title("First Row of v vs Linear Fit")
    plt.legend()
    plt.show()

    # Plot residuals for v
    plt.figure()
    residuals_v = y_vals_v - linear_func(x_vals_v, *popt_v)
    plt.scatter(x_vals_v, residuals_v, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals of v's First Row Fit")
    plt.show()

    # Subtract impact of lines
    u_prime = linear_func(x_vals_u, *popt_u)
    v_prime = linear_func(x_vals_v, *popt_v)
    impact = s[0] * u_prime[:, None] @ v_prime[None, :]
    adjusted_res = res - impact
    imshow(impact, title="adjustment", renderer=renderer)

    # adjusted_res = res - s[0] * (u[:, 0:1] @ v[:,0:1].T) * (popt_u[0] * x_vals_u[:, None] + popt_v[0] * x_vals_v[None, :])

    imshow(adjusted_res, title='Adjusted res', renderer=renderer)

    # SVD of adjusted_res
    u_adj, s_adj, v_adj = torch.svd(adjusted_res)
    line(s_adj, title='Singular Values of Adjusted res', renderer=renderer)
    imshow(u_adj, title='u of residuals', renderer=renderer)
    imshow(v_adj, title='v of residuals', renderer=renderer)

    # Extracting diagonal and off-diagonal entries
    diagonal_entries = torch.diag(adjusted_res)
    off_diagonal_entries = adjusted_res - torch.diag_embed(diagonal_entries)
    off_diagonal_entries = off_diagonal_entries[off_diagonal_entries != 0]

    # Finding the smallest diagonal entry and the largest off-diagonal entry
    min_diagonal_entry = diagonal_entries.min().item()
    max_off_diagonal_entry = off_diagonal_entries.max().item()

    # Printing the results
    print(f"Smallest diagonal entry: {min_diagonal_entry} ({pm_range(diagonal_entries)})")
    print(f"Largest off-diagonal entry: {max_off_diagonal_entry} ({pm_range(off_diagonal_entries)})")

    line(diagonal_entries, title='Diagonal Entries', renderer=renderer)

    off_diagonal_entries = off_diagonal_entries.flatten()
    # Histogram plot
    plt.hist(diagonal_entries.numpy(), bins=50, color='blue', alpha=0.7, label='Diagonal entries')
    plt.hist(off_diagonal_entries.numpy(), bins=50, color='red', alpha=0.5, label='Off-diagonal entries')
    plt.legend(loc='upper right')
    plt.title('Histogram of Diagonal and Off-diagonal Entries')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram plot
    plt.hist(diagonal_entries.numpy(), bins=50, color='blue', alpha=0.7, label='Diagonal entries', density=True)
    plt.hist(off_diagonal_entries.numpy(), bins=50, color='red', alpha=0.5, label='Off-diagonal entries', density=True)
    plt.legend(loc='upper right')
    plt.title('Density Histogram of Diagonal and Off-diagonal Entries')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()


    centered_adjusted_res = -adjusted_res + adjusted_res.diag()[:, None]
    nonzero_centered_adjusted_res = centered_adjusted_res[centered_adjusted_res != 0]

    imshow(centered_adjusted_res, title='adjusted copying.diag()[:,None] - adjusted copying', renderer=renderer)
    print(f"range on nonzero centered adjusted res: {pm_range(nonzero_centered_adjusted_res)}")

    statistics = [
        ('copying', res),
        ('diag', res_diag),
        ('off-diag', res_off_diagonal),
        ('centered', centered),
        ('nonzero centered', nonzero_centered),
    ]

    summaries = {name: summarize(value, name=name, renderer=renderer, histogram=False) for name, value in statistics}
    for k, v in summaries.items():
        print(k, v)
    return res

    # best_fit_u1 = linear_func(x_vals_u, *popt_u)

    # best_fit_v1 = linear_func(x_vals_v, *popt_v)

    # # 3. Calculate approximation of res
    # approximated_res = s[0] * torch.tensor(best_fit_u1).unsqueeze(1).mm(torch.tensor(best_fit_v1).unsqueeze(0))

    # # 4. Subtract approximation from original
    # residuals = res - approximated_res

    # # Display approximation and residuals
    # imshow(approximated_res, title="Approximated res", renderer=renderer)
    # imshow(residuals, title="Residuals after subtraction", renderer=renderer)

    # # Return svd of residuals
    # u_res, s_res, v_res = torch.svd(residuals)
    # line(s_res, title='singular values of residuals', renderer=renderer)
    # imshow(u_res, title='u of residuals', renderer=renderer)
    # imshow(v_res, title='v of residuals', renderer=renderer)

    # ... [rest of your function]



    # # 1. Project res onto the first singular vectors
    # first_component = s[0] * torch.outer(u[:, 0], v[:, 0])

    # # 2. Fit the projected data linearly
    # imshow(first_component, title='First Singular Vector', renderer=renderer)
    # # x_vals = torch.arange(first_component.numel())
    # # y_vals = first_component.view(-1).cpu().numpy()
    # # popt, _ = curve_fit(linear_func, x_vals, y_vals)
    # # best_fit = linear_func(x_vals, *popt)

    # # # Print the best-fit equation
    # # print(f"y = {popt[0]:.5f} * x + {popt[1]:.5f}")

    # # # 3. Display the best-fit line and residuals
    # # plt.figure()
    # # plt.scatter(x_vals, y_vals, label='Data', alpha=0.5)
    # # plt.plot(x_vals, best_fit, 'r-', label='Best-fit Line')
    # # plt.legend()
    # # plt.title("Impact of First Singular Value")

    # # residuals = y_vals - best_fit.numpy()
    # # plt.figure()
    # # plt.scatter(x_vals, residuals, alpha=0.5)
    # # plt.axhline(0, color='red', linestyle='--')
    # # plt.title("Residuals")

    # # # 4. Subtract the best-fit impact from res
    # # corrected_res = res - first_component.view_as(res)

    # # # Display corrected_res
    # # imshow(corrected_res, title='Corrected W_E @ W_V @ W_O @ W_U', renderer=renderer)

    # # # 5. Perform SVD on the corrected_res and display it again
    # # u_corr, s_corr, v_corr = torch.svd(corrected_res)
    # # line(s_corr, title='Singular values of corrected copying', renderer=renderer)

    # statistics = [
    #     ('copying', res),
    #     ('centered', centered),
    #     ('nonzero centered', nonzero_centered),
    # ]

    # # return {name: summarize(value, name=name, renderer=renderer, histogram=True) for name, value in statistics}




    # return summarize(res, name='W_pos @ W_V @ W_O @ W_U', renderer=renderer, linear_fit=True)
    # res = ((W_E + W_pos[-1,:]) @ W_U).detach()
    # self_overlap = res.diag()
    # centered = res - self_overlap
    # imshow(res, renderer=renderer)
    # line(self_overlap, renderer=renderer)
    # imshow(centered, renderer=renderer)
    # imshow(centered[:,1:], renderer=renderer)
    # statistics = [
    #     ('overlap (incl pos)', res),
    #     ('self-overlap (incl pos)', self_overlap),
    #     ('self-overlap after 0 (incl pos)', self_overlap[1:]),
    #     ('centered overlap (incl pos)', centered),
    #     ('centered overlap after 0 (incl pos)', centered[:,1:]),
    # ]
    # return {name: summarize(value, name=name, renderer=renderer) for name, value in statistics}


# In[ ]:


res = calculate_copying(simpler_model, renderer='png')
# %%

row_maxes = res.max(dim=1).values
row_mins = res.min(dim=1).values
largest_row_range = (row_maxes - row_mins).max().item()
print(largest_row_range)

# %%

# get diagonal above the main diagonal
first_diagonal = res.diag(diagonal=1)
res_without_last_row = res[:-1,:]
summarize(res_without_last_row - first_diagonal[:, None])


# In[ ]:


def calculate_copying_with_pos(model: HookedTransformer, renderer=None):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_E @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    res_pos = (W_pos @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    res_pos_min, res_pos_max = res_pos.min(dim=0).values, res_pos.max(dim=0).values
    res_diag = res.diag() + res_pos_min
    res_above_diag = -(res + res_pos_max[None,:]) + res_diag[:, None]
    imshow(res_above_diag, title='(W_E + worst(W_pos)) @ W_V @ W_O @ W_U', renderer=renderer,
              xaxis="logit affected", yaxis="input token")
    res_above_diag_off_diag = res_above_diag[torch.eye(d_vocab) == 0]
    first_diagonal = res.diag(diagonal=1) + res_pos_min[:-1]
    res_above_first_diagonal = -(res[:-1,:] + res_pos_max[None,:]) + first_diagonal[:, None]
    statistics = [
       ('res_above_diag_off_diag', res_above_diag_off_diag),
         ('res_above_first_diagonal', res_above_first_diagonal),
    ]
    for name, value in statistics:
        print(name, summarize(value, name=name, renderer=renderer, histogram=True))


calculate_copying_with_pos(simpler_model, renderer='png')

# In[ ]:


# res = calculate_copying(simpler_model, renderer='png')
# %%

row_maxes = res.max(dim=1).values
row_mins = res.min(dim=1).values
largest_row_range = (row_maxes - row_mins).max().item()
print(largest_row_range)

# %%

# get diagonal above the main diagonal
first_diagonal = res.diag(diagonal=1)
res_without_last_row = res[:-1,:]
summarize(res_without_last_row - first_diagonal[:, None])



# In[ ]:


orig_thresh = torch._tensor_str.PRINT_OPTS.threshold
torch.set_printoptions(threshold=1000000)
#print(res)
torch.set_printoptions(threshold=orig_thresh)


# ## Attention Scaling Factor

# In[ ]:


def calculate_attn(model: HookedTransformer, pos: Optional[int] = None, renderer=None):
    W_U, W_E, W_pos, W_Q, W_K = model.W_U, model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    if pos is None:
        return [calculate_attn(model, pos=i, renderer=renderer) for i in range(n_ctx)]
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_Q.shape == (1, 1, d_model, d_model)
    assert W_K.shape == (1, 1, d_model, d_model)
    residm1 = (W_E + W_pos[-1,:][None,:])
    resid = (W_E + W_pos[pos,:][None,:])
    q = (residm1 @ W_Q)[0,0,:,:]
    k = (resid @ W_K)[0,0,:,:]
    res = (q @ k.T).detach()
    # imshow(res, title=f'(W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos[{pos}]).T', renderer=renderer)
    centered = res - res.mean(dim=-1, keepdim=True)
    imshow(centered, title=f'centered (W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos[{pos}]).T', renderer=renderer,
           xaxis="Key token", yaxis="Query token")
    return centered
    # stats = [summarize(centered[i], name=f'pos {pos} row {i}', linear_fit=True, renderer=renderer) for i in range(centered.shape[0])]

# In[ ]:


calculate_attn(simpler_model, renderer='png')

# %%

# check for monotonicity violations
def check_monotonicity(model: HookedTransformer, renderer=None):
    count = 0
    centered_scores = calculate_attn(model, renderer=renderer)
    for pos, centered_score in enumerate(centered_scores):
        for row_n, row in enumerate(centered_score):
            for i in range(row.shape[0] - 1):
                for j in range(i + 1, row.shape[0]):
                    if row[i] > row[j]:
                        count += 1
                        print(f"{i, j} at row {row_n} pos {pos}, magnitude {row[i] - row[j]:.3f}")
    return count

monotonicity_violation_count = check_monotonicity(simpler_model, renderer='png')
print(f"Monotonicity violations: {monotonicity_violation_count}")

# %%

# Scatterplot of attention score differences when query token is one of the keys
points = []
centered_scores = calculate_attn(simpler_model, renderer='png')
for centered_score in centered_scores:
    for row_n, row in enumerate(centered_score):
        for i in range(row.shape[0]):
            points.append((i - row_n, row[i].item() - row[row_n].item()))
x, y = zip(*points)
# scatterplot
plt.scatter(x, y, alpha=0.5)
# set axis names
plt.xlabel('Distance from diagonal')
plt.ylabel('Attention score difference')

# %%

points = []
centered_scores = calculate_attn(simpler_model, renderer='png')
for centered_score in centered_scores:
    for row_n, row in enumerate(centered_score):
        for i in range(row.shape[0]):
            if i != row_n:
                points.append((row[i].item() - row[row_n].item()) / (i - row_n))
# histogram
plt.hist(points, bins=100, edgecolor='black')

# %%

def calculate_attn_by_pos(model: HookedTransformer, pos=False, renderer=None):
    W_U, W_E, W_pos, W_Q, W_K = model.W_U, model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_Q.shape == (1, 1, d_model, d_model)
    assert W_K.shape == (1, 1, d_model, d_model)
    residm1 = (W_E + W_pos[-1,:][None,:])
    
    resid = (W_E if not pos else W_pos[0,:][None,:] - W_pos[1, :][None,:])
    resid_name = 'W_E' if not pos else f'(W_pos[0] - W_pos[1])'
    q = (residm1 @ W_Q)[0,0,:,:]
    k = (resid @ W_K)[0,0,:,:]
    res = (q @ k.T).detach()
    # imshow(res, title=f'(W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos[{pos}]).T', renderer=renderer)
    centered = res - res.mean(dim=-1, keepdim=True) if not pos else res
    imshow(centered, title=f'centered (W_E + W_pos[-1]) @ W_Q @ W_K.T @ {resid_name}.T', renderer=renderer,
           xaxis="Key token", yaxis="Query token")
    print(centered.shape)
    return summarize(centered, name=f'centered (W_E + W_pos[-1]) @ W_Q @ W_K.T @ {resid_name}.T', 
                     renderer=renderer,
                     include_value=True)
    # stats = [summarize(centered[i], name=f'pos {pos} row {i}', linear_fit=True, renderer=renderer) for i in range(centered.shape[0])]

# %%

[calculate_attn_by_pos(simpler_model, p) for p in (True, False)]

# %%

points = []
centered_score = calculate_attn_by_pos(simpler_model, renderer='png', pos=False)['value']
for row_n, row in enumerate(centered_score):
    for i in range(row.shape[0]):
        if i != row_n:
            points.append((row[i].item() - row[row_n].item())  / (i - row_n))
# histogram
plt.hist(points, bins=100, edgecolor='black')
print(min(points))

# In[ ]:


simpler_model.W_Q.shape


# ## Attention Patterns

# In[ ]:


def calculate_qk_attn_heatmap(model, keypos=-1, querypos=-1, do_layernorm=True):
    attn = model.blocks[0].attn
    all_token_embeddings = model.embed(range(D_VOCAB))
    positional_embeddings = model.pos_embed(all_token_embeddings)

    token_embeddings_at_keypos = all_token_embeddings + positional_embeddings[:,keypos,:] if keypos > -1 else all_token_embeddings
    token_embeddings_at_querypos = all_token_embeddings + positional_embeddings[:,querypos,:] if querypos > -1 else all_token_embeddings

    # layernorm before attention
    if do_layernorm:
        token_embeddings_at_keypos = model.blocks[0].ln1(token_embeddings_at_keypos)
        token_embeddings_at_querypos = model.blocks[0].ln1(token_embeddings_at_querypos)

    embeddings_key = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                            token_embeddings_at_keypos, attn.W_K)
    embeddings_query = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                            token_embeddings_at_querypos, attn.W_Q)

    qk_circuit_attn_heatmap = einsum(
        "n_heads d_vocab_q d_head, n_heads d_vocab_k d_head -> ... d_vocab_q d_vocab_k",
        embeddings_query, embeddings_key
        ).detach().cpu().numpy()

    plt.rcParams['figure.figsize'] = [20, 10]
    return qk_circuit_attn_heatmap

def calculate_qk_attn_heatmap_normed(model, querypos=-1, do_layernorm=True, skip_var=True):
    all_token_embeddings = model.embed(range(D_VOCAB))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    all_heatmaps = torch.stack([torch.tensor(calculate_qk_attn_heatmap(model, cur_keypos, querypos, do_layernorm=do_layernorm)) for cur_keypos in range(positional_embeddings.shape[-2])])
    avg = einops.reduce(all_heatmaps, "keypos d_vocab_q d_vocab_k -> d_vocab_q ()", 'mean')
    var = einops.reduce(all_heatmaps, "keypos d_vocab_q d_vocab_k -> d_vocab_q ()", torch.var)
    #print(all_heatmaps.shape, avg.shape)
    #print(avg)
    res = (all_heatmaps - avg)
    if not skip_var: res = res * (var ** -0.5)
    return res

def plot_qk_heatmap(model, keypos=-1, querypos=-1, do_layernorm=True):
  qk_attn_heatmap = calculate_qk_attn_heatmap(model, keypos=keypos, querypos=querypos, do_layernorm=do_layernorm)

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(qk_attn_heatmap, cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()

def plot_qk_heatmaps_normed(model, keypositions=None, querypos=-1, do_layernorm=True, skip_var=True):
    if keypositions is None:
        all_token_embeddings = model.embed(range(D_VOCAB))
        positional_embeddings = model.pos_embed(all_token_embeddings)
        keypositions = range(positional_embeddings.shape[-2])

    heatmaps = calculate_qk_attn_heatmap_normed(model, querypos=querypos, do_layernorm=do_layernorm, skip_var=skip_var)
    for keypos in keypositions:
        fig, ax = plt.subplots(figsize=(8, 8))
        qk_attn_heatmap = heatmaps[keypos]
        graph = ax.imshow(qk_attn_heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar(graph)
        plt.tight_layout()
        plt.show()
    print(heatmaps.shape) # torch.Size([2, 64, 64]), keypos d_vocab_q d_vocab_k
    # do linear regression on the heatmaps, with

def plot_avg_qk_heatmap(model, keypositions, querypos=-1, do_layernorm=True):
  heatmaps = []

  for keypos in keypositions:
    heatmaps.append(calculate_qk_attn_heatmap(model, keypos=keypos, querypos=querypos, do_layernorm=do_layernorm))

  qk_circuit_attn_heatmap = np.mean(heatmaps, axis=0)

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(qk_circuit_attn_heatmap, cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()

plot_qk_heatmaps_normed(simpler_model, querypos=1, skip_var=True)
#for keypos in (0, 1): plot_qk_heatmap(model, keypos=keypos, querypos=1, do_layernorm=True)


# # Model Setup

# A simple one-layer attention only model with a context length of 2.

# In[ ]:


cfg = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=2,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=SEED,
    device=DEVICE,
    attn_only=True
)
model = HookedTransformer(cfg, move_to_device=True)


# In[ ]:





# # Training

# ### Training Loop

# We train for 500 epochs with 10 batches of 128 tokens per epoch. (This is somewhat arbitrary as the intention was mostly to get a good model quickly.)


# In[ ]:


TRAIN_MODEL = ALWAYS_TRAIN_MODEL
if not ALWAYS_TRAIN_MODEL:
    try:
        cached_data = torch.load(MODEL_PTH_PATH)
        model.load_state_dict(cached_data['model'])
        #model_checkpoints = cached_data["checkpoints"]
        #checkpoint_epochs = cached_data["checkpoint_epochs"]
        #test_losses = cached_data['test_losses']
        train_losses = cached_data['train_losses']
        #train_indices = cached_data["train_indices"]
        #test_indices = cached_data["test_indices"]
    except Exception as e:
        print(e)
        TRAIN_MODEL = TRAIN_MODEL_IF_CANT_LOAD


# In[ ]:


if TRAIN_MODEL:
    train_losses = train_model(model, n_epochs=500, batch_size=128, sequence_length=2)


# In[ ]:


if TRAIN_MODEL:
    data = {
                "model":model.state_dict(),
                "config": model.cfg,
                "train_losses": train_losses,
            }
    if OVERWRITE_DATA or not os.path.exists(MODEL_PTH_PATH):
        torch.save(
            data,
            MODEL_PTH_PATH)
    else:
        print(f'WARNING: Not overwriting {MODEL_PTH_PATH} because it already exists.')
        ext = 0
        while os.path.exists(f"{MODEL_PTH_PATH}.{ext}"):
            ext += 1
        torch.save(
            data,
            f"{MODEL_PTH_PATH}.{ext}")
        print(f'WARNING: Wrote to {MODEL_PTH_PATH}.{ext} instead.')


# As we can see accuracy is high and loss is low.

# In[ ]:


line(train_losses, xaxis="Epoch", yaxis="Loss")


# # Exporting the Model to Coq

# In[ ]:


# dir(model)


# In[ ]:


# print(model)


# In[ ]:


#print(cfg.__dict__)


# In[ ]:


#{f.name:(f.type, f) for f in dataclasses.fields(cfg)}


# In[ ]:


# set(type(v) for k, v in cfg.__dict__.items())


# In[ ]:


#dir(model.blocks[0].ln1)


# In[ ]:


# model.blocks[0].attn.mask


# In[ ]:





# In[ ]:


EXPORT_TO_COQ = True
if EXPORT_TO_COQ:
    print('Module cfg.')
    for f in dataclasses.fields(cfg):
        val = dataclasses.asdict(cfg)[f.name]
        ty = f.type
        if f.name == 'attn_types' and ty == 'Optional[List]': ty = 'Optional[List[str]]'
        print(f'  Definition {f.name} := {strify(val, ty=ty)}.')
    print('End cfg.')

    for name in (#'OV',
    #'QK',
    #'T_destination',
    'W_E',
    #'W_E_pos',
    'W_K',
    'W_O',
    'W_Q',
    'W_U',
    'W_V',
    #'W_in',
    #'W_out',
    'W_pos', 'b_K',
    'b_O',
    'b_Q',
    'b_U',
    'b_V',):
    #'b_in',
    #'b_out'):
        print(f'Definition {name} :=')
        print(strify(getattr(model, name)))
        print('.')

    for layer, block in enumerate(model.blocks):
        for mod, names in (('ln1', ('b', 'w')), ('attn', ('W_Q', 'W_K', 'W_O', 'W_V', 'b_Q', 'b_K', 'b_O', 'b_V'))):
            for name in names:
                print(f'Definition L{layer}_{mod}_{name} :=')
                print(strify(getattr(getattr(block, mod), name)))
                print('.')

    for mod, names in (('ln_final', ('b', 'w')), ):
        for name in names:
            print(f'Definition {mod}_{name} :=')
            print(strify(getattr(getattr(model, mod), name)))
            print('.')


# In[ ]:


# Test
# model.embed(torch.tensor([0, 1]))


# In[ ]:


# cosine dissimilarity of embeddings
#all_token_embeddings = model.embed(range(D_VOCAB))


# In[ ]:


# model.blocks[0].ln1


# In[ ]:


# norms = all_token_embeddings.norm(dim=-1)
# print(norms.mean(), norms.min(), norms.max(), norms.var())
# norms = torch.stack([model.blocks[0].ln1(i) for i in all_token_embeddings]).norm(dim=-1)
# print(norms.mean(), norms.min(), norms.max(), norms.var())


# In[ ]:


# positional_embeddings = model.pos_embed(all_token_embeddings)


# In[ ]:


# all_token_pos_embed = all_token_embeddings[:,None,:] + positional_embeddings


# In[ ]:


# all_token_pos_embed.shape


# In[ ]:


# pos_embed_vec = positional_embeddings[0]


# In[ ]:


# centered_all_token_pos_embed = all_token_pos_embed - all_token_pos_embed.mean(dim=-1, keepdim=True)
# centered_vec = centered_all_token_pos_embed.mean(dim=1).mean(dim=0)
# print(centered_vec.var(dim=-1).sqrt())
# all_stddevs = (centered_all_token_pos_embed - centered_vec).var(dim=-1).sqrt()
# all_scale = 1/all_stddevs
# print(all_stddevs.mean(), all_stddevs.var(), all_scale.mean(), all_scale.var())
# print(all_stddevs.max() / all_stddevs.mean(), all_stddevs.mean() / all_stddevs.min())
# print(all_scale.max() / all_scale.mean(), all_scale.mean() / all_scale.min())


# In[ ]:


# print(pos_embed_vec[0].T @ pos_embed_vec[1] / pos_embed_vec[0].norm() / pos_embed_vec[1].norm())
# print(pos_embed_vec.var(dim=-1))
# centered_all_token_embeddings = all_token_embeddings - all_token_embeddings.mean(axis=-1, keepdim=True)
# centered_pos_embed_vec = pos_embed_vec - pos_embed_vec.mean(axis=-1, keepdim=True)
# from fancy_einsum import einsum
# cos_sim_between_tokens_and_positions = einsum("tok_emb d_model, pos_emb d_model -> tok_emb pos_emb", centered_all_token_embeddings / centered_all_token_embeddings.norm(dim=-1, keepdim=True), centered_pos_embed_vec / centered_pos_embed_vec.norm(dim=-1, keepdim=True))


# In[ ]:


# Create two tensors: one for the x-coordinates and one for the y-coordinates
x = torch.arange(0, 64)
y = torch.arange(0, 64)

# Use the cartesian_prod function to get all combinations of x and y
all_integers = torch.cartesian_prod(x, y)

# Reshape the tensor to the required shape (64^2, 2)
all_integers = all_integers.reshape(64**2, 2)
all_integers.shape
#model(torch.tensor([0, 1]))


# In[ ]:


all_integers_result = model(all_integers)
print(f"loss: {loss_fn(all_integers_result, all_integers)}")
print(f"acc: {acc_fn(all_integers_result, all_integers)}")


# In[ ]:


all_integers_ans = all_integers_result[:,-1]
ans = all_integers_ans.argmax(dim=-1)
expected = all_integers.max(dim=-1).values
alt_expected = all_integers.min(dim=-1).values
correct_idxs = (ans == expected)
very_wrong_idxs = ~((ans == expected) | (ans == alt_expected))
print(all_integers[~correct_idxs], very_wrong_idxs.sum())
#list(zip(all_integers[~correct_idxs], all_integers_ans[~correct_idxs]))


# # Interpretability

# ## Unembed

# In[ ]:


def plot_unembed_cosine_similarity(model):
    all_token_embeddings = model.embed(range(D_VOCAB))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    all_token_pos_embed = all_token_embeddings[:,None,:] + positional_embeddings
    #print(model.W_U.shape, all_token_embeddings.shape, positional_embeddings.shape)
    # torch.Size([32, 64]) torch.Size([64, 32]) torch.Size([64, 2, 32])
    avg = F.normalize(all_token_embeddings.sum(dim=0), dim=-1)
    # overlap between model.W_U and token embedings
    input_overlap = all_token_pos_embed @ model.W_U
    print(f"Definition max_input_output_overlap := {input_overlap.abs().max()}.")
    line(F.cosine_similarity(avg[None,:], all_token_embeddings, dim=-1))



plot_unembed_cosine_similarity(model)


# In[ ]:


line(model.b_U)
line(model.blocks[0].ln1.b @ model.W_U + model.b_U)


# In[ ]:


print(model.W_U.shape)
print(model.blocks[0].ln1.b)
print(model.blocks[0].ln1.w)
print(model.blocks[0].ln1.b.shape)
print(model.blocks[0].ln1.w.shape)


# In[ ]:


def analyze_svd(M, descr=''):
    U, S, Vh = torch.svd(M)
    if descr: descr = f' for {descr}'
    line(S, title=f"Singular Values{descr}")
    imshow(U, title=f"Principal Components on U{descr}")
    imshow(Vh, title=f"Principal Components on Vh{descr}")


# In[ ]:


analyze_svd(model.W_U)


# In[ ]:


def analyze_svd(W):
    U, S, Vh = torch.svd(W)
    line(S, title="Singular Values")
    imshow(U, title="Principal Components on the Input")
    imshow(Vh)
print(model.W_U.shape)
print((model.blocks[0].ln1.w[:,None] * model.W_U).shape)
analyze_svd(model.blocks[0].ln1.w[:,None] * model.W_U)


# ## Attention Patterns
# 
# First, we visualize the attention patterns for a few inputs to see if this will give us an idea of what the model is doing.

# We begin by getting a batch of data and running a feedforward pass through the model, storing the resulting logits as well as the activations (in cache).

# In[ ]:


data_train, data_test = get_data(sequence_length=2)
train_data_gen = make_generator_from_data(data_train, batch_size=128)
tokens = next(train_data_gen)
logits, cache = model.run_with_cache(tokens)


# We get the attention pattern from the cache:

# In[ ]:


attention_pattern = cache["pattern", 0, "attn"]


# Let us now visualize the attention patterns for the first few datapoints:
# 
# We see that for sequences (42,22),(20,17), and (33, 21) most of the second tokens attention is on the first token (which has a larger value). On the other hand, for (52, 59) and (1, 13) the second token pays the most attention to itself. This suggests that the head learns to pay more attention to larger tokens for the second position.

# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[0])), attention=attention_pattern[0])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[1])), attention=attention_pattern[1])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[2])), attention=attention_pattern[2])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[4])), attention=attention_pattern[4])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[5])), attention=attention_pattern[5])


# ## Visualizing the QK-Circuit
# 
# The QK-circuit determines which tokens are attended to by the OV-circuit. By multiplying the embedding of every token with the full QK-Circuit of a particular head, we get a heatmap that shows us the attention every token is paying to every other token for that head. Based on what we have seen we looking at attention patterns we would expect that the QK-circuit of our model's singular head will always be more attentive to tokens with larger values.

# Here is are helper functions for visualizing qk-circuits. The function *calculate_qk_attn_heatmap* is mostly based on this [function](https://github.com/MatthewBaggins/one-attention-head-is-all-you-need/blob/main/notebooks/1_Sorting_Fixed_Length_Lists_with_One_Head.ipynb) used by Bagiński and Kolly (B & K). However, I found it necessary to add the positional embedding to get a useful visualization.

# In[ ]:


all_token_embeddings = model.embed(range(D_VOCAB))
positional_embeddings = model.pos_embed(all_token_embeddings)
print(all_token_embeddings.shape, positional_embeddings.shape)

# Assuming you have a tensor called 'tensor' with shape torch.Size([64, 32])
# Normalize the tensor along its last dimension
tensor_norm = torch.nn.functional.normalize(all_token_embeddings, dim=1)

# Compute the cosine similarity
cos_sim = torch.mm(tensor_norm, tensor_norm.t())

if False:
    # Sort the columns by the first row
    indices = cos_sim[0].argsort(descending=True)
    sorted_cos_sim = cos_sim
    sorted_cos_sim = sorted_cos_sim[:, indices]
    sorted_cos_sim = sorted_cos_sim[indices, :]

    # Convert the tensor to numpy for visualization
    sorted_cos_sim_np = sorted_cos_sim.detach().numpy()

    # Show heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap = ax.imshow(sorted_cos_sim_np, cmap="hot", interpolation="nearest")
    plt.colorbar(heatmap)
    plt.tight_layout()
    plt.show()


from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Convert the cosine similarity tensor to a numpy array
cos_sim_np = cos_sim.detach().numpy()

# Show heatmap
fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.imshow(cos_sim_np, cmap="hot", interpolation="nearest")
plt.colorbar(heatmap)
plt.tight_layout()
plt.show()



# Compute the distance matrix
dist_mat = 1 - cos_sim_np

# Find the maximal diagonal element
max_diag_element = np.max(np.diag(dist_mat))
print(f'Maximal diagonal element before setting to zero: {max_diag_element}')

# Explicitly set the diagonal elements to zero
np.fill_diagonal(dist_mat, 0)

# Perform hierarchical clustering
links = linkage(squareform(dist_mat), method='average')

# Get the order of rows according to the hierarchical clustering
order = leaves_list(links)

# Rearrange the rows and columns of the cosine similarity matrix
reordered_cos_sim = cos_sim_np[order, :][:, order]


def avg_cos_sim(order, cos_sim_np):
    """Compute average cosine similarity between adjacent elements"""
    return np.mean(np.abs(np.diff(cos_sim_np[np.ix_(order, order)], axis=0)))

for _ in range(len(cos_sim_np) ** 2):
    # Initial order of indices
    order = np.arange(len(cos_sim_np))

    current_avg_cos_sim = avg_cos_sim(order, reordered_cos_sim)

    # Flag to keep track of whether any pair was swapped
    any_swapped = False

    # Iteratively select a pair of indices to swap if it decreases the average cosine similarity
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            # Swap a pair of indices
            new_order = order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]

            # Compute the average cosine similarity with the new order
            new_avg_cos_sim = avg_cos_sim(new_order, reordered_cos_sim)

            # If the new average cosine similarity is lower, update the order and the current average cosine similarity
            if new_avg_cos_sim < current_avg_cos_sim:
                order = new_order
                current_avg_cos_sim = new_avg_cos_sim
                any_swapped = (i, j)
                break

        # If a pair was swapped, break the outer loop as well
        if any_swapped:
            break
    print(f"A pair of indices was swapped: {any_swapped}" if any_swapped else "No pair of indices was swapped")
    if any_swapped:
        reordered_cos_sim = reordered_cos_sim[new_order, :][:, new_order]
    else:
        break

# Show heatmap
fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.imshow(reordered_cos_sim, cmap="hot", interpolation="nearest")
plt.colorbar(heatmap)
plt.tight_layout()
plt.show()


# In[ ]:


model.blocks[0].attn.W_K.shape


# In[ ]:


W_QK = einsum("n_heads d_model_query d_head, n_heads d_model_key d_head  -> n_heads d_model_query d_model_key",
            model.blocks[0].attn.W_Q, model.blocks[0].attn.W_K)
all_token_embeddings = model.embed(range(D_VOCAB))
positional_embeddings = model.pos_embed(all_token_embeddings)

token_embeddings_at_querypos = all_token_embeddings + positional_embeddings[:,-1,:]
token_embeddings_at_querypos = model.blocks[0].ln1(token_embeddings_at_querypos)
W_QK_query = einsum("n_heads d_model_query d_head, n_heads d_model_key d_head, d_vocab_query d_model_query  -> n_heads d_vocab_query d_model_key",
            model.blocks[0].attn.W_Q, model.blocks[0].attn.W_K, token_embeddings_at_querypos)
W_QK_query.shape
cos_sims = torch.tensor([[torch.nn.functional.cosine_similarity(W_QK_query[0,i], W_QK_query[0,j], dim=0) for i in range(W_QK_query.shape[1])] for j in range(W_QK_query.shape[1])])
print(cos_sims.min(), cos_sims.max())
px.imshow(cos_sims)


# In[ ]:


W_QK_dir = (W_QK_query / W_QK_query.norm(dim=1, keepdim=True)).sum(dim=1)
W_QK_dir = W_QK_dir / W_QK_dir.norm(dim=-1, keepdim=True)
W_QK_dir


# In[ ]:


def calculate_qk_attn_heatmap(model, keypos=-1, querypos=-1, do_layernorm=True):
    attn = model.blocks[0].attn
    all_token_embeddings = model.embed(range(D_VOCAB))
    positional_embeddings = model.pos_embed(all_token_embeddings)

    token_embeddings_at_keypos = all_token_embeddings + positional_embeddings[:,keypos,:] if keypos > -1 else all_token_embeddings
    token_embeddings_at_querypos = all_token_embeddings + positional_embeddings[:,querypos,:] if querypos > -1 else all_token_embeddings

    # layernorm before attention
    if do_layernorm:
        token_embeddings_at_keypos = model.blocks[0].ln1(token_embeddings_at_keypos)
        token_embeddings_at_querypos = model.blocks[0].ln1(token_embeddings_at_querypos)

    embeddings_key = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                            token_embeddings_at_keypos, attn.W_K)
    embeddings_query = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                            token_embeddings_at_querypos, attn.W_Q)

    qk_circuit_attn_heatmap = einsum(
        "n_heads d_vocab_q d_head, n_heads d_vocab_k d_head -> ... d_vocab_q d_vocab_k",
        embeddings_query, embeddings_key
        ).detach().cpu().numpy()

    plt.rcParams['figure.figsize'] = [20, 10]
    return qk_circuit_attn_heatmap

def calculate_qk_attn_heatmap_normed(model, querypos=-1, do_layernorm=True, skip_var=True):
    all_token_embeddings = model.embed(range(D_VOCAB))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    all_heatmaps = torch.stack([torch.tensor(calculate_qk_attn_heatmap(model, cur_keypos, querypos, do_layernorm=do_layernorm)) for cur_keypos in range(positional_embeddings.shape[-2])])
    avg = einops.reduce(all_heatmaps, "keypos d_vocab_q d_vocab_k -> d_vocab_q ()", 'mean')
    var = einops.reduce(all_heatmaps, "keypos d_vocab_q d_vocab_k -> d_vocab_q ()", torch.var)
    #print(all_heatmaps.shape, avg.shape)
    #print(avg)
    res = (all_heatmaps - avg)
    if not skip_var: res = res * (var ** -0.5)
    return res

def plot_qk_heatmap(model, keypos=-1, querypos=-1, do_layernorm=True):
  qk_attn_heatmap = calculate_qk_attn_heatmap(model, keypos=keypos, querypos=querypos, do_layernorm=do_layernorm)

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(qk_attn_heatmap, cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()

def plot_qk_heatmaps_normed(model, keypositions=None, querypos=-1, do_layernorm=True, skip_var=True):
    if keypositions is None:
        all_token_embeddings = model.embed(range(D_VOCAB))
        positional_embeddings = model.pos_embed(all_token_embeddings)
        keypositions = range(positional_embeddings.shape[-2])

    heatmaps = calculate_qk_attn_heatmap_normed(model, querypos=querypos, do_layernorm=do_layernorm, skip_var=skip_var)
    for keypos in keypositions:
        fig, ax = plt.subplots(figsize=(8, 8))
        qk_attn_heatmap = heatmaps[keypos]
        graph = ax.imshow(qk_attn_heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar(graph)
        plt.tight_layout()
        plt.show()
    print(heatmaps.shape) # torch.Size([2, 64, 64]), keypos d_vocab_q d_vocab_k
    # do linear regression on the heatmaps, with

def plot_avg_qk_heatmap(model, keypositions, querypos=-1, do_layernorm=True):
  heatmaps = []

  for keypos in keypositions:
    heatmaps.append(calculate_qk_attn_heatmap(model, keypos=keypos, querypos=querypos, do_layernorm=do_layernorm))

  qk_circuit_attn_heatmap = np.mean(heatmaps, axis=0)

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(qk_circuit_attn_heatmap, cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()

plot_qk_heatmaps_normed(model, querypos=1, skip_var=True)
#for keypos in (0, 1): plot_qk_heatmap(model, keypos=keypos, querypos=1, do_layernorm=True)


# In[ ]:





# In[ ]:


# cosine similarity


# In[ ]:


def constant_function(x, a):
    return a

def linear_function(x, a, b):
    return a * x + b

def compute_best_fit_and_error(direction_dot_embed):
    n_head, d_vocab = direction_dot_embed.shape

    coefficients = torch.empty((n_head, 2))  # To store the coefficients a, b for each row
    max_abs_errors = torch.empty(n_head)  # To store the max abs error for each row
    errors = torch.empty((n_head, d_vocab))
    predicted = torch.empty((n_head, d_vocab))
    negative_pairs = []
    diff_values = []

    x_values = np.arange(d_vocab)

    # Create a meshgrid of indices
    idxi, idxj = np.meshgrid(x_values, x_values)
    # Exclude the diagonal (i == j)
    mask = idxi != idxj
    pairs = list(zip(idxi[mask], idxj[mask]))  # create a list of pairs (i, j)

    for i in range(n_head):
        row = direction_dot_embed[i].detach().numpy()

        # Use curve_fit to find a, b that best fit the data in this row
        coeff, _ = curve_fit(linear_function, x_values, row)
        coefficients[i] = torch.from_numpy(coeff)

        # Compute the predicted y values using these coefficients
        y_pred = coeff[0] * x_values + coeff[1]

        # Compute the absolute error for each value, and take the maximum
        cur_errors = row - y_pred
        max_abs_errors[i] = np.abs(cur_errors).max()
        errors[i] = torch.from_numpy(cur_errors)
        predicted[i] = torch.from_numpy(y_pred)

        # Compute (pos[i] - pos[j]) / (i - j) for all pairs (i, j)
        values = (row[idxi] - row[idxj]) / (idxi - idxj)

        # Select only the values where i != j
        values = values[mask]
        negative_pairs.append([pair for pair, value in zip(pairs, values) if value < 0])

        diff_values.append(values)

    return coefficients, max_abs_errors, errors, predicted, diff_values, negative_pairs


# In order to find the ordering of `direction_dot_embed_error` that maximizes the number of negative values in the difference `not_a_line[1:] - not_a_line[:-1]`, one approach is to use a brute-force method and iterate over all possible permutations of `direction_dot_embed_error`.
# 
# However, this method becomes computationally expensive and unfeasible as the size of `direction_dot_embed_error` increases, since the number of permutations grows factorially with the size of the array.
# 
# A more efficient approach is to observe that the difference `not_a_line[1:] - not_a_line[:-1]` is equivalent to `direction_dot_embed_error[i+1] * (i+1) - direction_dot_embed_error[i] * i` for `i` from `0` to `len(direction_dot_embed_error) - 2`.
# 
# This implies that we want to maximize the number of times `direction_dot_embed_error[i+1] / (i+1) < direction_dot_embed_error[i] / i` for `i` from `0` to `len(direction_dot_embed_error) - 2`. This is achieved when `direction_dot_embed_error` is sorted in ascending order of its elements divided by their indices. Therefore, we can sort `direction_dot_embed_error` in this order, and this will give us the desired ordering.
# 
# Here is the corresponding code:
# 
# ```python
# import torch
# import numpy as np
# 
# # Assuming 'direction_dot_embed_error' is your tensor
# direction_dot_embed_error = torch.tensor([...])  # fill in with your values
# 
# # Get the indices that would sort the array in the desired order
# indices = np.argsort(direction_dot_embed_error.numpy() / np.arange(1, len(direction_dot_embed_error) + 1))
# 
# # Use these indices to sort 'direction_dot_embed_error'
# sorted_tensor = direction_dot_embed_error[indices]
# 
# print(sorted_tensor)
# ```
# 
# In this code, `np.argsort` is used to get the indices that would sort `direction_dot_embed_error` in ascending order of its elements divided by their indices. These indices are then used to sort `direction_dot_embed_error`.
# 
# Note: Replace `direction_dot_embed_error = torch.tensor([...])` with your actual tensor. This code also assumes that `direction_dot_embed_error` is a 1D tensor. If it's not, you may need to modify the code accordingly.

# In[ ]:


def count_monotonicity_violations_line(result_tensor, m):
    # Count the number of pairs of indices (i, j), i != j, for which
    # (result_tensor[i] + m*i - result_tensor[j] + m*j) / (i - j) is negative
    count = 0
    for i in range(len(result_tensor)):
        for j in range(i + 1, len(result_tensor)):
            if ((result_tensor[i] + m*i - result_tensor[j] + m*j) / (i - j)) < 0:
                count += 1
    return count


# In[ ]:


def reorder_tensor_greedy(tensor, m):
    # Convert to numpy for easier handling
    tensor_np = tensor.detach().clone().numpy()

    # Initialize the result with the maximum positive value
    result = [np.max(tensor_np)]
    tensor_np = np.delete(tensor_np, np.argmax(tensor_np))

    while len(tensor_np) > 0:
        # Find values that maintain the condition
        candidates = tensor_np[tensor_np - result[-1] < -m]

        if len(candidates) > 0:
            # If such values exist, select the maximum
            next_value = np.max(candidates)
        else:
            # Otherwise, select the maximum of the remaining values
            next_value = np.max(tensor_np)

        # Add the selected value to the result
        result.append(next_value)

        # Remove the selected value from the list of remaining values
        tensor_np = np.delete(tensor_np, np.where(tensor_np == next_value)[0][0])

    # Convert the result back to a tensor
    result_tensor = torch.tensor(result)

    # Count the number of indices for which the difference between
    # successive elements in the result is less than -m
    # diff = result_tensor[1:] - result_tensor[:-1]
    # count = torch.sum(diff < -m).item()

    count = count_monotonicity_violations_line(result_tensor, m)

    return result_tensor, count



# In[ ]:


def plot_QK_cosine_similarity(model, keypos=-1, querypos=-1, do_layernorm=True):
    attn = model.blocks[0].attn
    all_token_embeddings = model.embed(range(D_VOCAB))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    normed_all_token_embeddings = F.normalize(all_token_embeddings, dim=-1)

    token_embeddings_at_keypos = all_token_embeddings + positional_embeddings[:,keypos,:] if keypos > -1 else all_token_embeddings
    token_embeddings_at_querypos = all_token_embeddings + positional_embeddings[:,querypos,:] if querypos > -1 else all_token_embeddings

    # layernorm before attention
    if do_layernorm:
        token_embeddings_at_keypos = model.blocks[0].ln1(token_embeddings_at_keypos)
        token_embeddings_at_querypos = model.blocks[0].ln1(token_embeddings_at_querypos)

    #embeddings_key = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
    #                        token_embeddings_at_keypos, attn.W_K)
    #embeddings_query = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
    #                        token_embeddings_at_querypos, attn.W_Q)
    embeddings_query_waiting_for_key = einsum("d_vocab_query d_model_query, n_heads d_model_query d_head, n_heads d_model_key d_head -> n_heads d_vocab_query d_model_key",
                            token_embeddings_at_querypos, attn.W_Q, attn.W_K)

    QK = einsum("n_heads d_model_query d_head, n_heads d_model_key d_head -> n_heads d_model_query d_model_key",
                            attn.W_Q, attn.W_K)

    analyze_svd(embeddings_query_waiting_for_key[0], descr="embeddings_query_waiting_for_key")
    analyze_svd(QK[0], descr="QK")
    U, S, Vh = torch.svd(embeddings_query_waiting_for_key[0])
    print(Vh[0])
    print(Vh.T[0])
    print((U @ torch.diag(S) @ Vh.T)[0])
    print((U @ torch.diag(S) @ Vh.T).T[0])
    imshow(U @ torch.diag(S) @ Vh.T, title="tmp")
    #qk_circuit_attn_heatmap = einsum(
    #    "n_heads d_vocab_q d_head, n_heads d_vocab_k d_head -> ... d_vocab_q d_vocab_k",
    #    embeddings_query, embeddings_key
    #    ).detach().cpu().numpy()

    imshow(embeddings_query_waiting_for_key[0])


    direction = embeddings_query_waiting_for_key
    #direction = direction / direction.norm(dim=-1, keepdim=True)
    direction = direction.sum(dim=1) / direction.shape[1]
    print(f"Definition size_direction := {direction}.")
    direction = direction / direction.norm(dim=-1)
    print(f"Definition normed_size_direction := {direction}.")
    print(all_token_embeddings.shape, direction.shape)
    proj_direction_scale = einsum("n_head d_model_key, n_head d_vocab_query d_model_key -> n_head d_vocab_query",
                                  direction,
                                  embeddings_query_waiting_for_key)[:,:,None]
    print(proj_direction_scale.shape)
    proj_direction = proj_direction_scale * einops.rearrange(direction, "n_head d_model -> n_head () d_model")
    print(proj_direction.shape)
    remaining_directions = embeddings_query_waiting_for_key - proj_direction
    print(remaining_directions.shape)
    remaining_directions = remaining_directions.norm(dim=-1)
    print(remaining_directions.shape)
    direction_key_overlap = einsum("n_head d_model_key, n_head d_vocab_query d_model_key -> d_vocab_query n_head",
                direction,
                embeddings_query_waiting_for_key)
    print(direction_key_overlap.shape)
    print(f"Definition min_attention_query_size_direction_overlap := {direction_key_overlap.min()}.")
    direction_dot_embed = einsum("n_head d_model, d_vocab d_model -> n_head d_vocab", direction, normed_all_token_embeddings)
    direction_dot_pos_embed = einsum("n_head d_model, pos d_model -> n_head pos", direction, positional_embeddings[0])
    print(f"Definition max_direction_dot_pos_embed := {direction_dot_pos_embed.abs().max()}.")
    # linear fit of direction_dot_embed
    direction_dot_embed_coefficients, direction_dot_embed_max_abs_errors, direction_dot_embed_error, direction_dot_embed_predicted, direction_dot_embed_diff_values, direction_dot_embed_neg_values = \
          compute_best_fit_and_error(direction_dot_embed)

    direction_dot_embed_diffs = direction_dot_embed[...,1:] - direction_dot_embed[...,:-1]
    #direction_dot_embed_coef = direction_dot_embed_diffs.mean(dim=-1, keepdim=True)
    #direction_dot_embed_offset = direction_dot_embed.mean(dim=-1, keepdim=True)
    #direction_dot_embed_diff_error = direction_dot_embed_diffs - torch.arange(direction_dot_embed_diffs.shape[-1]) * direction_dot_embed_coef + direction_dot_embed_offset)
    print(direction_dot_embed_diffs)
    print(direction_dot_embed_diffs.abs())
    line(direction_dot_embed_diffs.T, title="direction_dot_embed_diffs")
    line(direction_dot_embed_diffs.T.abs(), title="direction_dot_embed_diffs abs")
    #line(direction_dot_embed_diff_error.T, title="direction_dot_embed_diff_error")
    #print(direction_dot_embed_coef, direction_dot_embed_offset)


    #direction_dot_embed_coef_better, _ = curve_fit(constant_function, np.arange(direction_dot_embed_diffs.shape[-1]), direction_dot_embed_diffs[0].detach().numpy())
    #direction_dot_embed_diff_error_better = direction_dot_embed_diffs - (torch.arange(direction_dot_embed_diffs.shape[-1]) * direction_dot_embed_coef + direction_dot_embed_offset)
    #line(direction_dot_embed_diffs.T, title="direction_dot_embed_diffs")
    #line(direction_dot_embed_diff_error.T, title="direction_dot_embed_diff_error")
    #print(direction_dot_embed_coef, direction_dot_embed_offset)


    # indices = np.argsort(direction_dot_embed_error[0].numpy() / np.arange(1, len(direction_dot_embed_error[0]) + 1))

    # Use these indices to sort 'direction_dot_embed_error'
    # sorted_direction_dot_embed_error = direction_dot_embed_error[:,indices]
    print(direction_dot_embed_error.mean(), direction_dot_embed_error.var())
    # randomly reorder direction_dot_embed_error, put in tmp
    tmp = direction_dot_embed_error[0].detach().clone().numpy()
    np.random.shuffle(tmp)
    print(tmp)
    line(tmp)
    print(count_monotonicity_violations_line(torch.tensor(tmp), direction_dot_embed_coefficients[0, 0].item()))
    print(f"Definition ")
    sorted_direction_dot_embed_error, bad_count = reorder_tensor_greedy(direction_dot_embed_error[0], direction_dot_embed_coefficients[0, 0].item())
    sorted_direction_dot_embed_error = sorted_direction_dot_embed_error[None,:]

    # sorted_direction_dot_embed_error, _ = direction_dot_embed_error.sort(dim=-1, descending=True)
    print(direction_dot_embed_coefficients, direction_dot_embed_max_abs_errors, direction_dot_embed)
    line(direction_key_overlap, title="direction @ query_waiting_for_key")
    line(remaining_directions.T, title="norm of remaining direction")
    line(F.cosine_similarity(direction, all_token_embeddings, dim=-1), title="cos_sim(direction, embed)")
    print(positional_embeddings.shape)
    line(direction_dot_embed.T, title="direction @ normed embed")
    line(torch.cat([direction_dot_embed, direction_dot_embed_predicted], dim=0).T, title="direction @ normed embed, + fit")
    print(bad_count)
    line(torch.cat([direction_dot_embed_predicted, direction_dot_embed_predicted + sorted_direction_dot_embed_error], dim=0).T, title="direction @ normed embed bad fit")

    # Plot the histogram
    print(len(direction_dot_embed_neg_values[0]) // 2)
    print(list(sorted([p for p in direction_dot_embed_neg_values[0] if p[0] < p[1]])))
    plt.hist(direction_dot_embed_diff_values[0], bins=30, edgecolor='black')
    plt.title("Distribution of (pos[i] - pos[j]) / (i - j)")
    plt.xlabel("(pos[i] - pos[j]) / (i - j)")
    plt.ylabel("Frequency")
    plt.show()

    line(direction_dot_embed_error.T, title="direction_dot_normed_embed_error")
    line(direction_dot_pos_embed.T, title="direction @ pos_embed")




plot_QK_cosine_similarity(model, querypos=1)


# In[ ]:


# Define the new y-values
y_data = np.array([ 0.1489, -0.0367,  0.0212,  0.0849,  0.0888, -0.0442,  0.0718,  0.1001,
         -0.0248, -0.0368,  0.1282, -0.0162, -0.0143,  0.0322,  0.0338,  0.0117,
          0.0433,  0.0052,  0.0020,  0.0147,  0.0292,  0.0332,  0.0057,  0.0179,
          0.0396,  0.0105,  0.0155,  0.0052,  0.0287,  0.0154,  0.0412,  0.0086,
         -0.0038,  0.0312,  0.0004,  0.0182,  0.0229,  0.0139,  0.0406, -0.0046,
          0.0535, -0.0230, -0.0240,  0.0587,  0.0244,  0.0140, -0.0212,  0.0431,
          0.0795, -0.0155,  0.0244, -0.0360, -0.0031,  0.1230, -0.0462,  0.0042,
          0.0041,  0.0068,  0.0727, -0.0303,  0.1436,  0.0374, -0.1088])

x_data = np.arange(len(y_data))

# Calculate the initial guess for the period of the sinusoids
sign_flips = np.sum(np.diff(np.sign(np.diff(y_data)) != 0))
initial_guess = [0.01, 0.01, 0.01, 2 * np.pi / (len(y_data) / (sign_flips / 2)), 0.01]

# Define the functions to fit
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def absolute_shift_func(x, a, b, c):
    return a * np.abs(x - b) + c

    # Define the more complex functions to fit
def linear_sinusoid_func(x, a, b, c, d):
    return (a * x + b) * np.sin(c * x + d)

def quadratic_sinusoid_func(x, a, b, c, d, e):
    return (a * x**2 + b * x + c) * np.sin(d * x + e)

def absolute_shift_sinusoid_func(x, a, b, c, d, e):
    return (a * np.abs(x - b) + c) * np.sin(d * x + e)

def linear_abs_sinusoid_func(x, a, b, c, d):
    return (a * x + b) * np.abs(np.sin(c * x + d))

def quadratic_abs_sinusoid_func(x, a, b, c, d, e):
    return (a * x**2 + b * x + c) * np.abs(np.sin(d * x + e))

def absolute_shift_abs_sinusoid_func(x, a, b, c, d, e):
    return (a * np.abs(x - b) + c) * np.abs(np.sin(d * x + e))

# Fit the data to the functions
popt_linear, _ = curve_fit(linear_func, x_data, y_data)
popt_quadratic, _ = curve_fit(quadratic_func, x_data, y_data)
popt_absolute_shift, _ = curve_fit(absolute_shift_func, x_data, y_data)

try:
    popt_linear_sinusoid, _ = curve_fit(linear_sinusoid_func, x_data, y_data, p0=initial_guess[:4])
except RuntimeError:
    popt_linear_sinusoid = None

try:
    popt_quadratic_sinusoid, _ = curve_fit(quadratic_sinusoid_func, x_data, y_data, p0=initial_guess)
except RuntimeError:
    popt_quadratic_sinusoid = None

try:
    popt_absolute_shift_sinusoid, _ = curve_fit(absolute_shift_sinusoid_func, x_data, y_data, p0=initial_guess)
except RuntimeError:
    popt_absolute_shift_sinusoid = None

try:
    popt_linear_abs_sinusoid, _ = curve_fit(linear_abs_sinusoid_func, x_data, y_data, p0=initial_guess[:4])
except RuntimeError:
    popt_linear_abs_sinusoid = None

try:
    popt_quadratic_abs_sinusoid, _ = curve_fit(quadratic_abs_sinusoid_func, x_data, y_data, p0=initial_guess)
except RuntimeError:
    popt_quadratic_abs_sinusoid = None

try:
    popt_absolute_shift_abs_sinusoid, _ = curve_fit(absolute_shift_abs_sinusoid_func, x_data, y_data, p0=initial_guess)
except RuntimeError:
    popt_absolute_shift_abs_sinusoid = None

# Compute the errors for each function
error_linear = y_data - linear_func(x_data, *popt_linear)
error_quadratic = y_data - quadratic_func(x_data, *popt_quadratic)
error_absolute_shift = y_data - absolute_shift_func(x_data, *popt_absolute_shift)

if popt_linear_sinusoid is not None:
    error_linear_sinusoid = y_data - linear_sinusoid_func(x_data, *popt_linear_sinusoid)

if popt_quadratic_sinusoid is not None:
    error_quadratic_sinusoid = y_data - quadratic_sinusoid_func(x_data, *popt_quadratic_sinusoid)

if popt_absolute_shift_sinusoid is not None:
    error_absolute_shift_sinusoid = y_data - absolute_shift_sinusoid_func(x_data, *popt_absolute_shift_sinusoid)

if popt_linear_abs_sinusoid is not None:
    error_linear_abs_sinusoid = y_data - linear_abs_sinusoid_func(x_data, *popt_linear_abs_sinusoid)

if popt_quadratic_abs_sinusoid is not None:
    error_quadratic_abs_sinusoid = y_data - quadratic_abs_sinusoid_func(x_data, *popt_quadratic_abs_sinusoid)

if popt_absolute_shift_abs_sinusoid is not None:
    error_absolute_shift_abs_sinusoid = y_data - absolute_shift_abs_sinusoid_func(x_data, *popt_absolute_shift_abs_sinusoid)

print(popt_linear, popt_quadratic, popt_absolute_shift, popt_linear_sinusoid, popt_quadratic_sinusoid, popt_absolute_shift_sinusoid, popt_linear_abs_sinusoid, popt_quadratic_abs_sinusoid, popt_absolute_shift_abs_sinusoid)
# Define the x values for the plot
x_plot = np.linspace(min(x_data), max(x_data), 1000)

# Plot the data and the fitted functions
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, color='black', label='Data')
plt.plot(x_plot, linear_func(x_plot, *popt_linear), label='Linear')
plt.plot(x_plot, quadratic_func(x_plot, *popt_quadratic), label='Quadratic')
plt.plot(x_plot, absolute_shift_func(x_plot, *popt_absolute_shift), label='Absolute shift')

if popt_linear_sinusoid is not None:
    plt.plot(x_plot, linear_sinusoid_func(x_plot, *popt_linear_sinusoid), label='Linear times a sinusoid')

if popt_quadratic_sinusoid is not None:
    plt.plot(x_plot, quadratic_sinusoid_func(x_plot, *popt_quadratic_sinusoid), label='Quadratic times a sinusoid')

if popt_absolute_shift_sinusoid is not None:
    plt.plot(x_plot, absolute_shift_sinusoid_func(x_plot, *popt_absolute_shift_sinusoid), label='Absolute shift times a sinusoid')

if popt_linear_abs_sinusoid is not None:
    plt.plot(x_plot, linear_abs_sinusoid_func(x_plot, *popt_linear_abs_sinusoid), label='Linear times abs of a sinusoid')

if popt_quadratic_abs_sinusoid is not None:
    plt.plot(x_plot, quadratic_abs_sinusoid_func(x_plot, *popt_quadratic_abs_sinusoid), label='Quadratic times abs of a sinusoid')

if popt_absolute_shift_abs_sinusoid is not None:
    plt.plot(x_plot, absolute_shift_abs_sinusoid_func(x_plot, *popt_absolute_shift_abs_sinusoid), label='Absolute shift times abs of a sinusoid')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[ ]:


# Set the new y_values
y_values = np.array([ 0.1489, -0.0367,  0.0212,  0.0849,  0.0888, -0.0442,  0.0718,  0.1001,
         -0.0248, -0.0368,  0.1282, -0.0162, -0.0143,  0.0322,  0.0338,  0.0117,
          0.0433,  0.0052,  0.0020,  0.0147,  0.0292,  0.0332,  0.0057,  0.0179,
          0.0396,  0.0105,  0.0155,  0.0052,  0.0287,  0.0154,  0.0412,  0.0086,
         -0.0038,  0.0312,  0.0004,  0.0182,  0.0229,  0.0139,  0.0406, -0.0046,
          0.0535, -0.0230, -0.0240,  0.0587,  0.0244,  0.0140, -0.0212,  0.0431,
          0.0795, -0.0155,  0.0244, -0.0360, -0.0031,  0.1230, -0.0462,  0.0042,
          0.0041,  0.0068,  0.0727, -0.0303,  0.1436,  0.0374, -0.1088])

# Compute the number of sign flips
num_sign_flips = np.sum(np.diff(np.sign(y_values)) != 0)

# Compute the initial guess for the period
initial_guess_period = len(y_values) / (num_sign_flips / 2)

initial_guess_period


# In[ ]:


# Fit the data to the functions
popt_linear, _ = curve_fit(linear_func, x_data, y_values)
popt_quadratic, _ = curve_fit(quadratic_func, x_data, y_values)
popt_absolute_shift, _ = curve_fit(absolute_shift_func, x_data, y_values)

# Initial guesses for the parameters
initial_guess = [0.01, 0.01, 2*np.pi/initial_guess_period, 0.01, 0.01]

# Fit the data to the functions
try:
    popt_linear_sinusoid, _ = curve_fit(linear_sinusoid_func, x_data, y_values, p0=initial_guess[:4])
except RuntimeError:
    popt_linear_sinusoid = None

try:
    popt_quadratic_sinusoid, _ = curve_fit(quadratic_sinusoid_func, x_data, y_values, p0=initial_guess)
except RuntimeError:
    popt_quadratic_sinusoid = None

try:
    popt_absolute_shift_sinusoid, _ = curve_fit(absolute_shift_sinusoid_func, x_data, y_values, p0=initial_guess)
except RuntimeError:
    popt_absolute_shift_sinusoid = None

try:
    popt_linear_abs_sinusoid, _ = curve_fit(linear_abs_sinusoid_func, x_data, y_values, p0=initial_guess[:4])
except RuntimeError:
    popt_linear_abs_sinusoid = None

try:
    popt_quadratic_abs_sinusoid, _ = curve_fit(quadratic_abs_sinusoid_func, x_data, y_values, p0=initial_guess)
except RuntimeError:
    popt_quadratic_abs_sinusoid = None

try:
    popt_absolute_shift_abs_sinusoid, _ = curve_fit(absolute_shift_abs_sinusoid_func, x_data, y_values, p0=initial_guess)
except RuntimeError:
    popt_absolute_shift_abs_sinusoid = None

popt_linear, popt_quadratic, popt_absolute_shift, popt_linear_sinusoid, popt_quadratic_sinusoid, popt_absolute_shift_sinusoid, popt_linear_abs_sinusoid, popt_quadratic_abs_sinusoid, popt_absolute_shift_abs_sinusoid


# In[ ]:


# Plot the data
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_values, color='black', label='Data')

# Plot the fitted functions
plt.plot(x_plot, linear_func(x_plot, *popt_linear), label='Linear')
plt.plot(x_plot, quadratic_func(x_plot, *popt_quadratic), label='Quadratic')
plt.plot(x_plot, absolute_shift_func(x_plot, *popt_absolute_shift), label='Absolute shift')

if popt_linear_sinusoid is not None:
    plt.plot(x_plot, linear_sinusoid_func(x_plot, *popt_linear_sinusoid), label='Linear times a sinusoid')

if popt_quadratic_sinusoid is not None:
    plt.plot(x_plot, quadratic_sinusoid_func(x_plot, *popt_quadratic_sinusoid), label='Quadratic times a sinusoid')

if popt_absolute_shift_sinusoid is not None:
    plt.plot(x_plot, absolute_shift_sinusoid_func(x_plot, *popt_absolute_shift_sinusoid), label='Absolute shift times a sinusoid')

if popt_linear_abs_sinusoid is not None:
    plt.plot(x_plot, linear_abs_sinusoid_func(x_plot, *popt_linear_abs_sinusoid), label='Linear times abs of a sinusoid')

if popt_quadratic_abs_sinusoid is not None:
    plt.plot(x_plot, quadratic_abs_sinusoid_func(x_plot, *popt_quadratic_abs_sinusoid), label='Quadratic times abs of a sinusoid')

if popt_absolute_shift_abs_sinusoid is not None:
    plt.plot(x_plot, absolute_shift_abs_sinusoid_func(x_plot, *popt_absolute_shift_abs_sinusoid), label='Absolute shift times abs of a sinusoid')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.title('Curve fitting for various models')
plt.show()

# Compute the errors for each function
error_linear = y_values - linear_func(x_data, *popt_linear)
error_quadratic = y_values - quadratic_func(x_data, *popt_quadratic)
error_absolute_shift = y_values - absolute_shift_func(x_data, *popt_absolute_shift)

if popt_linear_sinusoid is not None:
    error_linear_sinusoid = y_values - linear_sinusoid_func(x_data, *popt_linear_sinusoid)

if popt_quadratic_sinusoid is not None:
    error_quadratic_sinusoid = y_values - quadratic_sinusoid_func(x_data, *popt_quadratic_sinusoid)

if popt_absolute_shift_sinusoid is not None:
    error_absolute_shift_sinusoid = y_values - absolute_shift_sinusoid_func(x_data, *popt_absolute_shift_sinusoid)

if popt_linear_abs_sinusoid is not None:
    error_linear_abs_sinusoid = y_values - linear_abs_sinusoid_func(x_data, *popt_linear_abs_sinusoid)

if popt_quadratic_abs_sinusoid is not None:
    error_quadratic_abs_sinusoid = y_values - quadratic_abs_sinusoid_func(x_data, *popt_quadratic_abs_sinusoid)

if popt_absolute_shift_abs_sinusoid is not None:
    error_absolute_shift_abs_sinusoid = y_values - absolute_shift_abs_sinusoid_func(x_data, *popt_absolute_shift_abs_sinusoid)

# Plot the errors
plt.figure(figsize=(12, 8))

plt.plot(x_data, error_linear, label='Linear')
plt.plot(x_data, error_quadratic, label='Quadratic')
plt.plot(x_data, error_absolute_shift, label='Absolute shift')

if popt_linear_sinusoid is not None:
    plt.plot(x_data, error_linear_sinusoid, label='Linear times a sinusoid')

if popt_quadratic_sinusoid is not None:
    plt.plot(x_data, error_quadratic_sinusoid, label='Quadratic times a sinusoid')

if popt_absolute_shift_sinusoid is not None:
    plt.plot(x_data, error_absolute_shift_sinusoid, label='Absolute shift times a sinusoid')

if popt_linear_abs_sinusoid is not None:
    plt.plot(x_data, error_linear_abs_sinusoid, label='Linear times abs of a sinusoid')

if popt_quadratic_abs_sinusoid is not None:
    plt.plot(x_data, error_quadratic_abs_sinusoid, label='Quadratic times abs of a sinusoid')

if popt_absolute_shift_abs_sinusoid is not None:
    plt.plot(x_data, error_absolute_shift_abs_sinusoid, label='Absolute shift times abs of a sinusoid')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.title('Error for each fitted model')
plt.show()
# https://chat.openai.com/share/e5b12207-e550-4df0-944c-fc121eb09d96


# Let us look at what the full QK-circuit does in our model. In the below heatmap values at point (x,y) show how much attention is paid to token x from token y.
# 
# As you can see this map looks quite random. This is probably because, unlike in B&K's work, it is only the position and not the value of the token which is important for attention. At second position, more attention should always be paid to higher token valus, regardless of the exact value of the token.

# In[ ]:


plot_qk_heatmap(model)


# With this insight, let us draw new heatmaps showing the attention paid by tokens at position to tokens at position 0 and 1. These look like what we might expect, with higher values getting more attention.

# In[ ]:


for do_layernorm in (True, False): plot_qk_heatmap(model, keypos=0, querypos=1, do_layernorm=do_layernorm)


# In[ ]:


for do_layernorm in (True, False): plot_qk_heatmap(model, keypos=1, querypos=1, do_layernorm=do_layernorm)


# In[ ]:


plot_qk_heatmaps_normed(model, querypos=1)


# Somewhat surprisingly, the heatmap still looks the same for two tokens that are both at position 0. Perhaps, it was easiest to learn key- and value-embeddings that always map higher values to higher attention for any position.
# 
# If we assume that the OV-circuit is copying (see analysis below), then this could suggest:
# - the model's prediction for the second token will always be the first token.
# - for models that are trained to take the maximum of longer sequences, their prediction after a subsequence will be the maximum of that subsequence.

# In[ ]:


plot_qk_heatmap(model, keypos=0, querypos=0)


# Let us test the first hypothesis!
# First I will create some new data and let the model predict the logits:

# In[ ]:


data_train, data_test = get_data(sequence_length=2)
train_data_gen = make_generator_from_data(data_train, batch_size=128)
tokens = next(train_data_gen)
logits = model(tokens)


# Now let us check how often the model predicts that the second token will be the same as the first. This code takes the index of the highest logit from the model predictions for the second token and compares it to the first token. Then it divides how often the two coincide by the batch size to get an accuracy score.  

# In[ ]:


torch.sum((torch.argmax(logits, dim=2)[:,0] == tokens[:,0]).float()).item() / 128


# As we can see the model almost always predicts the second token to be equal to the first in this batch!

# In[ ]:


print(model)


# In[ ]:


len(list(model.parameters()))


# ## Analyzing the OV-Circuit
# 
# After the QK-Circuit determines which tokens will be attended to by a head, the OV-Circuit determines the computations that are applied to those tokens. In the above section we saw that the QK-Circuit will attend to the highest token in the sequence. As our model is doing the max operation, this means we can expect that OV-circuit's job will be to "copy" that token, by increasing the logits of the corresponding token.
# 
# I will be using two metrics to verify if the circuit is copying:
# 
# Firstly, plotting a heatmap of token attentions should display a clearly visible diagonal line, with values along the diagonal being higher than elsewhere. This means that a tokens increases its own logits.  
# 
# Another way of detecting a copying circuit is by [looking at the fraction of the circuit's eigenvalues](https://transformer-circuits.pub/2021/framework/index.html#summarizing-ovqk-matrices) that are positive.
# 
# Below are helper function for those metrics:

# In[ ]:


def plot_ov_heatmap(model, pos=-1, do_layernorm=True):
  attn = model.blocks[0].attn
  all_token_embeddings = model.embed(range(D_VOCAB))

  token_embeddings_at_pos = all_token_embeddings + model.pos_embed(all_token_embeddings)[:,pos,:] if pos > -1 else all_token_embeddings

  if do_layernorm:
        token_embeddings_at_pos = model.blocks[0].ln1(token_embeddings_at_pos)

  embeddings_value = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                          token_embeddings_at_pos, attn.W_V)

  embeddings_out = einsum("n_heads d_vocab d_model1, n_heads d_model1 d_model2 -> n_heads d_vocab d_model2",
                        embeddings_value, attn.W_O)

  ov_circuit_attn_heatmap = model.unembed(embeddings_out).detach()

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(ov_circuit_attn_heatmap[0], cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()


# In[ ]:


def get_full_ov_copying_score(model):
  full_OV_circuit = model.embed.W_E @ model.OV @ model.unembed.W_U
  full_OV_circuit_eigenvalues = full_OV_circuit.eigenvalues
  full_OV_copying_score = full_OV_circuit_eigenvalues.sum(dim=-1).real / full_OV_circuit_eigenvalues.abs().sum(dim=-1)
  return full_OV_copying_score.detach().item()


# As we can see, the heatmap looks as we expected, confirming our predicting about how the model works:

# In[ ]:


for pos in (0, 1):
    plot_ov_heatmap(model, pos=pos)


# The OV copying score is also very high:

# In[ ]:


get_full_ov_copying_score(model)


# In[ ]:


model.unembed


# In[ ]:





# # Exploring Longer Sequences
# 
# In this section I train two more models to take the max of sequences of lenght 3 and 6. This is to verify that they are using the same algorithm as the previous model, and the test the earlier hypothesis that models for larger sequences would also learn to predict the max of subsequences.

# ### Training New Models

# In[ ]:


cfg_3t = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=3,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=SEED,
    device=DEVICE,
    attn_only=True
)
model_3t = HookedTransformer(cfg_3t, move_to_device=True)


# In[ ]:


train_losses = train_model(model_3t, n_epochs=1000, batch_size=128, sequence_length=3)


# In[ ]:


train_losses = train_model(model_3t, n_epochs=1000, batch_size=128, sequence_length=3)


# In[ ]:


line(train_losses, xaxis="Epoch", yaxis="Loss")


# In[ ]:


cfg_6t = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=6,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=SEED,
    device=DEVICE,
    attn_only=True
)
model_6t = HookedTransformer(cfg_6t, move_to_device=True)


# In[ ]:


def train_model_for_large_sequence_length(
    model,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    sequence_length=2
  ):
  lr = 1e-3
  betas = (.9, .999)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

  train_losses = []

  train_data_gen = large_data_gen(n_digits=D_VOCAB, sequence_length=sequence_length, batch_size=batch_size, context="train")

  for epoch in tqdm.tqdm(range(n_epochs)):
    epoch_losses = []
    for _ in range(batches_per_epoch):
      tokens = next(train_data_gen)
      logits = model(tokens)
      losses = loss_fn(logits, tokens, return_per_token=True)
      losses.mean().backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_losses.extend(losses.detach())

    train_losses.append(np.mean(epoch_losses))

    if epoch % 10 == 0:
      print(f'Epoch {epoch} train loss: {train_losses[-1]}')

  model.eval()
  test_data_gen = large_data_gen(n_digits=D_VOCAB, sequence_length=sequence_length, batch_size=batch_size * 20, context="test")
  data_test = next(test_data_gen)
  logits = model(data_test)
  acc = acc_fn(logits, data_test)

  print(f"Test accuracy after training: {acc}")

  return train_losses


# In[ ]:


train_losses = train_model_for_large_sequence_length(model_6t, n_epochs=1000, batch_size=128, sequence_length=6)


# In[ ]:


line(train_losses, xaxis="Epoch", yaxis="Loss")


# ## Sequence Length 3

# ### Attention Patterns
# 
# The patterns we see here fit with out predictions. Tokens at every position attend to the highest value token.

# In[ ]:


data_train, data_test = get_data(sequence_length=3)
train_data_gen = make_generator_from_data(data_train, batch_size=128)
tokens = next(train_data_gen)
logits, cache = model_3t.run_with_cache(tokens)
print("Loss:", loss_fn(logits, tokens).item())


# In[ ]:


attention_pattern = cache["pattern", 0, "attn"]


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[0])), attention=attention_pattern[0])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[1])), attention=attention_pattern[1])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[2])), attention=attention_pattern[2])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[3])), attention=attention_pattern[3])


# ### Circuit Analysis
# 
# As there more positional encoding now we plot the average over all key positions for a given query position.
# 
# We see a similar pattern as with the previous model at every position. However, it seems to be slightly less pronounced for earlier positions. This suggest that we might see an even weaker effect when investigating the model for sequences of lenght 6 later.

# In[ ]:


plot_avg_qk_heatmap(model_3t, list(range(3)), querypos=0)


# In[ ]:


plot_avg_qk_heatmap(model_3t, list(range(3)), querypos=1)


# In[ ]:


plot_avg_qk_heatmap(model_3t, list(range(3)), querypos=2)


# In[ ]:


plot_ov_heatmap(model_3t, pos=0)


# In[ ]:


plot_ov_heatmap(model_3t, pos=1)


# In[ ]:


plot_ov_heatmap(model_3t, pos=2)


# ### Evaluating Subsequence Accuracy
# 
# Here I test the hypothesis that the model also learns to take the max of subsequences (with which I mean a sequences starting at position 0 and endind somewhwere before the sequence length). We already saw that for sequence lenght 2, the model predicts the second element to be the same as the first which can be seen as taking the max over the subsequence of length 1. As the model I am investigating here takes sequences of lenght 3, I can now also look at the model's prediction for subsequences of length 2:

# In[ ]:


data_train, data_test = get_data(sequence_length=3)
train_data_gen = make_generator_from_data(data_train, batch_size=128)
tokens = next(train_data_gen)
logits = model_3t(tokens)


# The following function evaluates the accuracy of predictions for subsequences:

# In[ ]:


def subsequence_accuracy(logits, tokens, ss_length, batch_size=128):
  correct_preds = torch.sum((torch.argmax(logits, dim=2)[:,ss_length] == torch.max(tokens[:,:ss_length+1], dim=1)[0]).float()).item()
  return correct_preds / batch_size


# As before, the model always predicts the first and second token to be same:

# In[ ]:


subsequence_accuracy(logits, tokens, 0)


# It also achieves a reasonably high accuracy of 92% on the task of predict the max on the subsequence of length 2, though this is noticeably lower than the accuracy for taking the max over the whole sequence:

# In[ ]:


subsequence_accuracy(logits, tokens, 1)


# ## Sequence Length 6
# 

# ### Attention Pattern
# 
# From the example attention patterns, it does not look like the model is still learning to take the subsequence maximum.

# In[ ]:


tokens = next(large_data_gen())
logits, cache = model_6t.run_with_cache(tokens)


# In[ ]:


attention_pattern = cache["pattern", 0, "attn"]


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[0])), attention=attention_pattern[0])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[1])), attention=attention_pattern[1])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[2])), attention=attention_pattern[2])


# In[ ]:


cv.attention.attention_heads(tokens=list(map(lambda t: str(t.item()), tokens[3])), attention=attention_pattern[3])


# ### Circuit Analysis
# 
# For the QK circuits, we see that only the attention from position 6 (index 5) looks as before. For most other positions the pattern looks random, and for position 5 (index 4) it is looks like lower tokens are getting higher attention! In the next section we will see that the model gets very low accuracy for taking the max of subsequences up to position 5, which is what we would expect from this attention pattern.  

# In[ ]:


plot_avg_qk_heatmap(model_6t, list(range(6)), querypos=0)


# In[ ]:


plot_avg_qk_heatmap(model_6t, list(range(6)), querypos=1)


# In[ ]:


plot_avg_qk_heatmap(model_6t, list(range(6)), querypos=2)


# In[ ]:


plot_avg_qk_heatmap(model_6t, list(range(6)), querypos=3)


# In[ ]:


plot_avg_qk_heatmap(model_6t, list(range(6)), querypos=4)


# In[ ]:


plot_avg_qk_heatmap(model_6t, list(range(6)), querypos=5)


# In[ ]:


plot_ov_heatmap(model_6t, pos=0)


# In[ ]:


plot_ov_heatmap(model_6t, pos=1)


# In[ ]:


plot_ov_heatmap(model_6t, pos=2)


# In[ ]:


plot_ov_heatmap(model_6t, pos=3)


# In[ ]:


plot_ov_heatmap(model_6t, pos=4)


# In[ ]:


plot_ov_heatmap(model_6t, pos=5)


# ### Evaluating Subsequence Accuracy
# 
# The following bar plot shows the accuracy for predicting the max of a subsequence of lenght n. It confirms the suspicion of our circuit analysis, that the model's capabilities did not generalize to subsequences.

# In[ ]:


ss_accuracies = [subsequence_accuracy(logits, tokens, i) for i in range(6)]


# In[ ]:


plt.bar(list(range(1,7)), ss_accuracies, align='center')
plt.show()


# # Removing The Positional Embedding

# ## Training a model for sequences of length 6 with no positional embedding

# In[ ]:


def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.
    model.pos_embed.W_pos.requires_grad = False


# In[ ]:


cfg_6t_no_pos = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=6,
    d_vocab=D_VOCAB,
    act_fn="relu",
    seed=SEED,
    device=DEVICE,
    attn_only=True
)
model_6t_no_pos = HookedTransformer(cfg_6t_no_pos, move_to_device=True)
deactivate_position(model_6t_no_pos)


# In[ ]:


train_losses = train_model_for_large_sequence_length(model_6t_no_pos, n_epochs=1000, batch_size=128, sequence_length=6)


# In[ ]:


line(train_losses, xaxis="Epoch", yaxis="Loss")


# ## QK Circuit Without Positional Embedding
# 
# As we can see, the QK heatmap is now interpretable without the positional embedding. Once again we see that more attention is paid to tokens of higher cardinality.

# In[ ]:


plot_qk_heatmap(model_6t_no_pos)


# ## Subsequence Accuracy
# 
# Without the positional embedding the model can not tell which token is the last one. Hence, it learns to take the maximum of every subsequence.

# In[ ]:


tokens = next(large_data_gen())
logits, cache = model_6t_no_pos.run_with_cache(tokens)


# In[ ]:


ss_accuracies = [subsequence_accuracy(logits, tokens, i) for i in range(6)]
plt.bar(list(range(1,7)), ss_accuracies, align='center')
plt.show()

# %%