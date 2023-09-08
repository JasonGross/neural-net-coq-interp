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


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
import circuitsvis as cv
from fancy_einsum import einsum
import dataclasses
from pathlib import Path
from IPython import get_ipython

from coq_export_utils import strify
from analysis_utils import line, summarize, plot_QK_cosine_similarity, \
    analyze_svd, calculate_OV_of_pos_embed, calculate_attn, calculate_attn_by_pos, \
    calculate_copying, calculate_copying_with_pos, calculate_embed_and_pos_embed_overlap, \
    calculate_embed_overlap, calculate_pos_embed_overlap, check_monotonicity, \
    compute_slack, plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
import analysis_utils
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from training_utils import compute_all_tokens, get_data, make_generator_from_data

import os, sys
from importlib import reload

from scipy.optimize import curve_fit


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
UNDERTRAINED_MODEL_PTH_PATH = PTH_BASE_PATH / 'max-of-two-undertrained.pth'

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


undertrain_simpler_cfg = HookedTransformerConfig(
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
undertrain_simpler_model = HookedTransformer(undertrain_simpler_cfg, move_to_device=True)


# Disable the biases, as we don't need them for this task and it makes things easier to interpret.

# In[ ]:


for name, param in undertrain_simpler_model.named_parameters():
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
        cached_data = torch.load(UNDERTRAINED_MODEL_PTH_PATH)
        undertrain_simpler_model.load_state_dict(cached_data['model'])
        #model_checkpoints = cached_data["checkpoints"]
        #checkpoint_epochs = cached_data["checkpoint_epochs"]
        #test_losses = cached_data['test_losses']
        undertrain_simpler_train_losses = cached_data['train_losses']
        #train_indices = cached_data["train_indices"]
        #test_indices = cached_data["test_indices"]
    except Exception as e:
        print(e)
        TRAIN_MODEL = TRAIN_MODEL_IF_CANT_LOAD


# In[ ]:


if TRAIN_MODEL:
    undertrain_simpler_train_losses = train_model(undertrain_simpler_model, n_epochs=32, batch_size=128, adjacent_fraction=True)


# In[ ]:


if TRAIN_MODEL:
    data = {
                "model":undertrain_simpler_model.state_dict(),
                "config": undertrain_simpler_model.cfg,
                "train_losses": undertrain_simpler_train_losses,
            }
    if OVERWRITE_DATA or not os.path.exists(UNDERTRAINED_MODEL_PTH_PATH):
        torch.save(
            data,
            UNDERTRAINED_MODEL_PTH_PATH)
    else:
        print(f'WARNING: Not overwriting {UNDERTRAINED_MODEL_PTH_PATH} because it already exists.')
        ext = 0
        while os.path.exists(f"{UNDERTRAINED_MODEL_PTH_PATH}.{ext}"):
            ext += 1
        torch.save(
            data,
            f"{UNDERTRAINED_MODEL_PTH_PATH}.{ext}")
        print(f'WARNING: Wrote to {UNDERTRAINED_MODEL_PTH_PATH}.{ext} instead.')


# As we can see accuracy is high and loss is low.

# In[ ]:


line(undertrain_simpler_train_losses, xaxis="Epoch", yaxis="Loss")


EXPORT_TO_COQ = True
if EXPORT_TO_COQ:
    coq_export_params(undertrain_simpler_model)

# In[ ]:


all_integers = compute_all_tokens(undertrain_simpler_model)
all_integers.shape
#model(torch.tensor([0, 1]))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'all_integers_result = undertrain_simpler_model(all_integers)\n')


# In[ ]:


print(f"loss: {loss_fn(all_integers_result, all_integers)}")
print(f"acc: {acc_fn(all_integers_result, all_integers)}")


# In[ ]:


all_integers_ans = all_integers_result[:,-1].cpu().detach()
ans = all_integers_ans.argmax(dim=-1)
expected = all_integers.max(dim=-1).values
alt_expected = all_integers.min(dim=-1).values
correct_idxs = (ans == expected)
very_wrong_idxs = ~((ans == expected) | (ans == alt_expected))
print(all_integers[~correct_idxs], very_wrong_idxs.sum())
# %%

calculate_attn(undertrain_simpler_model)
# %%
analysis_utils.calculate_copying(undertrain_simpler_model)
# %%
