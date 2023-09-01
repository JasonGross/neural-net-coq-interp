# %%
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
import wandb
import datetime

from coq_export_utils import strify
# from analysis_utils import line, summarize, plot_QK_cosine_similarity, \
#     analyze_svd, calculate_OV_of_pos_embed, calculate_attn, calculate_attn_by_pos, \
#     calculate_copying, calculate_copying_with_pos, calculate_embed_and_pos_embed_overlap, \
#     calculate_embed_overlap, calculate_pos_embed_overlap, check_monotonicity, \
#     compute_slack, plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from training_utils import compute_all_tokens, get_data, make_generator_from_data

import os, sys
from importlib import reload

from scipy.optimize import curve_fit



# %%

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 1
N_HEADS = 1
D_MODEL = 32
D_HEAD = 32
D_MLP = None
D_VOCAB = 64
SEED = 123

ALWAYS_TRAIN_MODEL = False
IN_COLAB = False
SAVE_IN_GOOGLE_DRIVE = False
OVERWRITE_DATA = True
TRAIN_MODEL_IF_CANT_LOAD = True



# %%

# %%

simpler_cfg = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=5,
    d_vocab=D_VOCAB,
    seed=SEED,
    device=DEVICE,
    attn_only=True,
    normalization_type=None,
)
# %%

model = HookedTransformer(simpler_cfg).to(DEVICE)

# %%

# test large_data_gen
gen = large_data_gen(n_digits=10, sequence_length=5, batch_size=128, context="train", device=DEVICE, adjacent_fraction=0.5)
gen.__next__()

# %%

# where we save the model
if IN_COLAB:
    # if SAVE_IN_GOOGLE_DRIVE:
    #     from google.colab import drive
    #     drive.mount('/content/drive/')
    #     PTH_BASE_PATH = Path('/content/drive/MyDrive/Colab Notebooks/')
    # else:
    #     PTH_BASE_PATH = Path("/workspace/_scratch/")
    pass
else:
    PTH_BASE_PATH = Path(os.getcwd())

PTH_BASE_PATH = PTH_BASE_PATH / 'transformer-takes-max'

if not os.path.exists(PTH_BASE_PATH):
    os.makedirs(PTH_BASE_PATH)

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_PTH_PATH = PTH_BASE_PATH / f'max-of-n-{datetime_str}.pth'


TRAIN_MODEL = ALWAYS_TRAIN_MODEL
if not ALWAYS_TRAIN_MODEL:
    try:
        cached_data = torch.load(MODEL_PTH_PATH)
        model.load_state_dict(cached_data['model'])
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
    wandb.init(project=f'neural-net-coq-interp-max-{model.cfg.n_ctx}')

    simpler_train_losses = train_model(model, n_epochs=50000, batch_size=256, batches_per_epoch=10, 
            adjacent_fraction=0.3, use_complete_data=False, device=DEVICE, use_wandb=True)

    wandb.finish()


# In[ ]:


if TRAIN_MODEL:
    data = {
                "model":model.state_dict(),
                "config": model.cfg,
                "train_losses": simpler_train_losses,
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

# %%