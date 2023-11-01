# %% [markdown]
# <a href="https://colab.research.google.com/github/JasonGross/neural-net-coq-interp/blob/main/October_2023_Monthly_Algorithmic_Challenge_Sorted_List_Jason%2C_Thomas%2C_Rajashree.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # October 2023 Mechanistic Interpretability Challange: Sorted List
#
# The <a href="https://colab.research.google.com/drive/1IygYxp98JGvMRLNmnEbHjEGUBAxBkLeU">problem</a> is to interpret a model which has been trained to sort a list. The model is fed sequences like:```[11, 2, 5, 0, 3, 9, SEP, 0, 2, 3, 5, 9, 11]``` and has been trained to predict each element in the sorted list (in other words, the output at the `SEP` token should be a prediction of `0`, the output at `0` should be a prediction of `2`, etc).
#
#
# **TL;DR**: We’re obsessed with the question “what if we gear our interpretability analysis at making formal guarantees about model behavior”. We present a sketch of a formal guarantee that P(model outputs the first token (of ten tokens) correctly) >= 59%. 
#
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sorted-problem.png" width="350">
#
# Content flow: Our Approach, Set up, Graphs, Proofs, Conclusion.

# %% [markdown]
# # Our Approach
#
# ## Introduction
#
# Given a model M, and output behavior B that we care about, the standard workflow for mechanistic interpretability goes something like this: 
#
# 1. M is a very large computation graph, so we find subgraph M’ that is relevant to B. Then we make arguments to show that M’ is a reasonable factoring wrt B. A key example might be ablating irrelevant heads. Let’s call these moves independence relaxations. 
#
# 2. M’ is still a pretty large computational graph, but easier to analyze. Now we can isolate important properties P of M' by how they impact B. For example, in patching the classifying property is the result of running the irrelevant parts of the model on a sample from the corrupted distributions and the relevant parts of the model on a sample from the correct distribution. Let’s call these moves finding classifying properties. 
#
# Substantial analysis of independence relaxations and classifying properties can paint a compelling picture of model behavior. But we may still not be able to make any formal guarantees akin to “with probability X, M will do B because P is so and so”.
# So far, the best we've got is informal arguments substantiated by random sampling.
#
# On the other hand, a formal guarantee is a **precise** statement about M wrt B that can be **verified**. We think that usefulness towards making a formal guarantee can be a metric for evaluating interpretability analyses!
#
# The following interpretability analyses are geared towards making guarantees. As usual, we’ll present a hypothesis for how the model works, and gesture at evidence for our hypothesis. Beyond this, we’ll identify the computation that would tie up the evidence into a guarantee. Finally, we’ll demonstrate how we iteratively develop our independence relaxations and classifying properties to make stronger guarantees. 
#
# The methodology used here is being developed as a part of a larger project of Jason Gross, Rajashree Agrawal, and Thomas Kwa investigating formalizations of tiny transformers. We’ll publish an in depth analysis soon. This post is a short attempt at applying the methodology for fun.

# ## Initial Hypothesis
# To start off, the rough algorithm for the model seems to be: find the smallest value not smaller than the current token, which hasn't been "cancelled" by an equivalent copy appearing already in the sorted list
#
# Head 0 is mostly doing the cancelling, while head 1 is mostly doing the copying, except for token values around 28--37 where head 0 is doing copying and head 1 is doing nothing.
#
# Additional notes:
# - The skip connection (embed -> unembed) is a small bias against the current token, a smaller bias against numbers less than the current token, and a smaller bias in favor of numbers greater than the current token.
# - The layernorm scaling is fairly uniform at positions on the unsorted list but a bit less uniform on the sorted prefix (after the SEP token)
# - It seems like the cancelling doesn't work that well when there are tokens in the range where the head behavior is swapped, so most of the computation should work even in the absence of cancelling.  The cancelling presumably just tips the scales in marginal cases (and cases where there are duplicates), since most of the head's capacity is devoted to positive copying when such tokens are present.

# ## Formal Assertions
# 
# To validate the hypothesis, we need to establish a the following assertions:
#
# Let $S$ be the range of swapped tokens, $S = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37]$.
#
# Let $h_{k}$ denote head 0 for tokens $k \in S$ and head 1 otherwise.
#
# 1. When the query token is SEP in position 10, we find the minimum of the sequence. (A1)
# 2. When the query token is 50 in position 19, we emit 50. (A2)
# 3. When the query token is anything other than 50 in position 19, we emit the maximum of the sequence. (A3)
# 4. When the query is in positions between 11 and 18 inclusive, we follow the rough algorithm above. (A4)
#
# ## Guarantees Methodology
#
# We breakdown each of the assertions by evidence and computation required to make a formal guarantee.
#
# Argument of A1:
# 1. Attention by head $h_{k}$ is mostly monotonic decreasing in the value of the token $k$. Evidence: See graph of attention from SEP position.
# 2. The OV circuit on head $h_{k}$ copies the value $k$ more than anything else. Evidence: See graphs of OV circuits.
# 3. We pay enough more attention to the smallest token than to everything else combined and copy $k$ enough more than anything else that when we combine the effects of the two heads on other tokens, we still manage to copy the correct token. Computation: See attempts. 
# 
#
# Argument of A2:
#
# 1. The copying effects from attending to 50 in position 19 and one additional 50 in some position before 10 gives enough difference between 50 and anything else that we don't care what happens elsewhere. Evidence: See graph of layernorm scaling. 
# 2. Computation: TODO.
# 
#
# Argument of A3:
#
# 1. Attention by head $h_{k}$ in position 19 is mostly monotonic increasing in the value of the token $k$. Evidence: See graphs of attention. 
# 2. The OV circuit on head $h_{k}$ copies the value $k$ more than anything else. Evidence: See graphs of OV circuits.
# 3. We pay enough more attention to the largest token than to everything else combined and copy $k$ enough more than anything else that when we combine the effects of the two heads on other tokens, we still manage to copy the correct token. Computation: TODO.
# 
#
# Argument of A4:
#
# For all of the following, evidence is in graphs of attention, and the computation is a TODO.
# 1. For $k_1, k_2, q \not\in S$ with $k_1 < q \le k_2$, head 1 pays more attention to $k_2$ in positions before 10 than to $k_1$ in any position.
# 2. For $k_1, k_2, q \not\in S$ with $k_1 = q \le k_2$, head 1 pays more attention to $k_2$ in positions before 10 than to $k_1$ in positions after 10. 
# 3. For $k_1, k_2, q \not\in S$ with $q \le k_1 < k_2$, head 1 pays more attention to $k_1$ in positions before 10 than to $k_2$ in positions before 10. 
# 4. For $k_2 \in S$ with $k_1 < q \le k_2$, head 0 pays more attention to $k_2$ in positions before 10 than to $k_1$ in any position. 
# 5. For $k_2 \in S$ with $k_1 = q \le k_2$, head 0 pays more attention to $k_2$ in positions before 10 than to $k_1$ in positions after 10. 
# 6. For $k_1 \in S$ with $q \le k_1 < k_2$, head 0 pays more attention to $k_1$ in positions before 10 than to $k_2$ in positions before 10. 
#
#
# %% [markdown]
# # Code
# Can be run without reading. Results are in a separate section.
#
# %% [markdown]
# ## Model
# The model is attention-only, with 1 layer, and 2 attention heads per layer. It was trained with layernorm, weight decay, and an Adam optimizer with linearly decaying learning rate.
#

# %%
try:
    import google.colab # type: ignore
    IN_COLAB = True
except:
    IN_COLAB = False

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys

if IN_COLAB:
    # Install packages
    %pip install einops
    %pip install jaxtyping
    %pip install transformer_lens
    %pip install git+https://github.com/callummcdougall/eindex.git
    %pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

    # Code to download the necessary files (e.g. solutions, test funcs)
    import os, sys
    if not os.path.exists("chapter1_transformers"):
        !curl -o /content/main.zip https://codeload.github.com/callummcdougall/ARENA_2.0/zip/refs/heads/main
        !unzip /content/main.zip 'ARENA_2.0-main/chapter1_transformers/exercises/*'
        sys.path.append("/content/ARENA_2.0-main/chapter1_transformers/exercises")
        os.remove("/content/main.zip")
        os.rename("ARENA_2.0-main/chapter1_transformers", "chapter1_transformers")
        os.rmdir("ARENA_2.0-main")
        os.chdir("chapter1_transformers/exercises")
else:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    import os, sys
    if not os.path.exists("chapter1_transformers"):
        !curl -o main.zip https://codeload.github.com/callummcdougall/ARENA_2.0/zip/refs/heads/main
        !unzip main.zip 'ARENA_2.0-main/chapter1_transformers/exercises/*'
        os.remove("main.zip")
        os.rename("ARENA_2.0-main/chapter1_transformers", "chapter1_transformers")
        sys.path.append(f"{os.getcwd()}/chapter1_transformers/exercises")
        os.rmdir("ARENA_2.0-main")

# %%
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "october23_sorted_list"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.october23_sorted_list.dataset import SortedListDataset
from monthly_algorithmic_problems.october23_sorted_list.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")



# %%
filename = section_dir / "sorted_list_model.pt"

model = create_model(
    list_len=10,
    max_value=50,
    seed=0,
    d_model=96,
    d_head=48,
    n_layers=1,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);

# %%
from eindex import eindex

N = 500
dataset = SortedListDataset(size=N, list_len=10, max_value=50, seed=43)


# %% [markdown]
# ## Analysis Utils
#
# %%
#imports
from einops import einsum, rearrange, reduce
import torch
from matplotlib.widgets import Slider
from matplotlib import colors
from tqdm.auto import tqdm
from matplotlib.animation import FuncAnimation
import circuitsvis as cv
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import numpy as np
import transformer_lens.utils as utils
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer
import math

# ## Utils

# %%
# image utils
def imshow(tensor, renderer=None, xaxis="", yaxis="", color_continuous_scale="RdBu", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale=color_continuous_scale, labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", line_labels=None, **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, y=line_labels, **kwargs).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


def hist(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.histogram(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def make_tickvals_text(step=10, include_lastn=1, skip_lastn=0, dataset=dataset):
    all_tickvals_text = list(enumerate(dataset.vocab))
    tickvals_indices = list(range(0, len(all_tickvals_text) - 1 - include_lastn - skip_lastn, step)) + list(range(len(all_tickvals_text) - 1 - include_lastn, len(all_tickvals_text)))
    tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
    ticktext = [all_tickvals_text[i][1] for i in tickvals_indices]
    return tickvals, ticktext

# %% [markdown]
# ## Computation of Matrices
# Used in both visualization and as a cached computation in proofs)

# %%
@torch.no_grad()
def nanify_attn_sep_loc(attn_all, outdim='h qpos kpos qtok ktok', sep_loc=dataset.list_len, nanify_bad_self_attention=False):
    ref_outdim = 'h qpos kpos qtok ktok'
    attn_all = rearrange(attn_all, f'{outdim} -> {ref_outdim}')
    # attention paid from the SEP position always has query = SEP
    attn_all[:, sep_loc, :, :-1, :] = float('nan')
    # attention paid to the SEP position is always to SEP
    attn_all[:, :, sep_loc, :, :-1] = float('nan')
    # attention paid from non-SEP positions never has query = SEP
    attn_all[:, :sep_loc, :, -1, :], attn_all[:, sep_loc+1:, :, -1, :] = float('nan'), float('nan')
    # attention paid to non-SEP positions never has key = SEP
    attn_all[:, :, :sep_loc, :, -1], attn_all[:, :, sep_loc+1:, :, -1] = float('nan'), float('nan')
    # self-attention always has query token = key token
    if nanify_bad_self_attention:
        for h in range(attn_all.shape[0]):
            for pos in range(attn_all.shape[1]):
                for qtok in range(attn_all.shape[3]):
                    for ktok in range(attn_all.shape[4]):
                        if qtok != ktok:
                            attn_all[h, pos, pos, qtok, ktok] = float('nan')
    attn_all = rearrange(attn_all, f'{ref_outdim} -> {outdim}')
    return attn_all

@torch.no_grad()
def compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=None, nanify_bad_self_attention=False):
    resid = model.blocks[0].ln1(model.W_pos[:,None,:] + model.W_E[None,:,:])
    outdimsorted = [i for i in sorted(outdim.split(' ')) if i != '']
    assert outdimsorted == list(sorted(['h', 'qpos', 'kpos', 'qtok', 'ktok'])), f"outdim must be 'h qpos kpos qtok ktok' in some order, got {outdim}"
    q_all = einsum(resid,
                   model.W_Q[0,:,:,:],
                   'qpos qtok d_model_q, h d_model_q d_head -> qpos qtok h d_head') + model.b_Q[0]
    k_all = einsum(resid,
                   model.W_K[0,:,:,:],
                   'kpos ktok d_model_k, h d_model_k d_head -> kpos ktok h d_head') + model.b_K[0]
    attn_all = einsum(q_all, k_all, f'qpos qtok h d_head, kpos ktok h d_head -> {outdim}') \
        / model.blocks[0].attn.attn_scale
    attn_all_max = reduce(attn_all, f"{outdim} -> {outdim.replace('kpos', '()').replace('ktok', '()')}", 'max')
    # print(attn_all_max.shape)
    # print(attn_all.shape)
    # #attn_all[:,dataset.list_len:-1,:,:-1,:]
    attn_all = attn_all - attn_all_max
    if nanify_sep_loc is not None: attn_all = nanify_attn_sep_loc(attn_all, outdim=outdim, sep_loc=nanify_sep_loc, nanify_bad_self_attention=nanify_bad_self_attention)
    return attn_all

# %%
@torch.no_grad()
def compute_attention_patterns(model, sep_pos=dataset.list_len, nanify_impossible_nonmin=True, num_min=1):
    '''
    returns post-softmax attention paid to the minimum token, the nonminimum token, and to the sep token
    in shape (head, 2, mintok, nonmintok, 3)

    The 2 is for computing both the min attention paid to the min tok and the max attention paid to the min tok
    The 3 is for mintok, nonmintok, and sep
    '''
    attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=dataset.list_len)
    n_heads, _, _, d_vocab, _ = attn_all.shape
    d_vocab -= 1 # remove sep token
    num_nonmin = sep_pos - num_min
    attn_all = attn_all[:, sep_pos, :sep_pos+1, -1, :]
    # (h, kpos, ktok)
    # scores is presoftmax
    # print(attn_all[:, :, 22])
    # print(attn_all[:, -1:, -1])
    attn_scores = t.zeros((n_heads, 2, d_vocab, d_vocab, sep_pos+1), device=attn_all.device)
    # set attention paid to sep token
    attn_scores[:, :, :, :, -1] = attn_all[:, None, None, None, -1, -1]
    # remove sep token
    attn_all = attn_all[:, :-1, :-1]
    # sort attn_all along dim=1
    attn_all, _ = attn_all.sort(dim=1)
    # set min attention paid to min token across all positions
    attn_scores[:, 0, :, :, :num_min] = rearrange(attn_all[:, :num_min, :, None], 'h kpos ktokmin ktoknonmin -> h ktokmin ktoknonmin kpos')
    # set max attention paid to min token across all positions
    attn_scores[:, 1, :, :, :num_min] = rearrange(attn_all[:, -num_min:, :, None], 'h kpos ktokmin ktoknonmin -> h ktokmin ktoknonmin kpos')
    # set max attention paid to non-min-token across all positions (corresponds to min attention paid to min token)
    attn_scores[:, 0, :, :, num_min:-1] = rearrange(attn_all[:, -num_nonmin:, :, None], 'h kpos ktoknonmin ktokmin -> h ktokmin ktoknonmin kpos')
    # set min attention paid to non-min-token across all positions (corresponds to max attention paid to min token)
    attn_scores[:, 1, :, :, num_min:-1] = rearrange(attn_all[:, :num_nonmin, :, None], 'h kpos ktoknonmin ktokmin -> h ktokmin ktoknonmin kpos')

    # compute softmax
    attn_pattern_expanded = attn_scores.softmax(dim=-1)

    # remove rows corresponding to impossible non-min tokens
    if nanify_impossible_nonmin:
        for mintok in range(d_vocab):
            attn_pattern_expanded[:, :, mintok, :mintok, :] = float('nan')

    attn_pattern = t.zeros((n_heads, 2, d_vocab, d_vocab, 3), device=attn_all.device)
    attn_pattern[:, :, :, :, 0] = attn_pattern_expanded[:, :, :, :, :num_min].sum(dim=-1)
    attn_pattern[:, :, :, :, 1] = attn_pattern_expanded[:, :, :, :, num_min:-1].sum(dim=-1)
    attn_pattern[:, :, :, :, 2] = attn_pattern_expanded[:, :, :, :, -1]

    # for min = nonmin, move all attention to min
    for mintok in range(d_vocab):
        attn_pattern[:, :, mintok, mintok, 0] += attn_pattern[:, :, mintok, mintok, 1]
        attn_pattern[:, :, mintok, mintok, 1] = 0

    return attn_pattern
# %%
@torch.no_grad()
def compute_3way_attention_patterns(model, sep_pos=dataset.list_len, nanify_impossible_nonmin=True, num_min=1, num_nonmin1=1, attn_all=None, attn_all_outdim=None):
    '''
    returns post-softmax attention paid to the minimum token, two nonminimum tokens, and to the sep token
    in shape (head, 3, 2, mintok, nonmintok1, nonmintok2, 4)

    The 3, 2 is for the permutations of attention ordering by position, min vs nonmin1 vs nonmin2 in the lowest attention positions, and then which of the remaining two is in the highest attention position.
    The 4 is for mintok, nonmintok1, nonmintok2, and sep
    '''
    desired_attn_all_outdim = 'h qpos kpos qtok ktok'
    if attn_all is None:
        return compute_3way_attention_patterns(
            model, sep_pos=sep_pos, nanify_impossible_nonmin=nanify_impossible_nonmin, num_min=num_min, num_nonmin1=num_nonmin1,
            attn_all=compute_attn_all(model, outdim=desired_attn_all_outdim, nanify_sep_loc=sep_pos), attn_all_outdim=desired_attn_all_outdim)
    assert attn_all_outdim is not None
    attn_all = rearrange(attn_all, f'{attn_all_outdim} -> {desired_attn_all_outdim}')
    n_heads, _, _, d_vocab, _ = attn_all.shape
    d_vocab -= 1 # remove sep token
    num_nonmin2 = sep_pos - num_min
    attn_all = attn_all[:, sep_pos, :sep_pos+1, -1, :]
    # (h, kpos, ktok)
    # scores is presoftmax
    attn_scores = t.zeros((n_heads, 3, 2, d_vocab, d_vocab, d_vocab, sep_pos+1), device=attn_all.device)
    # set attention paid to sep token
    attn_scores[:, :, :, :, :, :, -1] = attn_all[:, None, None, None, None, None, -1, -1]
    # remove sep token
    attn_all = attn_all[:, :-1, :-1]
    # sort attn_all along dim=1
    attn_all, _ = attn_all.sort(dim=1)
    # set min attention paid to min token across all positions
    attn_scores[:, 0, :, :, :, :, :num_min] = rearrange(attn_all[:, :num_min, :, None, None, None], 'h kpos ktokmin ktoknonmin1 ktoknonmin2 minpos -> h minpos ktokmin ktoknonmin1 ktoknonmin2 kpos')
    # set max attention paid to min token across all positions
    attn_scores[:, 1:, 0, :, :, :, :num_min] = rearrange(attn_all[:, -num_min:, :, None, None, None], 'h kpos ktokmin ktoknonmin1 ktoknonmin2 minpos -> h minpos ktokmin ktoknonmin1 ktoknonmin2 kpos')
    # set middle attention paid to min token across all positions
    attn_scores[:, 1, 1, :, :, :, :num_min] = rearrange(attn_all[:, num_nonmin1:num_nonmin1+num_min, :, None, None], 'h kpos ktokmin ktoknonmin1 ktoknonmin2 -> h ktokmin ktoknonmin1 ktoknonmin2 kpos')
    attn_scores[:, 2, 1, :, :, :, :num_min] = rearrange(attn_all[:, -(num_nonmin1+num_min):-num_nonmin1, :, None, None], 'h kpos ktokmin ktoknonmin1 ktoknonmin2 -> h ktokmin ktoknonmin1 ktoknonmin2 kpos')

    # set min attention paid to nonmin2 token across all positions
    attn_scores[:, -1, :, :, :, :, -(num_nonmin2+1):-1] = rearrange(attn_all[:, :num_nonmin2, :, None, None, None], 'h kpos ktoknonmin2 ktokmin ktoknonmin1 minpos -> h minpos ktokmin ktoknonmin1 ktoknonmin2 kpos')
    # set max attention paid to nonmin2 token across all positions
    attn_scores[:, :-1, -1, :, :, :, -(num_nonmin2+1):-1] = rearrange(attn_all[:, -num_nonmin2:, :, None, None, None], 'h kpos ktoknonmin2 ktokmin ktoknonmin1 minpos -> h minpos ktokmin ktoknonmin1 ktoknonmin2 kpos')
    # set middle attention paid to nonmin2 token across all positions
    attn_scores[:, 0, 0, :, :, :, -(num_nonmin2+1):-1] = rearrange(attn_all[:, num_min:num_min+num_nonmin2, :, None, None], 'h kpos ktoknonmin2 ktokmin ktoknonmin1 -> h ktokmin ktoknonmin1 ktoknonmin2 kpos')
    attn_scores[:, 1, 0, :, :, :, -(num_nonmin2+1):-1] = rearrange(attn_all[:, -(num_min+num_nonmin2):-num_min, :, None, None], 'h kpos ktoknonmin2 ktokmin ktoknonmin1 -> h ktokmin ktoknonmin1 ktoknonmin2 kpos')

    # set min attention paid to nonmin1 token across all positions
    attn_scores[:, -1, :, :, :, :, num_min:num_min+num_nonmin1] = rearrange(attn_all[:, :num_nonmin1, :, None, None, None], 'h kpos ktoknonmin1 ktokmin ktoknonmin2 minpos -> h minpos ktokmin ktoknonmin1 ktoknonmin2 kpos')
    # set max attention paid to nonmin1 token across all positions
    attn_scores[:, :-1, -1, :, :, :, num_min:num_min+num_nonmin1] = rearrange(attn_all[:, -num_nonmin1:, :, None, None, None], 'h kpos ktoknonmin1 ktokmin ktoknonmin2 minpos -> h minpos ktokmin ktoknonmin1 ktoknonmin2 kpos')
    # set middle attention paid to nonmin1 token across all positions
    attn_scores[:, 0, 1, :, :, :, num_min:num_min+num_nonmin1] = rearrange(attn_all[:, num_min:num_min+num_nonmin1, :, None, None], 'h kpos ktoknonmin1 ktokmin ktoknonmin2 -> h ktokmin ktoknonmin1 ktoknonmin2 kpos')
    attn_scores[:, -1, 0, :, :, :, num_min:num_min+num_nonmin1] = rearrange(attn_all[:, -(num_min+num_nonmin1):-num_min, :, None, None], 'h kpos ktoknonmin1 ktokmin ktoknonmin2 -> h ktokmin ktoknonmin1 ktoknonmin2 kpos')

    # compute softmax
    attn_pattern_expanded = attn_scores.softmax(dim=-1)

    # remove rows corresponding to impossible non-min tokens
    if nanify_impossible_nonmin:
        for mintok in range(d_vocab):
            attn_pattern_expanded[:, :, :, mintok, :mintok, :, :] = float('nan')
            attn_pattern_expanded[:, :, :, mintok, :, :mintok, :] = float('nan')

    attn_pattern = t.zeros((n_heads, 3, 2, d_vocab, d_vocab, d_vocab, 4), device=attn_all.device)
    attn_pattern[:, :, :, :, :, :, 0] = attn_pattern_expanded[:, :, :, :, :, :, :num_min].sum(dim=-1)
    attn_pattern[:, :, :, :, :, :, 1] = attn_pattern_expanded[:, :, :, :, :, :, num_min:num_min+num_nonmin1].sum(dim=-1)
    attn_pattern[:, :, :, :, :, :, 2] = attn_pattern_expanded[:, :, :, :, :, :, num_min+num_nonmin1:-1].sum(dim=-1)
    attn_pattern[:, :, :, :, :, :, 3] = attn_pattern_expanded[:, :, :, :, :, :, -1]

    # remove rows corresponding to impossible non-min tokens
    if nanify_impossible_nonmin:
        for mintok in range(d_vocab):
            attn_pattern_expanded[:, :, :, mintok, :mintok, :, :] = float('nan')
            attn_pattern_expanded[:, :, :, mintok, :, :mintok, :] = float('nan')

    for mintok in range(d_vocab):
        # for min = nonmin1, move all attention to min
        attn_pattern[:, :, :, mintok, mintok, :, 0] += attn_pattern[:, :, :, mintok, mintok, :, 1]
        attn_pattern[:, :, :, mintok, mintok, :, 1] = 0
        # for min = nonmin2, move all attention to min
        attn_pattern[:, :, :, mintok, :, mintok, 0] += attn_pattern[:, :, :, mintok, :, mintok, 2]
        attn_pattern[:, :, :, mintok, :, mintok, 2] = 0
        # for nonmin1 = nonmin2, move all attention to nonmin1
        attn_pattern[:, :, :, :, mintok, mintok, 1] += attn_pattern[:, :, :, :, mintok, mintok, 2]
        attn_pattern[:, :, :, :, mintok, mintok, 2] = 0

    return attn_pattern

@torch.no_grad()
def compute_3way_attention_patterns_all_counts(model, sep_pos=dataset.list_len, nanify_impossible_nonmin=True, max_num_min=dataset.list_len-2, max_num_nonmin1=dataset.list_len-2, attn_all=None, attn_all_outdim='h qpos kpos qtok ktok'):
    '''
    returns post-softmax attention paid to the minimum token, two nonminimum tokens, and to the sep token
    in shape (num_min, num_nonmin1, head, 3, 2, mintok, nonmintok1, nonmintok2, 4)

    The 3, 2 is for the permutations of attention ordering by position, min vs nonmin1 vs nonmin2 in the lowest attention positions, and then which of the remaining two is in the highest attention position.
    The 4 is for mintok, nonmintok1, nonmintok2, and sep
    '''
    if attn_all is None:
        return compute_3way_attention_patterns_all_counts(
            model, sep_pos=sep_pos, nanify_impossible_nonmin=nanify_impossible_nonmin, max_num_min=max_num_min, max_num_nonmin1=max_num_nonmin1,
            attn_all=compute_attn_all(model, outdim=attn_all_outdim, nanify_sep_loc=sep_pos), attn_all_outdim=attn_all_outdim)

    n_heads, _, _, d_vocab, _ = attn_all.shape
    d_vocab -= 1 # remove sep token
    default = t.zeros((n_heads, 3, 2, d_vocab, d_vocab, d_vocab, 4), device=attn_all.device)
    default[...] = float('nan')

    return torch.stack([torch.stack([
        compute_3way_attention_patterns(model, sep_pos=sep_pos, nanify_impossible_nonmin=nanify_impossible_nonmin, num_min=num_min, num_nonmin1=num_nonmin1, attn_all=attn_all, attn_all_outdim=attn_all_outdim)
        if num_min + num_nonmin1 + 1 <= sep_pos else default
        for num_nonmin1 in range(1, max_num_nonmin1+1)
    ], dim=0)
    for num_min in range(1, max_num_min+1)], dim=0) # tqdm

# %%
@torch.no_grad()
def compute_EPVOU(model, nanify_sep_position=dataset.list_len):
    EPV = model.blocks[0].ln1(model.W_pos[:, None, :] + model.W_E[None, :, :])[None, :, :, :] @ model.W_V[0,:,None,:,:] + model.b_V[0, :, None, None, :]
    # (head, pos, input, d_head)
    # b_O is not split amongst the heads, so we distribute it evenly amongst heads
    EPVO = EPV @ model.W_O[0,:,None,:,:] + model.b_O[0, None, None, None, :] / model.cfg.n_heads
    # (head, pos, input, d_model)
    EPVOU = layernorm_noscale(EPVO) @ model.W_U
    # (head, pos, input, output)
    EPVOU = EPVOU - EPVOU.mean(dim=-1, keepdim=True)
    if nanify_sep_position is not None:
        # SEP is the token in the SEP position
        EPVOU[:, nanify_sep_position, :-1, :] = float('nan')
        # SEP never occurs in positions other than the SEP position
        EPVOU[:, :nanify_sep_position, -1, :], EPVOU[:, nanify_sep_position+1:, -1, :] = float('nan'), float('nan')
    return EPVOU
# %%
@torch.no_grad()
def compute_EUPU(model, nanify_sep_position=dataset.list_len):
    EUPU = layernorm_noscale(model.W_pos[:, None, :] + model.W_E[None, :, :]) @ model.W_U + model.b_U[None, None, :]
    EUPU = EUPU - EUPU.mean(dim=-1, keepdim=True)
    if nanify_sep_position is not None:
        # SEP is the token in the SEP position
        EUPU[nanify_sep_position, :-1, :] = float('nan')
        # SEP never occurs in positions other than the SEP position
        EUPU[:nanify_sep_position, -1, :], EUPU[nanify_sep_position+1:, -1, :] = float('nan'), float('nan')
    return EUPU

# %%
@torch.no_grad()
def compute_EPVOU_EUPU(model, nanify_sep_position=dataset.list_len, qtok=-1, qpos=dataset.list_len):
    '''
    return indexed by (head, n_ctx_k, d_vocab_k, d_vocab_out)
    '''
    EPVOU = compute_EPVOU(model, nanify_sep_position=nanify_sep_position)
    # (head, pos, input, output)
    EUPU = compute_EUPU(model, nanify_sep_position=nanify_sep_position)
    # (pos, input, output)
    return EPVOU + EUPU[qpos, qtok, None, None, None, :] / EPVOU.shape[0]

# %%
@torch.no_grad()
def layernorm_noscale(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(axis=-1, keepdim=True)

@torch.no_grad()
def layernorm_scales(x: torch.Tensor, eps: float = 1e-5, recip: bool = True) -> torch.Tensor:
    x = layernorm_noscale(x)
    scale = (x.pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    if recip: scale = 1 / scale
    return scale

# %% [markdown]
# # Exploratory Plots
#
# Before diving into the proof, we provide some plots that may help with understanding the above claims.  These are purely exploratory (aimed at hypothesis generation) and are not required for hypothesis validation.

# %% [markdown]
# ## Initial Layernorm Scaling


# %%
s = layernorm_scales(model.W_pos[:,None,:] + model.W_E[None,:,:])[...,0]
# the only token in position 10 is SEP
s[dataset.list_len, :-1] = float('nan')
# SEP never occurs in positions other than 10
s[:dataset.list_len, -1:], s[dataset.list_len+1:, -1:] = float('nan'), float('nan')
# we don't actually care about the prediction in the last position
s = s[:-1, :]
smin = s[~s.isnan()].min()
# s = s / smin
px.imshow(utils.to_numpy(s), color_continuous_scale='Sunsetdark', labels={"x":"Token Value", "y":"Position"}, title=f"Layer Norm Scaling", x=dataset.vocab).show(None)

# %% [markdown]
# ## Attention Plots

# %%
# Attention
attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=dataset.list_len)
attn_subset = attn_all[:, dataset.list_len, :dataset.list_len+1, -1, :]
zmin, zmax = attn_subset[~attn_subset.isnan()].min().item(), attn_subset[~attn_subset.isnan()].max().item()

fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=("Head 0", "Head 1"))
fig.update_layout(title="Attention (pre-softmax) from SEP to other tokens and positions")
tickvals, ticktext = make_tickvals_text(step=10, include_lastn=0, skip_lastn=1, dataset=dataset)
for h in range(model.cfg.n_heads):
    fig.add_trace(go.Heatmap(z=utils.to_numpy(attn_subset[h]), colorscale='Plasma', zmin=zmin, zmax=zmax, hovertemplate="Token: %{x}<br>Position: %{y}<br>Attention: %{z}<extra>Head " + str(h) + "</extra>"), row=1, col=h+1)
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, title_text="Key Token", row=1, col=h+1)
    fig.update_yaxes(title_text="Position of Key", row=1, col=h+1)
fig.show()

# %%
# Attention Across Positions
attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=dataset.list_len)
zmax = attn_all[~attn_all.isnan()].max().item()
zmin = attn_all[~attn_all.isnan()].min().item()

default_attn = t.zeros_like(attn_all[0, 0, 0])
default_attn[:, :] = float('nan')
default_attn = utils.to_numpy(default_attn)

n_cols = 10
n_rows_per_head = (dataset.seq_len - 1 - 1) // n_cols + 1
n_rows = n_rows_per_head * model.cfg.n_heads

tickvals, ticktext = make_tickvals_text(step=20, include_lastn=0, skip_lastn=0, dataset=dataset)

subplot_titles = []
for h in range(model.cfg.n_heads):
    subplot_titles += [f"{h}:{kpos}" for kpos in range(dataset.seq_len - 1)]
    subplot_titles += ["" for _ in range(dataset.seq_len - 1, n_cols * n_rows_per_head)]
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
fig.update_annotations(font_size=5)
fig.update_layout(title="Attention head:key_position, x=key token, y=query token")

def make_update(qpos, h=None, kpos=None, showscale=False, include_tickvals_ticktext=False):
    if h is None or kpos is None:
        return [make_update(qpos, h, kpos, include_tickvals_ticktext=include_tickvals_ticktext) for h in range(model.cfg.n_heads) for kpos in range(n_cols * n_rows_per_head)]
    cur_attn = attn_all[h, qpos, kpos] if kpos <= qpos else default_attn
    x = dataset.vocab
    cur_tickvals, cur_ticktext = tickvals, ticktext
    if kpos == dataset.list_len:
        cur_attn = cur_attn[:, -1:]
        # nan_column = torch.full((cur_attn.shape[0], 1), float('nan'), device=cur_attn.device, dtype=cur_attn.dtype)
        # cur_attn = torch.cat([nan_column, cur_attn, nan_column], dim=1)
        # x, cur_tickvals, cur_ticktext = [float('nan'), x[-1], float('nan')], [float('nan'), cur_tickvals[-1], float('nan')], ['', cur_ticktext[-1], '']
        x, cur_tickvals, cur_ticktext = x[-1:], cur_tickvals[-1:], cur_ticktext[-1:]
    trace = go.Heatmap(z=utils.to_numpy(cur_attn), colorscale='Plasma', zmin=zmin, zmax=zmax, showscale=showscale, x=x, y=dataset.vocab, hovertemplate="Key: %{x}<br>Query: %{y}<br>Attention: %{z}<extra>" + f"Head {h}<br>Key Pos {kpos}<br>Query Pos {qpos}" + "</extra>")
    if include_tickvals_ticktext: return trace, cur_tickvals, cur_ticktext
    return trace

def update(qpos):
    fig.data = []
    for i, (trace, cur_tickvals, cur_ticktext) in enumerate(make_update(qpos, include_tickvals_ticktext=True)):
        row, col = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(tickvals=cur_tickvals, ticktext=cur_ticktext, constrain='domain', row=row, col=col, tickfont=dict(size=5), title_font=dict(size=5))
        fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, row=row, col=col, tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=5), title_font=dict(size=5))

# Create the initial heatmap
update(dataset.seq_len-2)

# Create frames for each position
frames = [go.Frame(
    data=make_update(qpos),
    name=str(qpos)
) for qpos in range(dataset.list_len+1, dataset.seq_len - 1)]

fig.frames = frames

# # Add animation controls
# animation_settings = dict(
#     frame=dict(duration=1000, redraw=True),
#     fromcurrent=True,
#     transition=dict(duration=0)
# )

# Create slider
sliders = [dict(
    active=len(fig.frames) - 1,
    yanchor='top',
    xanchor='left',
    currentvalue=dict(font=dict(size=20), prefix='Query Position:', visible=True, xanchor='right'),
    transition=dict(duration=0),
    pad=dict(b=10, t=50),
    len=0.9,
    x=0.1,
    y=0,
    steps=[dict(args=[[frame.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                method='animate',
                label=frame.name) for frame in fig.frames]
)]

fig.update_layout(
    sliders=sliders
)

# fig.update_layout(
#     updatemenus=[dict(
#         type='buttons',
#         showactive=False,
#         buttons=[dict(label='Play',
#                       method='animate',
#                       args=[None, animation_settings])]
#     )]
# )

fig.show()

# %% [markdown]
# ## OV Attention Head Plots

# %%
EPVOU = compute_EPVOU(model, nanify_sep_position=dataset.list_len)
zmax = EPVOU[~EPVOU.isnan()].abs().max().item()
EPVOU = utils.to_numpy(EPVOU)

fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=[f"head {h}" for h in range(model.cfg.n_heads)])
# fig.update_annotations(font_size=12)
fig.update_layout(title="OV Logit Impact: x=Ouput Logit Token, y=Input Token<br>LN_noscale(LN1(W_pos[pos,:] + W_E) @ W_V[0,h] @ W_O[0, h]) @ W_U")

tickvals, ticktext = make_tickvals_text(step=10, include_lastn=0, skip_lastn=1, dataset=dataset)

def make_update(pos, h, adjust_sep=True):
    cur_EPVOU = EPVOU[h, pos]
    y = dataset.vocab
    if adjust_sep and pos == dataset.list_len:
        cur_EPVOU = cur_EPVOU[-1:, :]
        y = y[-1:]
    elif pos != dataset.list_len:
        cur_EPVOU = cur_EPVOU[:-1, :]
        y = y[:-1]
    return go.Heatmap(z=utils.to_numpy(cur_EPVOU), colorscale='Picnic_r', x=dataset.vocab, y=y, zmin=-zmax, zmax=zmax, showscale=(h == 0), hovertemplate="Input Token: %{y}<br>Output Token: %{x}<br>Logit: %{z}<extra>" + f"Head {h}<br>Pos {pos}" + "</extra>")

def update(pos):
    fig.data = []
    for h in range(model.cfg.n_heads):
        fig.add_trace(make_update(pos, h), row=1, col=h+1)
        fig.update_xaxes(constrain='domain', row=1, col=h+1) #, title_text="Output Logit Token"
        if pos == dataset.list_len:
            fig.update_yaxes(range=[-1,1], row=1, col=h+1)
        else:
            fig.update_yaxes(autorange='reversed', row=1, col=h+1)

# Create the initial heatmap
update(0)

# Create frames for each position
frames = [go.Frame(
    data=[make_update(pos, h, adjust_sep=True) for h in range(model.cfg.n_heads)],
    name=str(pos),
    layout={'yaxis': {'range': ([-1, 1] if pos == dataset.list_len else [len(dataset.vocab)-2, 0])},
        'yaxis2': {'range': ([-1, 1] if pos == dataset.list_len else [len(dataset.vocab)-2, 0])}},
) for pos in range(dataset.seq_len-1)]

fig.frames = frames

# # Add animation controls
# animation_settings = dict(
#     frame=dict(duration=1000, redraw=True),
#     fromcurrent=True,
#     transition=dict(duration=0)
# )

# Create slider
sliders = [dict(
    active=0,
    yanchor='top',
    xanchor='left',
    currentvalue=dict(font=dict(size=20), prefix='Position:', visible=True, xanchor='right'),
    transition=dict(duration=0),
    pad=dict(b=10, t=50),
    len=0.9,
    x=0.1,
    y=0,
    steps=[dict(args=[[frame.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                method='animate',
                label=str(pos)) for pos, frame in enumerate(fig.frames)]
)]

fig.update_layout(
    sliders=sliders
)

# fig.update_layout(
#     updatemenus=[dict(
#         type='buttons',
#         showactive=False,
#         buttons=[dict(label='Play',
#                       method='animate',
#                       args=[None, animation_settings])]
#     )]
# )

fig.show()

# %% [markdown]
# ## Skip Connection / Residual Stream Plots

# %%
n_rows = 2
n_cols = 1 + (dataset.list_len - 1) // n_rows
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"pos={pos}" for pos in range(dataset.list_len, dataset.seq_len - 1)])
fig.update_layout(title="Logit Impact from the embedding (without layernorm scaling)<br>LN_noscale(W_pos[pos,:] + W_E) @ W_U, y=Input, x=Ouput Logit Token")

EUPU = compute_EUPU(model, nanify_sep_position=dataset.list_len)
zmax = EUPU[~EUPU.isnan()].abs().max().item()
EUPU = utils.to_numpy(EUPU)
for i, pos in enumerate(range(dataset.list_len, dataset.seq_len-1)):
    r, c = i // n_cols, i % n_cols
    cur_EUPU = EUPU[pos]
    y = dataset.vocab
    if pos == dataset.list_len:
        cur_EUPU = cur_EUPU[-1:, :]
        y = y[-1:]
    else:
        cur_EUPU = cur_EUPU[:-1, :]
        y = y[:-1]
    fig.add_trace(go.Heatmap(z=utils.to_numpy(cur_EUPU), x=dataset.vocab, y=y, colorscale='Picnic_r', zmin=-zmax, zmax=zmax, showscale=(i==0), hovertemplate="Input Token: %{y}<br>Output Token: %{x}<br>Logit: %{z}<extra>" + f"Pos {pos}" + "</extra>"), row=r+1, col=c+1)
    fig.update_xaxes(constrain='domain', row=r+1, col=c+1) #, title_text="Output Logit Token"
    if pos == dataset.list_len:
        fig.update_yaxes(range=[-1,1], row=r+1, col=c+1)
    else:
        fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, row=r+1, col=c+1)
fig.show()

# %% [markdown]
# # Finding the Minimum with query SEP in Position 10

# %% [markdown]
# ## State Space Reduction
#
# **Lemma**: For a single attention head, it suffices to consider sequences with at most two distinct tokens.
#
# Note that we are comparing sequences by pre-final-layernorm-scaling gap between the logit of the minimum token and the logit of any other fixed token.
# Layernorm scaling is non-linear, but if we only care about accuracy and not log-loss, then we can ignore it (neither scaling nor softmax changes which logit is the largest).
#
# **Proof sketch**:
# We show that any sequence with three token values, $x < y < z$, is strictly dominated either by a sequence with just $x$ and $y$ or a sequence with just $x$ and $z$.
#
# Suppose we have $k$ copies of $x$, $n$ copies of $y$, and $\ell - k - n$ copies of $z$, the attention scores are $s_x$, $s_y$, and $s_z$, and the differences between the logit of $x$ and our chosen comparison logit (as computed by the OV circuit for each token) are $v_x$, $v_y$, and $v_z$.
# Then the difference in logit between $x$ and the comparison token is
# $$\left(k e^{s_x} v_x + n e^{s_y} v_y + (\ell - k - n)e^{s_z}v_z \right)\left(k e^{s_x} + n e^{s_y} + (\ell - k - n)e^{s_z}\right)^{-1}$$
# Rearrangement gives
# $$\left(\left(k e^{s_x} v_x + (\ell - k) e^{s_z} v_z\right) + n \left(e^{s_y} v_y - e^{s_z}v_z\right) \right)\left(\left(k e^{s_x} + (\ell - k) e^{s_z}\right) + n \left(e^{s_y} - e^{s_z}\right)\right)^{-1}$$
# This is a fraction of the form $\frac{a + bn}{c + dn}$.  Taking the derivative with respect to $n$ gives $\frac{bc - ad}{(c + dn)^2}$.  Noting that $c + dn$ cannot equal zero for any valid $n$, we get the the derivative never changes sign.  Hence our logit difference is maximized either at $n = 0$ or at $n = \ell - k$, and the sequence with just two values dominates the one with three.
#
# This proof generalizes straightforwardly to sequences with more than three values.
#
# Similarly, this proof shows that, when considering only a single attention head, it suffices to consider sequences of $\ell$ copies of the minimum token and sequences with one copy of the minimum token and $\ell - 1$ copies of the non-minimum token, as intermediate values are dominated by the extremes.
#

# %% [markdown]
# Let's bound how much attention is paid to the minimum token, non-minimum tokens (in aggregate), and the SEP token.
#
# First a plot.  We use green for "paying attention to the minimum token", red for "paying attention to the non-minimum token", and blue for "paying attention to the SEP token".

# %%
# swap the axes so that we have red for nonmin, green for min, and blue for sep
attn_patterns = torch.stack([compute_attention_patterns(model, num_min=num_min)[:, :, :, :, (1, 0, 2)] for num_min in range(1, dataset.list_len)], dim=0)
# (num_min, head, 2, mintok, nonmintok, 3)
attn_patterns[attn_patterns.isnan()] = 1
attn_patterns = utils.to_numpy(attn_patterns * 256)

fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=[f"head {h}" for h in range(model.cfg.n_heads)])# {minmax} attn on mintok" for h in range(model.cfg.n_heads) for minmax in ('min', 'max')])
# fig.update_annotations(font_size=12)
minmaxi_g = 0 # min attention, but it doesn't matter much
fig.update_layout(title=f"Attention ({('min', 'max')[minmaxi_g]} on min tok)")

all_tickvals_text = list(enumerate(dataset.vocab[:-1]))
tickvals_indices = list(range(0, len(all_tickvals_text) - 1, 10)) + [len(all_tickvals_text) - 1]
tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]


def make_update(h, minmaxi, num_min):
    cur_attn_pattern = attn_patterns[num_min - 1, h, minmaxi]
    # cur_hovertext = all_hovertext[num_min - 1][h][minmaxi]
    return go.Image(z=cur_attn_pattern, customdata=cur_attn_pattern / 256 * 100, hovertemplate="Non-min token: %{x}<br>Min token: %{y}<br>Min token attn: %{customdata[1]:.1f}%<br>Nonmin tok attn: %{customdata[0]:.1f}%<br>SEP attn: %{customdata[0]:.1f}%<extra>" + f"head {h}" + "</extra>")

def update(num_min):
    fig.data = []
    for h in range(model.cfg.n_heads):
        for minmaxi, minmax in list(enumerate(('min', 'max')))[:1]:
            col, row = h+1, minmaxi+1
            fig.add_trace(make_update(h, minmaxi, num_min), col=col, row=row)
            fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, constrain='domain', col=col, row=row, title_text="non-min tok") #, title_text="Output Logit Token"
            fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, col=col, row=row, title_text="min tok")

# Create the initial heatmap
update(1)

# Create frames for each position
frames = [go.Frame(
    data=[make_update(h, minmaxi, num_min) for h in range(model.cfg.n_heads) for minmaxi in (minmaxi_g, )],
    name=str(num_min)
) for num_min in range(1, dataset.list_len)]

fig.frames = frames

# Create slider
sliders = [dict(
    active=0,
    yanchor='top',
    xanchor='left',
    currentvalue=dict(font=dict(size=20), prefix='# copies of min token:', visible=True, xanchor='right'),
    transition=dict(duration=0),
    pad=dict(b=10, t=50),
    len=0.9,
    x=0.1,
    y=0,
    steps=[dict(args=[[frame.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                method='animate',
                label=str(num_min+1)) for num_min, frame in enumerate(fig.frames)]
)]

fig.update_layout(
    sliders=sliders
)


fig.show()

# %% [markdown]
#
# There are some remarkable things about this plot.
#
# 1. For sequences where the minimum token is 19 or larger, the model should basically never get the first token correct, because it's paying too much attention to either the SEP token or the wrong non-min token.
# 2. Even for sequences with 9 or 10 copies of the same number, if that number is 25--39, so much attention is paid to the SEP token (by both heads) that we predict that the model probably gets the wrong answer, even before analyzing OV.
# 3. The plot probably overestimates the incorrect behavior when the non-min token is relatively close to the min token, because the OV matrices do (small) positive copying of numbers below the current one.  We don't analyze this behavior in enough depth here to put hard bounds on when it's enough to compensate for paying attention to the wrong token, but a more thorough analysis would.
#
# Before using the above distributions to place concrete bounds on what fraction of outputs the network gets correct, let's compute cutoffs for the OV behavior.
#
# But first, is this actually right?  Which uniform sequences does the model get wrong?  What fraction of sequences starting at 19 get the wrong minimum?

# %%
uniform_predictions = [model(t.tensor([i] * dataset.list_len + [len(dataset.vocab) - 1] + [i] * dataset.list_len))[0, dataset.list_len].argmax(dim=-1).item() for i in range(len(dataset.vocab) - 1)]
wrong_uniform_predictions = [(i, p) for i, p in enumerate(uniform_predictions) if p != i]
print(f"The model incorrectly predicts the minimum for {len(wrong_uniform_predictions)} sequences ({', '.join(str(i) for i, p in wrong_uniform_predictions)}):\n{' '.join([f'{i} (model: {p})' for i, p in wrong_uniform_predictions])}")

# %% [markdown]
# Interestingly, the model manages to get 35, 36, 37 right, despite paying most attention to SEP.

# %%
n_total_datapoints = 10000
datapoints_per_batch = 1000
# Set a random seed, then generate n_datapoints sequences of length dataset.list_len of numbers between 19 and len(dataset.vocab) - 2, inclusive
# Set random seed for reproducibility
torch.manual_seed(42)
all_predictions = t.zeros((0,), dtype=torch.long)
all_minima = t.zeros((0,), dtype=torch.long)
low = 19
# true minimum, predicted minimum, # copies of minimum
results = t.zeros((len(dataset.vocab) - 1, len(dataset.vocab) - 1, dataset.list_len + 1))
with torch.no_grad():
    for _ in tqdm(range(n_total_datapoints // datapoints_per_batch)):
        for real_low in range(low, len(dataset.vocab) - 1):
            sequences = torch.randint(real_low, len(dataset.vocab) - 1, (datapoints_per_batch, dataset.list_len))
            sorted_sequences = sequences.sort(dim=-1).values
            minima = sorted_sequences[:, 0]
            n_copies = (sequences == minima.unsqueeze(-1)).sum(dim=-1)
            sequences = torch.cat([sequences, torch.full((datapoints_per_batch, 1), len(dataset.vocab) - 1, dtype=torch.long), sorted_sequences], dim=-1)
            # Compute the model's predictions for the minimum
            predictions = model(sequences)[:, dataset.list_len, :].argmax(dim=-1)
            # Compute the actual minimums
            all_predictions = torch.cat([all_predictions, predictions.cpu()])
            all_minima = torch.cat([all_minima, minima.cpu()])
            # count the number of copies of the minimum
            results[minima.cpu(), predictions.cpu(), n_copies.cpu()] += 1
            # if real_low == 45:
            #     good_sequences = sequences
            #     print(set(sequences.cpu()[(minima.cpu() == 45) & (predictions.cpu() == minima.cpu())]))
    # Compute the fraction of correct predictions
    correct = (predictions.cpu() == minima).float().mean().item()
# (true minimum, # copies of minimum)
fraction_correct = (results / results.sum(dim=1, keepdim=True)).diagonal(dim1=0, dim2=1)
# print(f"The model correctly predicts the minimum for {correct * 100}% of sequences of length {dataset.list_len} with numbers between {low} and {len(dataset.vocab) - 2} inclusive")
# # plot predictions - minima against minima
# scatter(all_minima, all_predictions - all_minima, title="Predicted - Actual Minimum vs Actual Minimum", xaxis="Actual Minimum", yaxis="Predicted - Actual Minimum")
# scatter(all_minima, all_predictions, title="Actual Minimum vs Predicted Minimum", xaxis="Actual Minimum", yaxis="Predicted - Actual Minimum")
# scatter(list(range(low, len(dataset.vocab) - 1)), [(all_predictions[all_minima == i] == i).float().mean().item() for i in range(low, len(dataset.vocab) - 1)], title="Fraction Correct vs Actual Minimum", xaxis="Actual Minimum", yaxis="Fraction Correct")
imshow(fraction_correct, title="Fraction Correct vs Actual Minimum and # Copies of Minimum", xaxis="Actual Minimum", yaxis="# Copies of Minimum")
# print(sequences)

# %% [markdown]
# So we see that with enough copies of the minimum, we can get things correct, but with only one or two copies, we tend not to.
#
# This makes sense, though.  Consider what fraction of sequences start at 19 or higher.

# %%
total_sequences = (len(dataset.vocab) - 1) ** dataset.list_len
# (minimum, n copies of minimum)
count_of_sequences = torch.zeros((len(dataset.vocab) - 1, dataset.list_len + 1), dtype=torch.long)
count_of_sequences[:, dataset.list_len] = 1 # one sequence for each min tok when everything is the same
for mintok in range(len(dataset.vocab) - 1):
    for n_copies in range(1, dataset.list_len):
        count_of_sequences[mintok, n_copies] = math.comb(dataset.list_len, n_copies) * (len(dataset.vocab) - 1 - mintok - 1)**(dataset.list_len - n_copies)
assert count_of_sequences.sum().item() == total_sequences, f"{np.abs(count_of_sequences.sum().item() - total_sequences)} != 0"
fraction_of_sequences = count_of_sequences.float() / total_sequences
imshow(fraction_of_sequences, title="Fraction of Sequences vs Actual Minimum and # Copies of Minimum", yaxis="Actual Minimum", xaxis="# Copies of Minimum")
fraction_of_sequences_all_counts = fraction_of_sequences.sum(dim=-1)
cumulative_fraction_of_sequences_all_counts = fraction_of_sequences_all_counts.cumsum(dim=0)
significant = cumulative_fraction_of_sequences_all_counts[cumulative_fraction_of_sequences_all_counts <= 0.99]
print(f"{cumulative_fraction_of_sequences_all_counts[significant.shape[0]+1] * 100:.1f}% of sequences start at or below {significant.shape[0] + 1}")

# %% [markdown]
#
# So since more than 99% of cases have their minimum at or below 19, it seem fine to only explain the behavior on sequences with minimum at or below 19.


# %% [markdown]
# ## OV Cutoffs
#
# For each pair of minimum and non-minimum tokens, we can ask: how much attention needs to be paid to the minimum token to ensure that the correct output logit is highest?
# We ask this question separately for when the remainder of the attention is paid to the non-min token vs paid to the SEP token.
# Note that to make use of the proof above about considering only sequences with at most two distinct tokens, we need to consider the behavior of the two heads independently.

# %% [markdown]
# Let's first find the worst-case OV behavior for head 0, indexed by (minimum token, output logit)
# %%
@torch.no_grad()
def compute_worst_OV(model, sep_pos=dataset.list_len, num_min=1):
    '''
    returns (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)
    '''
    EPVOU_EUPU = compute_EPVOU_EUPU(model, nanify_sep_position=sep_pos, qtok=-1, qpos=sep_pos)
    # (head, pos, input, output)
    EPVOU_EUPU_sep = EPVOU_EUPU[:, sep_pos, -1, :]
    EPVOU_EUPU = EPVOU_EUPU[:, :sep_pos, :-1, :]
    attn_patterns = compute_attention_patterns(model, sep_pos=sep_pos, num_min=num_min)
    # (head, 2, mintok, nonmintok, 3)
    # 2 is for max attn on min vs nonmin, 3 is for min, nonmin, sep
    results = t.zeros(tuple(list(attn_patterns.shape[:-1]) + [model.cfg.d_vocab_out]), device=attn_patterns.device)
    for mintok in range(model.cfg.d_vocab - 1):
        # center on the logit for the minimum token
        cur_EPVOU_EUPU = EPVOU_EUPU - EPVOU_EUPU[:, :, :, mintok].unsqueeze(dim=-1)
        # (head, pos, input, output)
        # reduce along position, since they're all nearly identical
        cur_EPVOU_EUPU = cur_EPVOU_EUPU.max(dim=1).values
        # (head, input, output)

        cur_EPVOU_EUPU_sep = EPVOU_EUPU_sep - EPVOU_EUPU_sep[:, mintok].unsqueeze(dim=-1)
        # (head, output)

        cur_attn_patterns = attn_patterns[:, :, mintok, :, :]
        # (head, 2, nonmintok, 3)

        results[:, :, mintok, :, :] = \
            einsum(cur_attn_patterns[..., 0], cur_EPVOU_EUPU[:, mintok, :],
                    "head minmax nonmintok, head output -> head minmax nonmintok output") \
            + einsum(cur_attn_patterns[..., 1], cur_EPVOU_EUPU,
                    "head minmax nonmintok, head nonmintok output -> head minmax nonmintok output") \
            + einsum(cur_attn_patterns[..., 2], cur_EPVOU_EUPU_sep,
                    "head minmax nonmintok, head output -> head minmax nonmintok output")
        # re-center on output logit for minimum token
        results[:, :, mintok, :, :] -= results[:, :, mintok, :, mintok].unsqueeze(dim=-1)

    return results.max(dim=1).values

@torch.no_grad()
def reduce_worst_OV(worst_OV):
    '''
    returns (n_heads, d_vocab_min)
    '''
    worst_OV = worst_OV.clone()
    results = torch.zeros_like(worst_OV[:, :, 0, 0])
    assert len(results.shape) == 2
    for mintok in range(worst_OV.shape[1]):
        # set diagonal to min to avoid including the logit of the min in the worst non-min logit
        worst_OV[:, mintok, :, mintok] = worst_OV[:, mintok, :, :].min(dim=-1).values
        # reduce for worst logit
        results[:, mintok] = worst_OV[:, mintok, mintok:, :].max(dim=-1).values.max(dim=-1).values
    return results

@torch.no_grad()
def compute_worst_OV_reduced(model, **kwargs):
    '''
    returns (n_heads, d_vocab_min)
    '''
    return reduce_worst_OV(compute_worst_OV(model, **kwargs))


# %%
line(compute_worst_OV_reduced(model).T, title="Worst OV behavior per head, Value vs Min Token")

# %% [markdown]
#
# Now let's compute, for each minimum and non-minimum token, whether or not the attention from head 1 is enough to overcome the worst behavior from head 0

# %%
@torch.no_grad()
def compute_slack(model, good_head_num_min=1, **kwargs):
    '''
    returns (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)
    '''
    # we consider the worst behavior of the bad heads with only one copy of the minimum (except for the diagonal, which has the maximum number of copies)
    worst_OV = compute_worst_OV(model, num_min=1, **kwargs)
    # for the good head, we pass in the number of copies of the minimum token
    worst_OV_good = compute_worst_OV(model, num_min=good_head_num_min, **kwargs)
    # (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)
    results = worst_OV_good.clone()
    for good_head in range(worst_OV.shape[0]):
        # reduce along nonmin tokens in other heads and add them in
        other_worst_OV = torch.cat((worst_OV[:good_head], worst_OV[good_head+1:]), dim=0)
        # (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)
        for mintok in range(worst_OV.shape[1]):
            results[good_head, mintok] += \
                reduce(other_worst_OV[:, mintok, mintok:, :],
                       "head nonmintok output -> head () output", reduction="max").sum(dim=0)
    return results

@torch.no_grad()
def compute_slack_reduced(model, good_head_num_min=1, **kwargs):
    '''
    returns (n_heads, d_vocab_min, d_vocab_nonmin)
    '''
    slack = compute_slack(model, good_head_num_min=good_head_num_min, **kwargs)
    # (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)
    for mintok in range(slack.shape[1]):
        # assert centered
        test_slack = slack[:, mintok, :, mintok]
        test_slack = test_slack[~test_slack.isnan()]
        assert (test_slack == 0).all(), test_slack.max().item()
        # set diagonal to min to avoid including the logit of the min in the worst non-min logit
        slack[:, mintok, :, mintok] = slack[:, mintok, :, :].min(dim=-1).values
    return slack.max(dim=-1).values

# %%
slacks = torch.stack([compute_slack_reduced(model, good_head_num_min=num_min) for num_min in range(1, dataset.list_len)], dim=0)
# (num_min, head, mintok, nonmintok)
zmax = slacks[~slacks.isnan()].abs().max().item()
# negate for coloring
slacks_sign = slacks.sign()
slacks_full = -utils.to_numpy(slacks)
slacks_sign = -utils.to_numpy(slacks_sign)

for slacks in (slacks_full, slacks_sign):
    fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=[f"slack on head {h}" for h in range(model.cfg.n_heads)])# {minmax} attn on mintok" for h in range(model.cfg.n_heads) for minmax in ('min', 'max')])
    fig.update_layout(title=f"Slack (positive for either head ⇒ model is correct)")

    all_tickvals_text = list(enumerate(dataset.vocab[:-1]))
    tickvals_indices = list(range(0, len(all_tickvals_text) - 1, 10)) + [len(all_tickvals_text) - 1]
    tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
    tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]


    def make_update(h, num_min, showscale=True):
        cur_slack = slacks[num_min - 1, h]
        return go.Heatmap(z=cur_slack,  colorscale='Picnic_r', zmin=-zmax, zmax=zmax, showscale=showscale)

    def update(num_min):
        fig.data = []
        for h in range(model.cfg.n_heads):
            col, row = h+1, 1
            fig.add_trace(make_update(h, num_min), col=col, row=row)
            fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, constrain='domain', col=col, row=row, title_text="non-min tok") #, title_text="Output Logit Token"
            fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, col=col, row=row, title_text="min tok")
        fig.update_traces(hovertemplate="Non-min token: %{x}<br>Min token: %{y}<br>Slack: %{z}<extra>head %{fullData.name}</extra>")

    # Create the initial heatmap
    update(1)

    # Create frames for each position
    frames = [go.Frame(
        data=[make_update(h, num_min) for h in range(model.cfg.n_heads)],
        name=str(num_min)
    ) for num_min in range(1, dataset.list_len)]

    fig.frames = frames

    # Create slider
    sliders = [dict(
        active=0,
        yanchor='top',
        xanchor='left',
        currentvalue=dict(font=dict(size=20), prefix='# copies of min token:', visible=True, xanchor='right'),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=[dict(args=[[frame.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                    method='animate',
                    label=str(num_min+1)) for num_min, frame in enumerate(fig.frames)]
    )]

    fig.update_layout(
        sliders=sliders
    )


    fig.show()

# %% [markdown]
# Let's count what fraction of sequences we can now explain the computation of the minimum on.

# %%
@torch.no_grad()
def calculate_correct_minimum_lower_bound(model, sep_pos=dataset.list_len):
    count = 0
    for min_token_copies in range(1, sep_pos):
        slack = compute_slack_reduced(model, good_head_num_min=min_token_copies, sep_pos=sep_pos)
        # (n_heads, d_vocab_min, d_vocab_nonmin)
        # find where we have slack
        slack = (slack < 0)
        for mintok in range(slack.shape[1]):
            # count the number of sequences with the specified number of copies of mintok and with other tokens drawn from the values where we have slack
            # we reduce along heads to find out which head permits us the most values with slack;
            # since the proof of convexity only works when we fix which head we're analyzing, we can't union permissible values across heads
            n_allowed_values = slack[:, mintok, mintok+1:].sum(dim=-1).max().item()
            if n_allowed_values > 0:
                count += math.comb(sep_pos, min_token_copies) * n_allowed_values ** (sep_pos - min_token_copies)
        if min_token_copies == sep_pos - 1:
            # add in the diagonal on this last round
            count += slack.any(dim=0).diagonal(dim1=-2, dim2=-1).sum().item()

    return count

@torch.no_grad()
def calculate_correct_minimum_lower_bound_fraction(model, sep_pos=dataset.list_len, **kwargs):
    total_sequences = (model.cfg.d_vocab - 1) ** sep_pos
    return calculate_correct_minimum_lower_bound(model, sep_pos=sep_pos, **kwargs) / total_sequences

# %%
print(f"Assuming no errors in our argument, we can prove that the model computes the correct minimum of the sequence in at least {calculate_correct_minimum_lower_bound(model)} cases ({100 * calculate_correct_minimum_lower_bound_fraction(model):.1f}% of the sequences).")

# %% [markdown]
# ## Summary of Results for First Sequence Token Predictions
#
# We've managed to prove a lower bound on correctness of the first token at 32%.  While this isn't great (the model in fact seems to do much better than this), this is only a preliminary analysis of the behavior.
#
# Recaping: We found that head 1 does positive copying on numbers outside of 25--39.  For numbers below 20, head 1 frequently manages to pay most attention to the smallest number.  Since fewer than 1% of sequences have a minimum above 19, we largely neglect the behavior on sequences with large minima.  We compute for each head the largest logit gap between the actual minima and any other logit, separately for each number of copies of the minimum token.  We then compute the worst-case behavior of the other heads.  We pessimize over position, which mostly does not matter.  We folded the computation of skip connection into the computation of the attention head.  We made a convexity argument that, as long as we consider each head's behavior independently, we can restrict our attention to sequences with at most two distinct tokens and still bound the behavior of other sequences.
#
# Although we don't do a more in-depth analysis of the prediction of the first token for the October challenge, we (Jason Gross and Rajashree Agrwal, and Thomas Kwa) are working on a project involving more deeply anlyzing the behavior of even smaller models (1 attention head, no layer norm, only computing the maximum of a list) in more detail, with proofs formalized in the proof assistant Coq.  We're in the process of writing up preliminary results, including connections with heuristic arguments, and hope to post on LessWrong and/or the Alignment Forum soon.  Keep an eye out!

# %% [markdown]
# ## More fine-grained analysis of first token prediction
#
# We currently are making very loose approximations when the model outputs the wrong minimum on $n$ copies of the minimum and $10 - n$ copies of some non-minimum token; we throw away all sequences that contain any copies of that token.  For example, the model outputs the wrong minimum for `[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`:

# %%
print(f"The model predicts {model(t.tensor([0] + [1] * (dataset.list_len - 1) + [len(dataset.vocab) - 1] + [0] + [1] * (dataset.list_len - 1))).argmax(dim=-1)[0, dataset.list_len].item()} instead of 0.")

# %% [markdown]
# But then we throw away all sequences that contain a 0, any number of 1s, and any other non-zero numbers!  But the model predicts the minimum correctly for many of these!

# %%
n_samples = 10000
sequences = torch.randint(2, len(dataset.vocab) - 2, (n_samples, dataset.list_len))
zero_idxs = torch.randint(0, dataset.list_len - 1, (n_samples,))
one_idxs = (zero_idxs + torch.randint(1, dataset.list_len - 1, (n_samples,))) % dataset.list_len
sequences[torch.arange(n_samples), zero_idxs], sequences[torch.arange(n_samples), one_idxs] = 0, 1
sorted_sequences, _ = sequences.sort(dim=-1)
# add sep
sequences = torch.cat([sequences, torch.full((n_samples, 1), len(dataset.vocab) - 1, dtype=torch.long), sorted_sequences], dim=-1)
# compute predictions
predictions = model(sequences)[:, dataset.list_len, :].argmax(dim=-1)
print(f"The model gets a {(predictions == 0).float().mean().item() * 100}% correct predictions of the minimum from a random sample of {n_samples} sequences with one 0, one 1, and the remainder of numbers above 2.")

# %% [markdown]
# So we can do much better.
#
# %% [markdown]
#
# Let's compute an approximation of the best result we could get with this independence relaxation (head 0 and head 1 independent except on the minimum token).

# %%
@torch.no_grad()
def compute_slack_for_sequences(model, sequences: t.Tensor):
    '''
    returns (batch, n_heads)
    '''
    # append sep to sequences
    batch, sep_pos = sequences.shape
    sequences = torch.cat([sequences, torch.full((sequences.shape[0], 1), len(dataset.vocab) - 1, dtype=torch.long, device=sequences.device)], dim=-1)
    attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=dataset.list_len)
    attns = attn_all[:, sep_pos, torch.arange(sequences.shape[-1])[None, :], -1, sequences]
    # (head, batch, pos)
    attns = attns.softmax(dim=-1)
    EPVOU_EUPU = compute_EPVOU_EUPU(model, nanify_sep_position=sep_pos, qtok=-1, qpos=sep_pos)
    # (head, pos, input, output)
    OVs = EPVOU_EUPU[:, torch.arange(sequences.shape[-1])[None, :], sequences, :]
    # (head, batch, pos, output)
    result = einsum(attns, OVs, "head batch pos, head batch pos output -> batch head output")

    # now we pessimize over the other heads
    # find the minima
    minima = sequences.min(dim=-1).values
    # count the number of copies of the minimum
    n_copies = (sequences == minima.unsqueeze(-1)).sum(dim=-1)

    # center logits
    result -= result[torch.arange(batch), :, minima].unsqueeze(dim=-1)

    all_worst_OVs = []
    for num_min in range(1, sep_pos):
        cur_worst_OV = compute_worst_OV(model, sep_pos=sep_pos, num_min=num_min)
        cur_worst_OV_reduced = torch.zeros_like(cur_worst_OV[:, :, 0, :])
        for mintok in range(cur_worst_OV.shape[1]):
            cur_cur_worst_OV = cur_worst_OV[:, mintok, mintok:, :]
            cur_cur_worst_OV -= cur_cur_worst_OV[:, :, mintok].unsqueeze(dim=-1)
            cur_worst_OV_reduced[:, mintok, :] = cur_cur_worst_OV.max(dim=-2).values
        all_worst_OVs.append(cur_worst_OV_reduced)
        if num_min == sep_pos - 1:
            for mintok in range(cur_worst_OV.shape[1]):
                cur_worst_OV_reduced[:, mintok, :] = cur_worst_OV[:, mintok, mintok, :]
                cur_worst_OV_reduced[:, mintok, :] -= cur_worst_OV_reduced[:, mintok, mintok].unsqueeze(dim=-1)
            all_worst_OVs.append(cur_worst_OV_reduced)
    all_worst_OVs = torch.stack(all_worst_OVs, dim=0)
    # (num_min, head, mintok, output)
    all_worst_OVs = all_worst_OVs[n_copies, :, minima, :]
    # (batch, head, output)

    # assert centered
    test = result[torch.arange(batch), :, minima]
    assert (test == 0).all(), f"1: {test.max().item()}"
    test = all_worst_OVs[torch.arange(batch), :, minima]
    assert (test == 0).all(), f"2: {test.max().item()}"

    # add in the worst behavior of the other heads
    for good_head in range(result.shape[1]):
        result[:, good_head, :] += all_worst_OVs[:, :good_head, :].sum(dim=-2)
        result[:, good_head, :] += all_worst_OVs[:, good_head+1:, :].sum(dim=-2)

    # set diagonal to minimum to avoid including the logit of the min in the worst non-min logit
    result[torch.arange(batch), :, minima] = result.min(dim=-1).values

    return result.max(dim=-1).values

@torch.no_grad()
def count_slack_for_sequences(model, sequences: t.Tensor):
    return (compute_slack_for_sequences(model, sequences) < 0).any(dim=-1).sum().item()

# %%
n_samples = 100000
samples_per_batch = 10000
torch.manual_seed(42)
total_sequences = 0
total_correct = 0
low, high = 0, len(dataset.vocab) - 2
with torch.no_grad():
    for _ in tqdm(range(1 + (n_samples - 1) // samples_per_batch)):
        sequences = torch.randint(low, high, (samples_per_batch, dataset.list_len))
        total_sequences += samples_per_batch
        total_correct += count_slack_for_sequences(model, sequences)
fraction_correct = total_correct / total_sequences
print(f"The model correctly predicts the minimum for at least {fraction_correct * 100}% of {total_sequences} sequences of length {dataset.list_len} with numbers between {low} and {high} inclusive.")



# %% [markdown]
#
# For each count of copies of the minimum, for each minimum token, we can divide the remaining token values into ones that are okay to have all copies of, and ones that are not.  Then we can consider each number of copies of each token that is not okay to have all copies of, and compute which other tokens it's okay to to fill the remainder of the sequence with.  We can then attempt to use this to compute the fraction of sequences that the model gets correct.

# %%
@torch.no_grad()
def compute_worst_OV_3way_shared_data(model, sep_pos=dataset.list_len, max_num_min=dataset.list_len-2, max_num_nonmin1=dataset.list_len-2):
    EPVOU_EUPU = compute_EPVOU_EUPU(model, nanify_sep_position=sep_pos, qtok=-1, qpos=sep_pos)
    # (head, pos, input, output)
    EPVOU_EUPU_sep = EPVOU_EUPU[:, sep_pos, -1, :]
    EPVOU_EUPU = EPVOU_EUPU[:, :sep_pos, :-1, :]
    attn_patterns = compute_3way_attention_patterns_all_counts(model, sep_pos=sep_pos, max_num_min=max_num_min, max_num_nonmin1=max_num_nonmin1)
    # (num_min, num_nonmin1, head, 3, 2, mintok, nonmintok1, nonmintok2, 4)
    # The 3, 2 is for the permutations of attention ordering by position, min vs nonmin1 vs nonmin2 in the lowest attention positions, and then which of the remaining two is in the highest attention position.
    # The 4 is for mintok, nonmintok1, nonmintok2, and sep
    return EPVOU_EUPU, EPVOU_EUPU_sep, attn_patterns

@torch.no_grad()
def compute_worst_OV_3way(model, mintok, sep_pos=dataset.list_len, max_num_min=dataset.list_len-2, max_num_nonmin1=dataset.list_len-2, EPVOU_EUPU=None, EPVOU_EUPU_sep=None, attn_patterns=None):
    '''
    returns (num_min, num_nonmin1, n_heads, d_vocab_nonmin1, d_vocab_nonmin2, d_vocab_out)
    '''
    if EPVOU_EUPU is None or EPVOU_EUPU_sep is None or attn_patterns is None:
        EPVOU_EUPU, EPVOU_EUPU_sep, attn_patterns = compute_worst_OV_3way_shared_data(model, sep_pos=sep_pos, max_num_min=max_num_min, max_num_nonmin1=max_num_nonmin1)
        return compute_worst_OV_3way(model, mintok, sep_pos=sep_pos, max_num_min=max_num_min, max_num_nonmin1=max_num_nonmin1, EPVOU_EUPU=EPVOU_EUPU, EPVOU_EUPU_sep=EPVOU_EUPU_sep, attn_patterns=attn_patterns)

    results = t.zeros(tuple(list(attn_patterns.shape[:-4]) + list(attn_patterns.shape[-3:-1]) + [model.cfg.d_vocab_out]), device=attn_patterns.device)
    # center on the logit for the minimum token
    cur_EPVOU_EUPU = EPVOU_EUPU - EPVOU_EUPU[:, :, :, mintok].unsqueeze(dim=-1)
    # (head, pos, input, output)
    # reduce along position, since they're all nearly identical
    cur_EPVOU_EUPU = cur_EPVOU_EUPU.max(dim=1).values
    # (head, input, output)

    cur_EPVOU_EUPU_sep = EPVOU_EUPU_sep - EPVOU_EUPU_sep[:, mintok].unsqueeze(dim=-1)
    # (head, output)

    cur_attn_patterns = attn_patterns[:, :, :, :, :, mintok, :, :, :]
    # (num_min, num_nommin, head, 3, 2, nonmintok1, nonmintok2, 4)

    results = \
        einsum(cur_attn_patterns[..., 0], cur_EPVOU_EUPU[:, mintok, :],
                "num_min num_nonmin head mini maxi nonmintok1 nonmintok2, head output -> num_min num_nonmin head mini maxi nonmintok1 nonmintok2 output") \
        + einsum(cur_attn_patterns[..., 1], cur_EPVOU_EUPU,
                "num_min num_nonmin head mini maxi nonmintok1 nonmintok2, head nonmintok1 output -> num_min num_nonmin head mini maxi nonmintok1 nonmintok2 output") \
        + einsum(cur_attn_patterns[..., 2], cur_EPVOU_EUPU,
                "num_min num_nonmin head mini maxi nonmintok1 nonmintok2, head nonmintok2 output -> num_min num_nonmin head mini maxi nonmintok1 nonmintok2 output") \
        + einsum(cur_attn_patterns[..., -1], cur_EPVOU_EUPU_sep,
                "num_min num_nonmin head mini maxi nonmintok1 nonmintok2, head output -> num_min num_nonmin head mini maxi nonmintok1 nonmintok2 output")
    # re-center on output logit for minimum token
    results -= results[:, :, :, :, :, :, :, mintok].unsqueeze(dim=-1)

    return reduce(results, "num_min num_nonmin head mini maxi nonmintok1 nonmintok2 output -> num_min num_nonmin head nonmintok1 nonmintok2 output", reduction="max")

@torch.no_grad()
def compute_slack_3way_reduced(model, good_head_max_num_min=dataset.list_len-2, good_head_max_num_nonmin1=dataset.list_len-2, **kwargs):
    '''
    returns (num_min, num_nonmin1, n_heads, d_vocab_min, d_vocab_nonmin1, d_vocab_nonmin2)
    '''
    # we consider the worst behavior of the bad heads with only one copy of the minimum (except for the diagonal, which has the maximum number of copies)
    worst_OV = compute_worst_OV(model, num_min=1, **kwargs)
    # (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)

    results = []
    EPVOU_EUPU, EPVOU_EUPU_sep, attn_patterns = compute_worst_OV_3way_shared_data(model, max_num_min=good_head_max_num_min, max_num_nonmin1=good_head_max_num_nonmin1, **kwargs)

    for mintok in tqdm(range(worst_OV.shape[1])):
        # for the good head, we pass in the number of copies of the minimum token
        worst_OV_good = compute_worst_OV_3way(model, mintok, max_num_min=good_head_max_num_min, max_num_nonmin1=good_head_max_num_nonmin1, EPVOU_EUPU=EPVOU_EUPU, EPVOU_EUPU_sep=EPVOU_EUPU_sep, **kwargs)

        # (num_min, num_nonmin1, n_heads, d_vocab_nonmin1, d_vocab_nonmin2, d_vocab_out)
        cur_results = worst_OV_good.clone()
        for good_head in range(worst_OV.shape[0]):
            # reduce along nonmin tokens in other heads and add them in
            other_worst_OV = torch.cat((worst_OV[:good_head], worst_OV[good_head+1:]), dim=0)
            # (n_heads, d_vocab_min, d_vocab_nonmin, d_vocab_out)

            # # assert centered 1
            # test_slack = cur_results[:, :, good_head, :, :, mintok]
            # test_slack = test_slack[~test_slack.isnan()]
            # assert (test_slack == 0).all(), f"cur_results 1 (good_head={good_head}): {test_slack.max().item()}"

            # # assert centered other
            # test_slack = other_worst_OV[:, mintok, mintok:, mintok]
            # test_slack = test_slack[~test_slack.isnan()]
            # assert (test_slack == 0).all(), f"other_worst_OV: {test_slack.max().item()}"

            cur_results[:, :, good_head] += \
                reduce(other_worst_OV[:, mintok, mintok:, :],
                    "head nonmintok output -> head () () () () output", reduction="max").sum(dim=0)

        # assert centered
        test_slack = cur_results[:, :, :, :, :, mintok]
        test_slack = test_slack[~test_slack.isnan()]
        assert (test_slack == 0).all(), f"cur_results: {test_slack.max().item()}"

        # set diagonal to min to avoid including the logit of the min in the worst non-min logit
        cur_results[:, :, :, :, :, mintok] = cur_results[:, :, :, :, :, :].min(dim=-1).values

        results.append(cur_results.max(dim=-1).values)
    return torch.stack(results, dim=3)

# %%
slacks_3way = compute_slack_3way_reduced(model)
# (num_min, num_nonmin1, n_heads, d_vocab_min, d_vocab_nonmin1, d_vocab_nonmin2)
# %%
# XXX TODO Figure out visualization https://stackoverflow.com/questions/77392813/how-do-i-make-interacting-updates-in-plotly
# zmax = slacks_3way[~slacks_3way.isnan()].abs().max().item()
# # negate for coloring
# slacks_3way_sign = slacks_3way.sign()
# slacks_3way_full = -utils.to_numpy(slacks_3way)
# slacks_3way_sign = -utils.to_numpy(slacks_3way_sign)

# all_tickvals_text = list(enumerate(dataset.vocab[:-1]))
# tickvals_indices = list(range(0, len(all_tickvals_text) - 1, 10)) + [len(all_tickvals_text) - 1]
# tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
# tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]

# for slacks_kind in (slacks_3way_full, slacks_3way_sign):
#     fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=[f"slack on head {h}" for h in range(model.cfg.n_heads)])
#     fig.update_layout(title=f"Slack (positive for either head ⇒ model is correct)")

#     def make_update(h, num_min, num_nonmin1, mintok, showscale=True):
#         cur_slack = slacks_kind[num_min - 1, num_nonmin1 - 1, h, mintok]
#         return go.Heatmap(z=cur_slack, colorscale='RdBu', zmin=-zmax, zmax=zmax, showscale=showscale)

#     def update(num_min, num_nonmin1, mintok):
#         fig.data = []
#         for h in range(model.cfg.n_heads):
#             col, row = h+1, 1
#             fig.add_trace(make_update(h, num_min, num_nonmin1, mintok), col=col, row=row)
#             fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, constrain='domain', col=col, row=row, title_text="non-min tok 2") #, title_text="Output Logit Token"
#             fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, col=col, row=row, title_text="non min tok 1")
#         fig.update_traces(hovertemplate="Non-min token 1: %{y}<br>Non-min token 2: %{x}<br>Slack: %{z}<extra>head %{fullData.name}</extra>")

#     # Create the initial heatmap
#     update(1, 1, 0)

#     # Create frames for each combination of mintok, num_min, and num_nonmin1
#     frames = [go.Frame(
#         data=[make_update(h, num_min, num_nonmin1, mintok) for h in range(model.cfg.n_heads)],
#         name=f"{num_min}-{num_nonmin1}-{mintok}"
#     ) for num_min in range(1, dataset.list_len - 1) for num_nonmin1 in range(1, dataset.list_len - 1) for mintok in range(len(dataset.vocab[:-1]))]

#     fig.frames = frames

#     # Create three sliders
#     sliders = [
#         # Slider for num_min
#         dict(
#             active=0,
#             yanchor='top',
#             xanchor='left',
#             currentvalue=dict(font=dict(size=20), prefix='# copies of min token:', visible=True, xanchor='right'),
#             transition=dict(duration=0),
#             pad=dict(b=10, t=50),
#             len=0.3,
#             x=0.1,
#             y=0,
#             steps=[dict(args=[[f"{num_min}-{num_nonmin1}-{mintok}"], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
#                         method='animate',
#                         label=str(num_min)) for num_min, num_nonmin1, mintok, frame in enumerate(fig.frames) if num_min == mintok]
#         ),
#         # Slider for num_nonmin1
#         dict(
#             active=0,
#             yanchor='top',
#             xanchor='left',
#             currentvalue=dict(font=dict(size=20), prefix='Num nonmin1:', visible=True, xanchor='right'),
#             transition=dict(duration=0),
#             pad=dict(b=10, t=150),
#             len=0.3,
#             x=0.1,
#             y=0.4,
#             steps=[dict(args=[[f"{num_min}-{num_nonmin1}-{mintok}"], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
#                         method='animate',
#                         label=str(num_nonmin1)) for num_min, num_nonmin1, mintok, frame in enumerate(fig.frames) if num_nonmin1 == mintok]
#         ),
#         # Slider for mintok
#         dict(
#             active=0,
#             yanchor='top',
#             xanchor='left',
#             currentvalue=dict(font=dict(size=20), prefix='Mintok:', visible=True, xanchor='right'),
#             transition=dict(duration=0),
#             pad=dict(b=10, t=250),
#             len=0.3,
#             x=0.1,
#             y=0.8,
#             steps=[dict(args=[[f"{num_min}-{num_nonmin1}-{mintok}"], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
#                         method='animate',
#                         label=str(mintok)) for num_min, num_nonmin1, mintok, frame in enumerate(fig.frames)]
#         )
#     ]

#     fig.update_layout(
#         sliders=sliders
#     )

#     fig.show()

# %%
@torch.no_grad()
def calculate_correct_minimum_lower_bound_3way(model, sep_pos=dataset.list_len, slacks_3way=None):
    count = 0
    if slacks_3way is None: slacks_3way = compute_slack_3way_reduced(model, sep_pos=sep_pos, good_head_max_num_min=sep_pos-2, good_head_max_num_nonmin1=sep_pos-2)
    # (num_min, num_nonmin1, n_heads, d_vocab_min, d_vocab_nonmin1, d_vocab_nonmin2)
    slacks_3way = (slacks_3way < 0)
    for min_token_copies in tqdm(range(1, sep_pos)):
        slack = compute_slack_reduced(model, good_head_num_min=min_token_copies, sep_pos=sep_pos)
        # (n_heads, d_vocab_min, d_vocab_nonmin)
        # find where we have slack
        slack = (slack < 0)
        for mintok in range(slack.shape[1]):
            # count the number of sequences with the specified number of copies of mintok and with other tokens drawn from the values where we have slack
            # we reduce along heads to find out which head permits us the most values with slack;
            # since the proof of convexity only works when we fix which head we're analyzing, we can't union permissible values across heads
            n_allowed_values, maxhead = slack[:, mintok, mintok+1:].sum(dim=-1).max(dim=0)
            n_allowed_values = n_allowed_values.item()
            if n_allowed_values > 0:
                count += math.comb(sep_pos, min_token_copies) * n_allowed_values ** (sep_pos - min_token_copies)
                for nonmin_token1_copies in range(1, sep_pos - min_token_copies):
                    # find all disallowed values which have enough slack on all allowed values as the nonmintok2
                    cur_3way_slack = slacks_3way[min_token_copies - 1, nonmin_token1_copies - 1, :, mintok, mintok+1:, mintok+1:]
                    # (head, d_vocab_nonmin1, d_vocab_nonmin2)
                    # subset the 3-way slack values to the ones where mintok1 is a token without enough (2-way) slack, and consider the cases where it does have enough (3-way) slack on all tokens with enough 2-way slack
                    # pytorch slicing is arcane; we want cur_3way_slack[:, ~slack[maxhead, mintok, mintok+1:], slack[maxhead, mintok, mintok+1:]], but this is not how pytorch does boolean slicing (which can only produce flat tensors), and we need to broadcast tensor int indices
                    cur_3way_slack = cur_3way_slack[torch.arange(0, cur_3way_slack.shape[0], device=cur_3way_slack.device)[:, None, None],
                                                    torch.arange(0, cur_3way_slack.shape[1], device=cur_3way_slack.device)[~slack[maxhead, mintok, mintok+1:]][None, :, None],
                                                    torch.arange(0, cur_3way_slack.shape[2], device=cur_3way_slack.device)[slack[maxhead, mintok, mintok+1:]][None, None, :]].all(dim=-1)
                    # (head, d_vocab_nonmin1)
                    # count how many tokens for nonmintok1 have enough slack in this way
                    n_allowed_nonmintok1_values = cur_3way_slack.sum(dim=-1).max().item()
                    if n_allowed_nonmintok1_values > 0:
                        count += math.comb(sep_pos, min_token_copies) * math.comb(sep_pos - min_token_copies, nonmin_token1_copies) * n_allowed_nonmintok1_values ** nonmin_token1_copies * n_allowed_values ** (sep_pos - min_token_copies - nonmin_token1_copies)
        if min_token_copies == sep_pos - 1:
            # add in the diagonal on this last round
            count += slack.any(dim=0).diagonal(dim1=-2, dim2=-1).sum().item()

    return count

@torch.no_grad()
def calculate_correct_minimum_lower_bound_fraction_3way(model, sep_pos=dataset.list_len, **kwargs):
    total_sequences = (model.cfg.d_vocab - 1) ** sep_pos
    return calculate_correct_minimum_lower_bound_3way(model, sep_pos=sep_pos, **kwargs) / total_sequences

# # %%
# slacks_3way = compute_slack_3way_reduced(model)
# # %%
# print(f"Assuming no errors in our argument, we can now prove that the model computes the correct minimum of the sequence in at least {calculate_correct_minimum_lower_bound(model)} cases ({100 * calculate_correct_minimum_lower_bound_fraction_3way(model, slacks_3way=slacks_3way):.1f}% of the sequences).")
# %%
print(f"Assuming no errors in our argument, we can now prove that the model computes the correct minimum of the sequence in at least {calculate_correct_minimum_lower_bound(model)} cases ({100 * calculate_correct_minimum_lower_bound_fraction_3way(model):.1f}% of the sequences).")

# %%
