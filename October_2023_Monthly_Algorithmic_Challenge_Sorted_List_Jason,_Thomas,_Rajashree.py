# %% [markdown]
# <a href="https://colab.research.google.com/github/JasonGross/neural-net-coq-interp/blob/main/October_2023_Monthly_Algorithmic_Challenge_Sorted_List_Jason%2C_Thomas%2C_Rajashree.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Monthly Algorithmic Challenge (October 2023): Sorted List
#
# This post is the fourth in the sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.
#
#
# If you prefer, you can access the Streamlit page [here](https://arena-ch1-transformers.streamlit.app/Monthly_Algorithmic_Problems).
#
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sorted-problem.png" width="350">

# %% [markdown]
# ## Setup

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

# %% [markdown]
#
#
# ## Prerequisites
#
# The following ARENA material should be considered essential:
#
# * **[1.1] Transformer from scratch** (sections 1-3)
# * **[1.2] Intro to Mech Interp** (sections 1-3)
#
# The following material isn't essential, but is recommended:
#
# * **[1.2] Intro to Mech Interp** (section 4)
# * **[1.4] Balanced Bracket Classifier** (all sections)
# * Previous algorithmic problems in the sequence
#

# %% [markdown]
# ## Difficulty
#
# **This problem is probably the easiest in the sequence so far**, so I expect solutions to have fully reverse-engineered it, as well as presenting adversarial examples and explaining how & why they work.**

# %% [markdown]
#
# ## Motivation
#
# Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.
#
# The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.
#
# Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

# %% [markdown]
# ## Logistics
#
# The solution to this problem will be published on this page at the start of November, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.
#
# If you try to interpret this model, you can send your attempt in any of the following formats:
#
# * Colab notebook,
# * GitHub repo (e.g. with ipynb or markdown file explaining results),
# * Google Doc (with screenshots and explanations),
# * or any other sensible format.
#
# You can send your attempt to me (Callum McDougall) via any of the following methods:
#
# * The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
# * My personal email: `cal.s.mcdougall@gmail.com`
# * LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)
#
# **I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.** It's possible that future challenges will also feature a monetary prize, but this is not guaranteed.
#
# Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st August.

# %% [markdown]
# ## What counts as a solution?
#
# Going through the solutions for the previous problem in the sequence (July: Palindromes) as well as the exercises in **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:
#
# * Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
# * Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
# * (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

# %% [markdown]
# ## Task & Dataset
#
# The problem for this month is interpreting a model which has been trained to sort a list. The model is fed sequences like:
#
# ```
# [11, 2, 5, 0, 3, 9, SEP, 0, 2, 3, 5, 9, 11]
# ```
#
# and has been trained to predict each element in the sorted list (in other words, the output at the `SEP` token should be a prediction of `0`, the output at `0` should be a prediction of `2`, etc).
#
# Here is an example of what this dataset looks like:

# %%
dataset = SortedListDataset(size=1, list_len=5, max_value=10, seed=42)

print(dataset[0].tolist())
print(dataset.str_toks[0])

# %% [markdown]
# The relevant files can be found at:
#
# ```
# chapter1_transformers/
# └── exercises/
#     └── monthly_algorithmic_problems/
#         └── october23_sorted_list/
#             ├── model.py               # code to create the model
#             ├── dataset.py             # code to define the dataset
#             ├── training.py            # code to training the model
#             └── training_model.ipynb   # actual training script
# ```
#

# %% [markdown]
# ## Model
#
# The model is attention-only, with 1 layer, and 2 attention heads per layer. It was trained with layernorm, weight decay, and an Adam optimizer with linearly decaying learning rate.
#
# You can load the model in as follows:
#

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

# %% [markdown]
# The code to process the state dictionary is a bit messy, but it's necessary to make sure the model is easy to work with. For instance, if you inspect the model's parameters, you'll see that `model.ln_final.w` is a vector of 1s, and `model.ln_final.b` is a vector of 0s (because the weight and bias have been folded into the unembedding).

# %%
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)

# %% [markdown]
# <details>
# <summary>Aside - the other weight processing parameters</summary>
#
# Here's some more code to verify that our weights processing worked, in other words:
#
# * The unembedding matrix has mean zero over both its input dimension (`d_model`) and output dimension (`d_vocab`)
# * All writing weights (i.e. `b_O`, `W_O`, and both embeddings) have mean zero over their output dimension (`d_model`)
# * The value biases `b_V` are zero (because these can just be folded into the output biases `b_O`)
#
# ```python
# W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
# t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))
#
# W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
# t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))
#
# W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
# t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))
#
# b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
# t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))
#
# W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
# t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))
#
# W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
# t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))
#
# b_V = model.b_V
# t.testing.assert_close(b_V, t.zeros_like(b_V))
# ```
#
# </details>

# %% [markdown]
#
# A demonstration of the model working:
#

# %%
from eindex import eindex

N = 500
dataset = SortedListDataset(size=N, list_len=10, max_value=50, seed=43)

logits, cache = model.run_with_cache(dataset.toks)
logits: t.Tensor = logits[:, dataset.list_len:-1, :]

targets = dataset.toks[:, dataset.list_len+1:]

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

avg_cross_entropy_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")

# %% [markdown]
# And a visualisation of its probability output for a single sequence:

# %%
def show(i):

    imshow(
        probs[i].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[i][j]}<br><sub>({j})</sub>" for j in range(dataset.list_len+1, dataset.seq_len)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>Unsorted = ({','.join(dataset.str_toks[i][:dataset.list_len])})",
        text=[
            ["〇" if (str_tok == target) else "" for target in dataset.str_toks[i][dataset.list_len+1: dataset.seq_len]]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=1000,
    )

show(0)

# %% [markdown]
# Best of luck! 🎈

# %% [markdown]
# The algorithm the transformer seems to run is roughly:
# **Find the smallest value not smaller than the current token, which hasn't been "cancelled" by an equivalent copy appearing already in the sorted list**
#
# Head 0 is mostly doing the cancelling, while head 1 is mostly doing the copying, except for token values around 28--37 where head 0 is doing copying and head 1 is doing nothing.
#
# Additional notes:
# - The skip connection (embed -> unembed) is a small bias against the current token, a smaller bias against numbers less than the current token, and a smaller bias in favor of numbers greater than the current token.
# - The layernorm scaling is fairly uniform at positions on the unsorted list but a bit less uniform on the sorted prefix (after the SEP token)
# - It seems like the cancelling doesn't work that well when there are tokens in the range where the head behavior is swapped, so most of the computation should work even in the absence of cancelling.  The cancelling presumably just tips the scales in marginal cases (and cases where there are duplicates), since most of the head's capacity is devoted to positive copying when such tokens are present.

# %% [markdown]
# To validate this hypothesis, we need to establish a couple of assertions:
#
# Let $S$ be the range of swapped tokens, $S = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37]$.
#
# Let $h_{k}$ denote head 0 for tokens $k \in S$ and head 1 otherwise.
#
# 1. When the query token is SEP in position 10, we find the minimum of the sequence
# 2. When the query token is 50 in position 19, we emit 50
# 3. When the query token is anything other than 50 in position 19, we emit the maximum of the sequence
# 4. When the query is in positions between 11 and 18 inclusive, we follow the rough algorithm above.
#
# We can break down:
# 1. Breakdown:
#    1. Attention by head $h_{k}$ is mostly monotonic decreasing in the value of the token $k$
#    2. The OV circuit on head $h_{k}$ copies the value $k$ more than anything else
#    3. We pay enough more attention to the smallest token than to everything else combined and copy $k$ enough more than anything else that when we combine the effects of the two heads on other tokens, we still manage to copy the correct token.
# 2. Breakdown:
#    1. The copying effects from attending to 50 in position 19 and one additional 50 in some position before 10 gives enough difference between 50 and anything else that we don't care what happens elsewhere.
# 3. Breakdown (TODO: Check if this is right, it might not be)
#    1. Attention by head $h_{k}$ in position 19 is mostly monotonic increasing in the value of the token $k$
#    2. The OV circuit on head $h_{k}$ copies the value $k$ more than anything else
#    3. We pay enough more attention to the largest token than to everything else combined and copy $k$ enough more than anything else that when we combine the effects of the two heads on other tokens, we still manage to copy the correct token.
# 4. Breakdown:
#    1. For $k_1, k_2, q \not\in S$ with $k_1 < q \le k_2$, head 1 pays more attention to $k_2$ in positions before 10 than to $k_1$ in any position
#    2. For $k_1, k_2, q \not\in S$ with $k_1 = q \le k_2$, head 1 pays more attention to $k_2$ in positions before 10 than to $k_1$ in positions after 10
#    3. For $k_1, k_2, q \not\in S$ with $q \le k_1 < k_2$, head 1 pays more attention to $k_1$ in positions before 10 than to $k_2$ in positions before 10
#    4. For $k_2 \in S$ with $k_1 < q \le k_2$, head 0 pays more attention to $k_2$ in positions before 10 than to $k_1$ in any position
#    5. For $k_2 \in S$ with $k_1 = q \le k_2$, head 0 pays more attention to $k_2$ in positions before 10 than to $k_1$ in positions after 10
#    6. For $k_1 \in S$ with $q \le k_1 < k_2$, head 0 pays more attention to $k_1$ in positions before 10 than to $k_2$ in positions before 10
#    7. TODO more stuff

# %% [markdown]
# # Setup
#

# %% [markdown]
# ## Imports

# %%
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

# %% [markdown]
# ## Utils

# %%
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", line_labels=None, **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, y=line_labels, **kwargs).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


def hist(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.histogram(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


# %% [markdown]
# # Computation of Matrices (used in both visualization and as a cached computation in proofs)

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
    attn_all = einsum(q_all, k_all, f'qpos qtok h d_head, kpos ktok h d_head -> {outdim}')
    attn_all_max = reduce(attn_all, f"{outdim} -> {outdim.replace('kpos', '()').replace('ktok', '()')}", 'max')
    # print(attn_all_max.shape)
    # print(attn_all.shape)
    # #attn_all[:,dataset.list_len:-1,:,:-1,:]
    attn_all = attn_all - attn_all_max
    if nanify_sep_loc is not None: attn_all = nanify_attn_sep_loc(attn_all, outdim=outdim, sep_loc=nanify_sep_loc, nanify_bad_self_attention=nanify_bad_self_attention)
    return attn_all
# %%
def compute_EPVOU(model, nanify_sep_position=dataset.list_len):
    EPV = model.blocks[0].ln1(model.W_pos[:, None, :] + model.W_E[None, :, :])[None, :, :, :] @ model.W_V[0,:,None,:,:] + model.b_V[0, :, None, None, :]
    # (head, pos, input, d_head)
    EPVO = EPV @ model.W_O[0,:,None,:,:] + model.b_O[0, None, None, None, :]
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
def compute_EUPU(model, nanify_sep_position=dataset.list_len):
    EUPU = layernorm_noscale(model.W_pos[:, None, :] + model.W_E[None, :, :]) @ model.W_U + model.b_U[None, None, :]
    EUPU = EUPU - EUPU.mean(dim=-1, keepdim=True)
    if nanify_sep_position is not None:
        # SEP is the token in the SEP position
        EUPU[nanify_sep_position, :-1, :] = float('nan')
        # SEP never occurs in positions other than the SEP position
        EUPU[:nanify_sep_position, -1, :], EUPU[nanify_sep_position+1:, -1, :] = float('nan'), float('nan')
    return EUPU

# %% [markdown]
# # Exploratory Plots
#
# Before diving into the proof, we provide some plots that may help with understanding the above claims.  These are purely exploratory (aimed at hypothesis generation) and are not required for hypothesis validation.

# %% [markdown]
# ## Initial Layernorm Scaling

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

# %%
s = layernorm_scales(model.W_pos[:,None,:] + model.W_E[None,:,:])[...,0]
# the only token in position 10 is SEP
s[dataset.list_len, :-1] = float('nan')
# SEP never occurs in positions other than 10
s[:dataset.list_len, -1:], s[dataset.list_len+1:, -1:] = float('nan'), float('nan')
# we don't actually care about the prediction in the last position
s = s[:-1, :]
smin = s[~s.isnan()].min()
s = s / smin
imshow(s, xaxis="Token Value", x=dataset.vocab, yaxis="Position", title=f"Layer Norm Scaling (multiplied by {smin:.3f})")

# %% [markdown]
# ## Attention Plots

# %%
attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=dataset.list_len)
fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=("Head 0", "Head 1"))
fig.update_layout(title="Attention from SEP to other tokens and positions")
all_tickvals_text = list(enumerate(dataset.vocab))
tickvals_indices = list(range(0, len(all_tickvals_text) - 2, 10)) + [len(all_tickvals_text) - 2, len(all_tickvals_text) - 1]
tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]
attn_subset = attn_all[:, dataset.list_len, :dataset.list_len+1, -1, :]
zmin, zmax = attn_subset[~attn_subset.isnan()].min().item(), attn_subset[~attn_subset.isnan()].max().item()
for h in range(model.cfg.n_heads):
    fig.add_trace(go.Heatmap(z=utils.to_numpy(attn_subset[h]), colorscale='Viridis', zmin=zmin, zmax=zmax), row=1, col=h+1)
    fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, title_text="Key Token", row=1, col=h+1)
    fig.update_yaxes(title_text="Position of Key", row=1, col=h+1)
fig.show()

# imshow(attn_all[], cmap='viridis')

# %%
# Here you may want to determine an appropriate grid size based on the number of plots
# For example, if you have a total of 12 plots, you might choose a 3x4 grid
n_rows = 4  # Example value, adjust as needed
n_cols = dataset.list_len  # Example value, adjust as needed

# Create a figure and a grid of subplots
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10))
fig.suptitle(f"Attention from:to")
# Flatten the 2D axis array to easily iterate over it in a single loop
ax = ax.flatten()

attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok', nanify_sep_loc=dataset.list_len)

default_attn = t.zeros_like(attn_all[0, 0, 0])
default_attn[:, :] = float('nan')
default_attn = default_attn.detach().cpu().numpy()
for r in range(n_rows):
    for c in range(n_cols):
        ax[r * n_rows + c].imshow(default_attn, cmap='viridis')
        ax[r * n_rows + c].set_title(f"")

def compute_attn_maps(qpos):
    attn_maps = []
    for kpos in range(qpos+1):
        for h in range(2):
            attn = attn_all[h, qpos, kpos, :, :]
            attn = attn.detach().cpu().numpy()

            attn_maps.append((h, kpos, attn))
    return attn_maps

def update(qpos):
    # Clear previous plots
    for a in ax:
        a.clear()

    attn_maps = compute_attn_maps(qpos)
    fig.suptitle(f"Attention head:key_position, x=key token, y=query token, query position={qpos}")

    for plot_idx, (h, kpos, attn) in enumerate(attn_maps):
        plot_idx = h * n_cols * 2 + kpos
            # break

        # Plot heatmap
        cax = ax[plot_idx].imshow(attn, cmap='viridis')

        # Set title and axis labels if desired
        ax[plot_idx].set_title(f"{h}:{kpos}")
        # ax[plot_idx].set_xlabel("key tok")
        # ax[plot_idx].set_ylabel("query tok")

        # Optionally add a colorbar
        # plt.colorbar(cax, ax=ax[plot_idx])

    plt.tight_layout()

# Create animation
ani = FuncAnimation(fig, update, frames=tqdm(range(dataset.list_len+1, dataset.seq_len-1)), interval=1000)
# ani = FuncAnimation(fig, update, frames=tqdm(range(dataset.list_len, dataset.list_len+2)))#, interval=1000)

# plt.show()
HTML(ani.to_jshtml())

# # Counter for the subplot index
# # plot_idx = 0

# for qpos in range(dataset.list_len, dataset.seq_len-1):
#     for kpos in range(qpos+1):
#         for h in range(2):
#             attn = model.blocks[0].ln1(model.W_pos[qpos,:] + model.W_E) @ model.W_Q[0,h,:,:] @ model.W_K[0,h,:,:].T @ model.blocks[0].ln1(model.W_pos[kpos,:] + model.W_E).T
#             attn = attn - attn.max(dim=-1, keepdim=True).values
#             attn = attn.detach().cpu().numpy()
#             # Check if the counter exceeds the available subplots and create new figure if needed
#             if plot_idx >= n_rows * n_cols:
#                 fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
#                 ax = ax.flatten()
#                 plot_idx = 0

#             # Plot heatmap
#             cax = ax[plot_idx].imshow(attn, cmap='viridis')

#             # Set title and axis labels if desired

#             ax[plot_idx].set_title(f"{h}: {qpos} -> {kpos}")
#             ax[plot_idx].set_xlabel("key tok")
#             ax[plot_idx].set_ylabel("query tok")

#             # Optionally add a colorbar
#             plt.colorbar(cax, ax=ax[plot_idx])

#             # Increment the counter
#             plot_idx += 1

#             # You might want to save or display the plot here if you're iterating over many subplots
#             if plot_idx == 0:
#                 plt.tight_layout()
#                 plt.show()

# %% [markdown]
# ## OV Attention Head Plots

# %%
n_rows_per_head = 4
n_rows = n_rows_per_head * model.cfg.n_heads
n_cols = 1 + (dataset.seq_len - 1) // n_rows_per_head
subplot_titles = [(f"h{h},p{pos}" if pos < dataset.seq_len - 1 else "") for h in range(model.cfg.n_heads) for pos in range(n_cols * n_rows_per_head)]
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
fig.update_annotations(font_size=12)
# fig.update_layout(
#     annotations=[
#         dict(font=dict(size=5))
#         for _ in subplot_titles
#     ]
# )
fig.update_layout(title="OV Logit Impact: x=Ouput Logit Token, y=Input Token<br>LN_noscale(LN1(W_pos[pos,:] + W_E) @ W_V[0,h] @ W_O[0, h]) @ W_U",
                  height=800)
all_tickvals_text = list(enumerate(dataset.vocab))
tickvals_indices = list(range(0, len(all_tickvals_text) - 1, 20)) + [len(all_tickvals_text) - 1]
tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]

EPVOU = compute_EPVOU(model, nanify_sep_position=dataset.list_len)
zmax = EPVOU[~EPVOU.isnan()].abs().max().item()
EPVOU = utils.to_numpy(EPVOU)

for h in range(model.cfg.n_heads):
    for i, pos in enumerate(range(dataset.seq_len-1)):
        r, c = h * n_rows_per_head + i // n_cols, i % n_cols
        cur_EPVOU = EPVOU[h, pos]
        if pos == dataset.list_len: cur_EPVOU = cur_EPVOU[-1:, :]
        fig.add_trace(go.Heatmap(z=utils.to_numpy(cur_EPVOU), colorscale='RdBu', zmin=-zmax, zmax=zmax, showscale=(i==0 and h == 0)), row=r+1, col=c+1)
        fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, constrain='domain', row=r+1, col=c+1) #, title_text="Output Logit Token"
        if pos == dataset.list_len:
            fig.update_yaxes(range=[-1,1], row=r+1, col=c+1)
        else:
            fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, row=r+1, col=c+1)
fig.show()

# %%
EPVOU = compute_EPVOU(model, nanify_sep_position=dataset.list_len)
zmax = EPVOU[~EPVOU.isnan()].abs().max().item()
EPVOU = utils.to_numpy(EPVOU)

fig, ax = plt.subplots(1, model.cfg.n_heads, figsize=(20, 10))
fig.suptitle(f"OV Logit Impact: x=Ouput Logit Token, y=Input Token\nLN_noscale(LN1(W_pos[pos,:] + W_E) @ W_V[0,h] @ W_O[0, h]) @ W_U")
# Flatten the 2D axis array to easily iterate over it in a single loop
ax = ax.flatten()

colorbar_added = False
def update(pos):
    global colorbar_added
    # Clear previous plots
    for a in ax:
        a.clear()

    fig.suptitle(f"OV Logit Impact: x=Ouput Logit Token, y=Input Token\nLN_noscale(LN1(W_pos[{pos},:] + W_E) @ W_V[0,h] @ W_O[0, h]) @ W_U")

    for h, cur_EPVOU in enumerate(EPVOU[:, pos]):
        cax = ax[h].imshow(cur_EPVOU, cmap='RdBu', vmin=-zmax, vmax=zmax)
        ax[h].set_title(f"head {h}")
        # ax[h].set_xlabel("output tok")
        # ax[h].set_ylabel("input tok")
        # Optionally add a colorbar
        if not colorbar_added:
            fig.colorbar(cax, ax=ax[h], fraction=0.05, pad=0.05)
    colorbar_added = True

    # plt.tight_layout()

# Create animation
ani = FuncAnimation(fig, update, frames=tqdm(range(dataset.seq_len-1)), interval=1000)
# ani = FuncAnimation(fig, update, frames=tqdm(range(2)))#, interval=1000)

# plt.show()
HTML(ani.to_jshtml())

# %%
EPVOU = compute_EPVOU(model, nanify_sep_position=dataset.list_len)
zmax = EPVOU[~EPVOU.isnan()].abs().max().item()
EPVOU = utils.to_numpy(EPVOU)

fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=[f"head {h}" for h in range(model.cfg.n_heads)])
# fig.update_annotations(font_size=12)
fig.update_layout(title="OV Logit Impact: x=Ouput Logit Token, y=Input Token<br>LN_noscale(LN1(W_pos[pos,:] + W_E) @ W_V[0,h] @ W_O[0, h]) @ W_U")

all_tickvals_text = list(enumerate(dataset.vocab))
tickvals_indices = list(range(0, len(all_tickvals_text) - 1, 10)) + [len(all_tickvals_text) - 1]
tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]

def make_update(pos, h, adjust_sep=True):
    cur_EPVOU = EPVOU[h, pos]
    if adjust_sep and pos == dataset.list_len: cur_EPVOU = cur_EPVOU[-1:, :]
    return go.Heatmap(z=utils.to_numpy(cur_EPVOU), colorscale='RdBu', zmin=-zmax, zmax=zmax, showscale=(h == 0))

def update(pos, update_title=True):
    fig.data = []
    for h in range(model.cfg.n_heads):
        fig.add_trace(make_update(pos, h), row=1, col=h+1)
        fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, constrain='domain', row=1, col=h+1) #, title_text="Output Logit Token"
        if pos == dataset.list_len:
            fig.update_yaxes(range=[-1,1], row=1, col=h+1)
        else:
            fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, row=1, col=h+1)
    if update_title: fig.update_layout(title=f"OV Logit Impact: x=Ouput Logit Token, y=Input Token<br>LN_noscale(LN1(W_pos[{pos},:] + W_E) @ W_V[0,h] @ W_O[0, h]) @ W_U")

# Create the initial heatmap
update(0, update_title=False)  # Assuming 0 is a valid starting position

# Create frames for each position
frames = [go.Frame(
    data=[make_update(pos, h, adjust_sep=False) for h in range(model.cfg.n_heads)],
    name=str(pos)
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

# ani = FuncAnimation(fig, update, frames=tqdm(range(dataset.seq_len-1)), interval=1000)

# HTML(ani.to_jshtml())

# %% [markdown]
# ## Skip Connection / Residual Stream Plots

# %%
n_rows = 2
n_cols = 1 + (dataset.list_len - 1) // n_rows
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"pos={pos}" for pos in range(dataset.list_len, dataset.seq_len - 1)])
fig.update_layout(title="Logit Impact from the embedding (without layernorm scaling): LN_noscale(W_pos[pos,:] + W_E) @ W_U, x=Ouput Logit Token, y=Input Token")
all_tickvals_text = list(enumerate(dataset.vocab))
tickvals_indices = list(range(0, len(all_tickvals_text) - 2, 10)) + [len(all_tickvals_text) - 2, len(all_tickvals_text) - 1]
tickvals = [all_tickvals_text[i][0] for i in tickvals_indices]
tickvals_text = [all_tickvals_text[i][1] for i in tickvals_indices]

EUPU = compute_EUPU(model, nanify_sep_position=dataset.list_len)
zmax = EUPU[~EUPU.isnan()].abs().max().item()
EUPU = utils.to_numpy(EUPU)
for i, pos in enumerate(range(dataset.list_len, dataset.seq_len-1)):
    r, c = i // n_cols, i % n_cols
    cur_EUPU = EUPU[pos]
    if pos == dataset.list_len: cur_EUPU = cur_EUPU[-1:, :]
    fig.add_trace(go.Heatmap(z=utils.to_numpy(cur_EUPU), colorscale='RdBu', zmin=-zmax, zmax=zmax, showscale=(i==0)), row=r+1, col=c+1)
    fig.update_xaxes(tickvals=tickvals, ticktext=tickvals_text, constrain='domain', row=r+1, col=c+1) #, title_text="Output Logit Token"
    if pos == dataset.list_len:
        fig.update_yaxes(range=[-1,1], row=r+1, col=c+1)
    else:
        fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1, row=r+1, col=c+1)
fig.show()


# %% [markdown]
# # Finding the Minimum with query SEP in Position 10

# %% [markdown]
# ## When the sequence contains nothing in $S$

# %% [markdown]
# ### More attention is paid to the minimum by head 1 than to anything else
#
# Operationalization:
#
# Simplest version:
# Independence relaxations:
# - Attention to token is independent of key position
# - Off-diagonal OV behavior is independent of which wrong answer we're considering
# - Attention to non-min tokens is independent of OV behavior
# - The EUPU behavior is irrelevant (independent of everything else)
# - Attention paid to non-min tokens is independent between the two heads
#
# Operationalization:
# - For each possible minimum sequence value ktokmin:
# - The sum of the two OV matrices on input key ktokmin makes the logit for ktokmin larger than any other logit by more than the largest logit gap in EUPU
# - The attention correction that comes from either head paying more attention to anything else is small enough that the logit gap is still large enough
#

# %% [markdown]
#
# We first compute the skip connection / EUPU (Embed @ Unembed + PosEmbed @ Unembed) behavior.

# %%
@torch.no_grad()
def skip_connection_impacts(model: HookedTransformer, pos=dataset.list_len, tok=-1) -> t.Tensor:
    EUPU = compute_EUPU(model)
    # (pos, input, output)
    EUPU = EUPU[pos, tok]
    return t.cat([EUPU[i+1:] - EUPU[i] for i in range(EUPU.shape[0] - 1)])

@torch.no_grad()
def worst_skip_connection_impact(model: HookedTransformer, **kwargs) -> float:
    return skip_connection_impacts(model, **kwargs).max().item()


hist(skip_connection_impacts(model), title="Skip Connection ((W_E + W_pos) @ W_U) Impact on Logits", xaxis="Logit diff (nonmin - min)")
print(worst_skip_connection_impact(model))

# %% [markdown]
#
# Now we compute the distribution of behaviors of OV

# %%
@torch.no_grad()
def ov_impacts(model: HookedTransformer, sep_pos=dataset.list_len) -> t.Tensor:
    EPVOU = compute_EPVOU(model)
    # (head, pos, input, output)
    EPVOU = EPVOU[:, :sep_pos, :-1, :]
    return t.cat([EPVOU[:, :, :, i+1:] - EPVOU[:, :, :, i:i+1] for i in range(EPVOU.shape[-1] - 1)], dim=-1)

@torch.no_grad()
def worst_ov_impact(model: HookedTransformer, **kwargs) -> float:
    return ov_impacts(model, **kwargs).max().item()

hist(ov_impacts(model).flatten(), title="OV Impact on Logits", xaxis="Logit diff (nonmin - min)", renderer='png')
print(worst_ov_impact(model))

# %% [markdown]
#
# Now we compute the copying behavior of OV, which handles the uniform sequences (all 0s, all 1s, all 2s, etc) and gives us a starting point for other sequences
#
# Let's first compute the sum of the OV matrices, as if 100% of the attention were paid to the minimum token, by both heads.

# %%
@torch.no_grad()
def ov_copying_impacts_full_attention(model: HookedTransformer, sep_pos=dataset.list_len) -> t.Tensor:
    EPVOU = compute_EPVOU(model)
    # (head, pos, input, output)
    EPVOU = EPVOU[:, :sep_pos, :-1, :]
    EPVOU = EPVOU.sum(dim=0)
    # (pos, input, output)
    EPVOU = EPVOU - EPVOU.diagonal(dim1=-2, dim2=-1)[:, :, None]
    # remove the diagonal
    diag_mask = torch.zeros_like(EPVOU) + torch.eye(EPVOU.size(-2), EPVOU.size(-1)).to(EPVOU.device)
    return EPVOU[diag_mask == 0]

hist(ov_copying_impacts_full_attention(model))


# %% [markdown]

#
# - Show that network correctly computes the min for the 50 sequences that are uniform (10 0s, 10 1s, 10 2s, etc)
# - For each possible minimum sequence value ktokmin, each possible "worst case" alternate sequence token ktoknonmin, and each possible wrong answer outtok:
# - For each head h:
# - If OV outputting ktokmin is greater than OV outputting outtok, then we want to consider the minimal attention paid to this token across all positions and have as few copies as possible
# - If OV outputting ktokmin is less than OV outputting outtok, then we want to consider the maximal attention paid to this token across all positions and have as many copies as possible
# - Find minimal/maximal attention paid to ktokmin and ktoknonmin across all positions
# - Softmax the attention according to which token should show up most
# - Find the worst-case OVs across all positions, and take the weighted average of the OVs according to the softmaxed attention
# - Validate that the
#
# - Find the minumum attention paid to this token across positions 0--9
# - For each possible non-minimum token ktok:
# - Find the ma

# %%
attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok')# , nanify_sep_loc=dataset.list_len)

# %%
print(f"Self attention with SEP: {attn_all[:, dataset.list_len, dataset.list_len, -1, -1]}")
other_attn = attn_all[:, dataset.list_len, :dataset.list_len, -1, :-1]
# (head, kpos, ktok)
other_attn_min, other_attn_max = other_attn.min(dim=1).values, other_attn.max(dim=1).values
other_attn_min_max = t.stack([other_attn_min, other_attn_max], dim=1)
line(utils.to_numpy(other_attn_min_max[1].T))
line(utils.to_numpy(other_attn_min_max[0].T))

# %%

# model([])

# # %% [markdown]
# # ## Exploratory Plots

# # %%
# from training.analysis_utils import imshow, line, analyze_svd, layernorm_noscale, layernorm_scales

# # %%
# print(dataset[0])
# print(', '.join(dataset.str_toks[0]))
# print(model(dataset[0]))
# print(model(dataset[0]).argmax(dim=-1))
# print([(i, v.item()) for i, v in enumerate(model(dataset[0]).argmax(dim=-1)[0])])

# # %%


# # %%
# imshow(EPVOU)

# # %%
# attention_pattern = cache['blocks.0.attn.hook_pattern'].shape
# cv.attention.attention_patterns(
#     tokens=dataset.str_toks,
#     attention=attention_pattern)

# # %%
# print(dataset.str_toks[0])
# for p in cache['blocks.0.attn.hook_pattern'][0]:
#     lbls = [f'{i}:{s}' for i, s in enumerate(dataset.str_toks[0])]
#     imshow(p, x=lbls, y=lbls)

# # %%
# cache.keys()

# # %%
# model.blocks[0].ln1.b

# # %%
# print(model.W_E.shape)

# # %%


# # %%


# # %%
# dataset[0]
# model(t.tensor([10, 15, 30, 15, 15, 40, 41, 42, 43, 44, 51, 10, 15, 15, 15, 30, 40, 41, 42, 43, 44])).argmax(dim=-1)

# # %%
# dataset.list_len

# # %%
# dataset.seq_len

# # %%


# # %%
# plt.imshow([[1,2,3],[4,5,6]])

# # %%
# model.W_E.shape

# # %%


# # %%


# # %%


# # %%


# # %%
# # Here you may want to determine an appropriate grid size based on the number of plots
# # For example, if you have a total of 12 plots, you might choose a 3x4 grid
# n_rows = 3  # Example value, adjust as needed
# n_cols = 4  # Example value, adjust as needed

# plot_idx = 0

# # Create a figure and a grid of subplots
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 10))

# # Flatten the 2D axis array to easily iterate over it in a single loop
# ax = ax.flatten()


# # Counter for the subplot index
# # plot_idx = 0
# attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok')

# for qpos in range(dataset.list_len, dataset.seq_len-1):
#     for kpos in range(qpos+1):
#         for h in range(2):
#             attn = attn_all[h, qpos, kpos].detach().cpu().numpy()
#             # Check if the counter exceeds the available subplots and create new figure if needed
#             if plot_idx >= n_rows * n_cols:
#                 fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
#                 ax = ax.flatten()
#                 plot_idx = 0

#             # Plot heatmap
#             cax = ax[plot_idx].imshow(attn, cmap='viridis')

#             # Set title and axis labels if desired

#             ax[plot_idx].set_title(f"{h}: {qpos} -> {kpos}")
#             ax[plot_idx].set_xlabel("key tok")
#             ax[plot_idx].set_ylabel("query tok")

#             # Optionally add a colorbar
#             plt.colorbar(cax, ax=ax[plot_idx])

#             # Increment the counter
#             plot_idx += 1

#             # You might want to save or display the plot here if you're iterating over many subplots
#             if plot_idx == 0:
#                 plt.tight_layout()
#                 plt.show()

# # %%
# # Here you may want to determine an appropriate grid size based on the number of plots
# # For example, if you have a total of 12 plots, you might choose a 3x4 grid
# n_rows = 4  # Example value, adjust as needed
# n_cols = 10  # Example value, adjust as needed

# # Create a figure and a grid of subplots
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10))

# # Flatten the 2D axis array to easily iterate over it in a single loop
# ax = ax.flatten()

# attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok')

# def compute_attn_maps(qpos):
#     attn_maps = []
#     for kpos in range(qpos+1):
#         for h in range(2):
#             attn = attn_all[h, qpos, kpos, :, :]
#             attn = attn.detach().cpu().numpy()

#             attn_maps.append((h, kpos, attn))
#     return attn_maps

# def update(qpos):
#     # Clear previous plots
#     for a in ax:
#         a.clear()

#     attn_maps = compute_attn_maps(qpos)

#     for plot_idx, (h, kpos, attn) in enumerate(attn_maps):
#         plot_idx = h * n_cols * 2 + kpos
#             # break

#         # Plot heatmap
#         cax = ax[plot_idx].imshow(attn, cmap='viridis')

#         # Set title and axis labels if desired
#         ax[plot_idx].set_title(f"{h}:{kpos}")
#         # ax[plot_idx].set_xlabel("key tok")
#         # ax[plot_idx].set_ylabel("query tok")

#         # Optionally add a colorbar
#         # plt.colorbar(cax, ax=ax[plot_idx])

#     plt.tight_layout()

# # Create animation
# ani = FuncAnimation(fig, update, frames=tqdm(range(dataset.list_len, dataset.seq_len-1)), interval=1000)

# plt.show()

# from IPython.display import HTML
# HTML(ani.to_jshtml())

# # # Counter for the subplot index
# # # plot_idx = 0

# # for qpos in range(dataset.list_len, dataset.seq_len-1):
# #     for kpos in range(qpos+1):
# #         for h in range(2):
# #             attn = model.blocks[0].ln1(model.W_pos[qpos,:] + model.W_E) @ model.W_Q[0,h,:,:] @ model.W_K[0,h,:,:].T @ model.blocks[0].ln1(model.W_pos[kpos,:] + model.W_E).T
# #             attn = attn - attn.max(dim=-1, keepdim=True).values
# #             attn = attn.detach().cpu().numpy()
# #             # Check if the counter exceeds the available subplots and create new figure if needed
# #             if plot_idx >= n_rows * n_cols:
# #                 fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
# #                 ax = ax.flatten()
# #                 plot_idx = 0

# #             # Plot heatmap
# #             cax = ax[plot_idx].imshow(attn, cmap='viridis')

# #             # Set title and axis labels if desired

# #             ax[plot_idx].set_title(f"{h}: {qpos} -> {kpos}")
# #             ax[plot_idx].set_xlabel("key tok")
# #             ax[plot_idx].set_ylabel("query tok")

# #             # Optionally add a colorbar
# #             plt.colorbar(cax, ax=ax[plot_idx])

# #             # Increment the counter
# #             plot_idx += 1

# #             # You might want to save or display the plot here if you're iterating over many subplots
# #             if plot_idx == 0:
# #                 plt.tight_layout()
# #                 plt.show()

# # %%
# model.ln_final

# # %%
# def ln_final_noscale(x):
#     return x - x.mean(axis=-1, keepdim=True)

# # %%
# # Here you may want to determine an appropriate grid size based on the number of plots
# # For example, if you have a total of 12 plots, you might choose a 3x4 grid
# n_rows = 3  # Example value, adjust as needed
# n_cols = 4  # Example value, adjust as needed

# # Create a figure and a grid of subplots
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))

# # Flatten the 2D axis array to easily iterate over it in a single loop
# ax = ax.flatten()

# # Counter for the subplot index
# plot_idx = 0

# for pos in range(dataset.seq_len-1):
#     for h in range(2):
#         EPVOU = ln_final_noscale(model.blocks[0].ln1(model.W_pos[pos,:] + model.W_E) @ model.W_V[0,h,:,:] @ model.W_O[0,h,:,:]) @ model.W_U
#         EPVOU = EPVOU - EPVOU.mean(dim=-1, keepdim=True)
#         EPVOU = EPVOU.detach().cpu().numpy()
#         # Check if the counter exceeds the available subplots and create new figure if needed
#         if plot_idx >= n_rows * n_cols:
#             fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
#             ax = ax.flatten()
#             plot_idx = 0

#         # Plot heatmap
#         cax = ax[plot_idx].imshow(EPVOU, cmap='RdBu', norm=colors.CenteredNorm())

#         # Set title and axis labels if desired

#         ax[plot_idx].set_title(f"h{h} P{pos}: EVOU+PVOU ln_final_noscale")
#         ax[plot_idx].set_xlabel("output tok")
#         ax[plot_idx].set_ylabel("key tok")

#         # Optionally add a colorbar
#         plt.colorbar(cax, ax=ax[plot_idx])

#         # Increment the counter
#         plot_idx += 1

#         # You might want to save or display the plot here if you're iterating over many subplots
#         if plot_idx == 0:
#             plt.tight_layout()
#             plt.show()

# # %%


# # %%
# # Here you may want to determine an appropriate grid size based on the number of plots
# # For example, if you have a total of 12 plots, you might choose a 3x4 grid
# n_rows = 3  # Example value, adjust as needed
# n_cols = 4  # Example value, adjust as needed

# # Create a figure and a grid of subplots
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))

# # Flatten the 2D axis array to easily iterate over it in a single loop
# ax = ax.flatten()

# # Counter for the subplot index
# plot_idx = 0

# for pos in range(dataset.list_len, dataset.seq_len-1):
#     EUPU = ln_final_noscale(model.W_pos[pos,:] + model.W_E) @ model.W_U
#     EUPU = EUPU.detach().cpu().numpy()
#     # Check if the counter exceeds the available subplots and create new figure if needed
#     if plot_idx >= n_rows * n_cols:
#         fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
#         ax = ax.flatten()
#         plot_idx = 0

#     # Plot heatmap
#     cax = ax[plot_idx].imshow(EUPU, cmap='RdBu')

#     # Set title and axis labels if desired

#     ax[plot_idx].set_title(f"{pos}")
#     ax[plot_idx].set_xlabel("output tok")
#     ax[plot_idx].set_ylabel("key tok")

#     # Optionally add a colorbar
#     plt.colorbar(cax, ax=ax[plot_idx])

#     # Increment the counter
#     plot_idx += 1

#     # You might want to save or display the plot here if you're iterating over many subplots
#     if plot_idx == 0:
#         plt.tight_layout()
#         plt.show()

# # %%
# attn_all = compute_attn_all(model, outdim='h qpos kpos qtok ktok')

# # %%
# analyze_svd(attn_all[1,10,0])

# # %%
# s = layernorm_scales(model.W_pos[:,None,:] + model.W_E[None,:,:])[...,0]
# s = s / s.min()
# imshow(s)
# # resid = model.blocks[0].ln1(model.W_pos[:,None,:] + model.W_E[None,:,:])
# # q_all = einsum(resid,
# #                    model.W_Q[0,:,:,:],
# #                    'qpos qtok d_model_q, h d_model_q d_head -> qpos qtok h d_head') + model.b_Q[0]
# #     k_all = einsum(resid,
# #                    model.W_K[0,:,:,:],
# #                    'kpos ktok d_model_k, h d_model_k d_head -> kpos ktok h d_head') + model.b_K[0]
# #     attn_all = einsum(q_all, k_all, f'qpos qtok h d_head, kpos ktok h d_head -> {outdim}')
# #     attn_all_max = reduce(attn_all, f"{outdim} -> {outdim.replace('kpos', '()').replace('ktok', '()')}", 'max')


# # %%



# x = x - x.mean(axis=-1, keepdim=True)  # [batch, pos, length]
#         scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
#             (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
#         )
#         x = x / scale  # [batch, pos, length]

# # %%
# model.blocks[0].ln1

# # %%
# sys.path.append(f"{os.getcwd()}/training")
# import visualization
# from visualization import visualize_on_input

# # %%
# dataset.toks[0]

# # %%
# visualize_on_input(model, dataset.toks[0])

# # %%
# targets[326]

# # %%
# ' '.join(dataset.str_toks[326])

# # %%

# imshow(logits[326])

# # %%
# ls = [5,6,33,40,41,42,43,44,45,46]
# # ls = [1, 1, 1, 1, 40, 40, 40, 40, 40]
# unsorted_list = t.tensor(ls)
# sorted_list = t.sort(unsorted_list).values
# toks = t.concat([unsorted_list, t.tensor([dataset.vocab.index("SEP")]), sorted_list], dim=-1)
# str_toks = [dataset.vocab[i] for i in toks]
# print(' '.join(str_toks))
# imshow(model(toks.unsqueeze(0))[0, dataset.list_len:-1, :])

# # %%
# probs_correct.min(dim=0,keepdim=True)


# # %%
# logits, cache = model.run_with_cache(dataset.toks)
# logits: t.Tensor = logits[:, dataset.list_len:-1, :]

# targets = dataset.toks[:, dataset.list_len+1:]

# logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
# probs = logprobs.softmax(-1)


# # %%
# model.W_pos.shape

# # %%
# model.W_E.shape

# # %%
# from einops import reduce

# # %%


# # %%
# W_pos_reduced = model.W_pos[dataset.list_len+1:-1]
# W_pos_mean = W_pos_reduced.mean(dim=0)
# W_pos_centered = W_pos_reduced - W_pos_mean[None,:]
# line(W_pos_mean)
# imshow(W_pos_centered)

# # %%
# U, S, Vh = t.svd(model.W_pos.detach().cpu()[:10])
# imshow(U)
# imshow(Vh)
# line(S)
# U, S, Vh = t.svd(model.W_pos.detach().cpu()[11:])
# imshow(U)
# imshow(Vh)
# line(S)

# # %%
# with t.no_grad():
#     # print((model.blocks[0].ln1(model.W_pos[dataset.list_len:-1,None,:] + model.W_E[None,:,:])).shape)
#     # print(model.W_Q[0,:,:,:].shape)
#     # print(model.W_K[0,:,:,:].shape)
#     attn_all = einsum(model.blocks[0].ln1(model.W_pos[dataset.list_len:-1,None,:] + model.W_E[None,:,:]),
#                       model.W_Q[0,:,:,:],
#                       model.W_K[0,:,:,:],
#                       model.blocks[0].ln1(model.W_pos[:-1,None,:] + model.W_E[None,:,:]),
#                    'qpos qtok d_model_q, h d_model_q d_head, h d_model_k d_head, kpos ktok d_model_k -> h qpos kpos qtok ktok')
#     # for h in range(2):
#     #     for qpos in range(attn_all.shape[1]):
#     #         for qtok in range(attn_all.shape[3]):
#     #             attn_all[h,qpos,:,qtok,:] = attn_all[h,qpos,:,qtok,:] - attn_all[h,qpos,:,qtok,:].max()
#     # attn_all = attn_all - attn_all.max(dim=2, keepdim=True).values
#     print(attn_all.shape)
#     attn_all_reduced = attn_all.clone()
#     for qpos in range(attn_all.shape[1]):
#         attn_all_reduced[:,qpos,:,:] = attn_all[:,qpos,:qpos+11,:].mean(dim=1, keepdim=True)
#     attn_all_reduced = attn_all_reduced.mean(dim=2)[:,1:-1].mean(dim=1)
#     print(attn_all_reduced.shape)
#     # subtract off diagonal
#     for h in range(attn_all_reduced.shape[0]):
#         attn_all_reduced[h] = attn_all_reduced[h] - attn_all_reduced[h].diag()[:, None]
#     attn_all_reduced = attn_all_reduced - attn_all_reduced.max(dim=-1,keepdim=True).values
#     # attn_all_reduced = attn_all_reduced - attn_all_reduced.max(dim=-1, keepdim=True).values
#     for h, attn in enumerate(attn_all_reduced):
#         imshow(attn, title=f"h{h} attn_all_reduced")
#     for h, attn in enumerate(attn_all_reduced):
#         U, S, Vh = t.svd(attn)
#         imshow(U, title=f"h{h} U attn_all_reduced")
#         imshow(Vh, title=f"h{h} Vh attn_all_reduced")
#         line(S, title=f"h{h} S attn_all_reduced")

#     # print(attn_all_reduced.shape)

#     # attn_all_reduced = attn_all[:,].mean(dim=1)
#     # attn_all_reduced = reduce(attn_all[:,], 'h qpos kpos qtok ktok -> h qpos kpos qtok', 'sum')
#     # all_attn_by_gap = []
#     # #t.zeros(list(attn_all.shape[:-1]) + [attn_all.shape[-1], attn_all.shape[-1]])
#     # for h in range(2):
#     #     for qpos in range(attn_all.shape[1]):
#     #         for kpos in range(qpos+1):
#     #             for qtok in range(attn_all.shape[3]):
#     #                 for ktok1 in range(attn_all.shape[4]):
#     #                     for ktok2 in range(attn_all.shape[4]):
#     #                         all_attn_by_gap.append(attn_all[h,qpos,kpos,qtok,ktok])

# # %%


# # %%
# model.blocks[0].ln1(model.W_pos[qpos,:] + model.W_E) @ model.W_Q[0,h,:,:] @ model.W_K[0,h,:,:].T @ model.blocks[0].ln1(model.W_pos[kpos,:] + model.W_E).T

# # %%
# attn = model.blocks[0].ln1(model.W_pos[qpos,:] + model.W_E) @ model.W_Q[0,h,:,:] @ model.W_K[0,h,:,:].T @ model.blocks[0].ln1(model.W_pos[kpos,:] + model.W_E).T
#             attn = attn - attn.max(dim=-1, keepdim=True).values
#             attn = attn.detach().cpu().numpy()


# # %%
# # with t.no_grad():
# #     attn_all = einsum(model.blocks[0].ln1(model.W_pos[dataset.list_len:-1,None,:] + model.W_E[None,:,:]),
# #                       model.W_Q[0,:,:,:],
# #                       model.W_K[0,:,:,:],
# #                       model.blocks[0].ln1(model.W_pos[:-1,None,:] + model.W_E[None,:,:]),
# #                    'qpos qtok d_model_q, h d_model_q d_head, h d_model_k d_head, kpos ktok d_model_k -> h qpos kpos qtok ktok')


# # %%
# model.W_O.shape

# # %%
# with t.no_grad():
#     EP = model.W_pos[:-1,None,:] + model.W_E[None,:,:]
#     EP = model.blocks[0].ln1(EP)
#     EPVO =  einsum(EP,
#                    model.W_V[0,:,:,:],
#                    model.W_O[0,:,:,:],
#                    '''pos vocab_in d_model_v,
#                     h d_model_v d_head,
#                     h d_head d_model_o
#                       -> h pos vocab_in d_model_o'''.replace('\n', ' '))
#     EPVO = ln_final_noscale(EPVO)
#     EPVOU = einsum(EPVO,
#                      model.W_U,
#                      '''h pos vocab_in d_model_o,
#                      d_model_o vocab_out
#                      -> h pos vocab_in vocab_out'''.replace('\n', ' '))
#     EPVOU_mean = t.cat([EPVOU[:,:dataset.list_len,:,:], EPVOU[:,dataset.list_len+1:-1,:,:]], dim=1).mean(dim=1,keepdim=True)
#     EPVOU = EPVOU - EPVOU_mean
#     # EPVOU_mean = EPVOU_mean - EPVOU_mean.max(dim=-1, keepdim=True).values
#     for h in range(2):
#         imshow(EPVOU_mean[h, 0], title=f"h{h} EPVOU.mean(dim=pos)")

#     for h in range(2):
#         line(EPVOU_mean[h, 0].diag(), title=f"h{h} EPVOU.mean(dim=pos).diag()")
#     line(EPVOU_mean[:, 0].sum(dim=0).diag(), title=f"h{h} EPVOU.mean(dim=pos).sum(dim=head).diag()")
#     # Here you may want to determine an appropriate grid size based on the number of plots
#     # For example, if you have a total of 12 plots, you might choose a 3x4 grid
#     n_rows = 3  # Example value, adjust as needed
#     n_cols = 4  # Example value, adjust as needed

#     # Create a figure and a grid of subplots
#     fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))

#     # Flatten the 2D axis array to easily iterate over it in a single loop
#     ax = ax.flatten()

#     # Counter for the subplot index
#     plot_idx = 0

#     for pos in range(EPVOU.shape[1]):
#         for h in range(2):
#             # Check if the counter exceeds the available subplots and create new figure if needed
#             if plot_idx >= n_rows * n_cols:
#                 fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
#                 ax = ax.flatten()
#                 plot_idx = 0

#             # Plot heatmap
#             cax = ax[plot_idx].imshow(EPVOU[h, pos].detach().cpu().numpy(), cmap='RdBu')

#             # Set title and axis labels if desired

#             ax[plot_idx].set_title(f"h{h} P{pos}: EVOU+PVOU ln_final_noscale")
#             ax[plot_idx].set_xlabel("output tok")
#             ax[plot_idx].set_ylabel("key tok")

#             # Optionally add a colorbar
#             plt.colorbar(cax, ax=ax[plot_idx])

#             # Increment the counter
#             plot_idx += 1

#             # You might want to save or display the plot here if you're iterating over many subplots
#             if plot_idx == 0:
#                 plt.tight_layout()
#                 plt.show()
#     # for pos in range(EPVOU.shape[1]):
#     #     for h in range(2):
#     #         imshow(EPVOU[h, pos], title=f"h{h} pos{pos} EPVOU")
#     #     break
#     # EPVOU_all = einsum()
# # for pos in range(dataset.seq_len-1):
# #     for h in range(2):
# #         EPVOU = ln_final_noscale(model.blocks[0].ln1(model.W_pos[qpos,:] + model.W_E) @ model.W_V[0,h,:,:] @ model.W_O[0,h,:,:]) @ model.W_U
# #         EPVOU = EPVOU.detach().cpu().numpy()


# # %%
# model.b_K

# # %%
