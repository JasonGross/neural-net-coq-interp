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
    simpler_train_losses = train_model(simpler_model, n_epochs=1500, batch_size=128, adjacent_fraction=True)


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


calculate_embed_and_pos_embed_overlap(simpler_model, renderer='png')



# In[ ]:


calculate_rowwise_embed_and_pos_embed_overlap(simpler_model, renderer='png')

# In[ ]:


calculate_OV_of_pos_embed(simpler_model)


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


    # stats = [summarize(centered[i], name=f'pos {pos} row {i}', linear_fit=True, renderer=renderer) for i in range(centered.shape[0])]

# In[ ]:


calculate_attn(simpler_model, renderer='png')

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

# %%

points = []
centered_score = calculate_attn_by_pos(simpler_model, renderer='png', pos=False)['value']
for row_n, row in enumerate(centered_score):
    for i in range(row.shape[0]):
        if i != row_n and abs(i - row_n) == 1:
            points.append((row[i].item() - row[row_n].item())  / (i - row_n))
# histogram
plt.hist(points, bins=100, edgecolor='black')
print(min(points))

# In[ ]:


simpler_model.W_Q.shape


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
    train_losses = train_model(model, n_epochs=500, batch_size=128)


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
    coq_export_params(model)


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


analyze_svd(model.W_U)

analyze_svd(model.blocks[0].ln1.w[:,None] * model.W_U)


# ## Attention Patterns
# 
# First, we visualize the attention patterns for a few inputs to see if this will give us an idea of what the model is doing.

# We begin by getting a batch of data and running a feedforward pass through the model, storing the resulting logits as well as the activations (in cache).

# In[ ]:

# Notebook runs fine until here, I won't bother fixing the rest --TK

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


# In[ ]:


# In[ ]:


plot_QK_cosine_similarity(model, querypos=1)


# In[ ]:
# Jason says the code below this is not useful


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