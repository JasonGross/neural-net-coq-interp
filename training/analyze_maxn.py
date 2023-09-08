# %%
import torch
import torch.nn as nn
import numpy as np
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
import circuitsvis as cv
from fancy_einsum import einsum
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
    compute_slack, plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from training_utils import compute_all_tokens, get_data, make_generator_from_data

import os, sys
from importlib import reload


# %%

if __name__ == '__main__':
    PTH_BASE_PATH = Path(os.getcwd())
    PTH_BASE_PATH = PTH_BASE_PATH / 'transformer-takes-max'
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
    model = HookedTransformer(simpler_cfg, move_to_device=True)

    cached_data = torch.load(SIMPLER_MODEL_PTH_PATH)
    model.load_state_dict(cached_data['model'])
# %%

# Accuracy of model

dataset = large_data_gen(n_digits=64, sequence_length=N_CTX, batch_size=128, context="test", device=DEVICE)

# %%
# Test accuracy of model and get wrong examples
accs = []
for i in tqdm.tqdm(range(3000)):
    batch = dataset.__next__()
    logits = model(batch)
    acc_batch = acc_fn(logits, batch, return_per_token=True)
    acc = acc_batch.mean().item()
    if acc < 1:
        # print out wrong examples
        wrong_indices = torch.where(acc_batch == 0)[0]
        # print(f"Wrong indices: {wrong_indices}")
        last_logits = logits[:, -1, :]
        model_output = torch.argmax(last_logits, dim=1)
        correct_answers = torch.max(batch, dim=1)[0]
        # Model logit on correct answers
        correct_logits = last_logits[torch.arange(len(logits)), correct_answers]
        model_output_logits = last_logits[torch.arange(len(logits)), model_output]
        logit_diff = correct_logits - model_output_logits
        print(f"Logit diff: {logit_diff[wrong_indices].detach().cpu().numpy()}")
        print(f"Wrong examples: {batch[wrong_indices].cpu().numpy()}, {model_output[wrong_indices].cpu().numpy()}")
    accs.append(acc) 
print(f"Accuracy: {np.mean(accs)}")

# %%
# Interpreting attention scores

W_E, W_pos, W_Q, W_K = model.W_E, model.W_pos, model.W_Q, model.W_K
d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
assert W_E.shape == (d_vocab, d_model)
assert W_pos.shape == (n_ctx, d_model)
assert W_Q.shape == (1, 1, d_model, d_model)
assert W_K.shape == (1, 1, d_model, d_model)

points = torch.zeros((d_vocab, d_vocab, n_ctx))
# We have two cases, x in position 0 and x in position 1.
for pos_of_max in range(n_ctx):
    last_resid = (W_E + W_pos[-1]) # (d_vocab, d_model). Rows = possible residual streams.
    key_tok_resid = (W_E + W_pos[pos_of_max]) # (d_model, d_vocab). Rows = possible residual streams.
    q = last_resid @ W_Q[0, 0, :, :] # (d_vocab, d_model).
    k = key_tok_resid @ W_K[0, 0, :, :] # (d_model, d_vocab).
    x_scores = q @ k.T # (d_vocab, d_vocab).
    # x_scores[i, j] is the score from query token i to key token j.
    for i, row in enumerate(x_scores):
        for j in range(row.shape[0]):
            if i > j:
                attn_delta = (row[j].item() - row[i].item())
                if attn_delta > 0:
                    print(f"query={i}, key={j}, attn_delta={attn_delta:.2f}, pos_of_max={pos_of_max}")
                points[i, j, pos_of_max] = attn_delta
                # points.append((row[j].item() - row[i].item()) * (1 if i < j else -1))
result = points.min().item()
print(f"Score coefficient: {result:.2f}")
# %%
# plt.hist(points.flatten())
# %%
# 2d plot of x_scores
plt.imshow(x_scores.detach().cpu().numpy())
# Set axis labels
plt.title("Attention scores")
plt.xlabel("Key token")
plt.ylabel("Query token")

# %%
list(enumerate(model(torch.tensor([1, 1, 1, 18, 19]))[0, -1, :]))

# %%
calculate_copying(model)



# %%
calculate_rowwise_embed_and_pos_embed_overlap(model)

# %%
list(enumerate(model(torch.tensor([36, 35, 40, 37, 32]))[0, -1, :]))

# %%
list(enumerate(model(torch.tensor([37, 37, 40, 27, 32]))[0, -1, :]))

# %%
# Run the model on a single example using run_with_cache,
# and look at activations.

all_logits, cache = model.run_with_cache(torch.tensor([18, 1, 18, 1, 19]))
logits = all_logits[0, -1, :].detach().cpu().numpy()
print(f"{logits[18]=}, {logits[19]=}")

# %%
pattern = cache['attn_scores', 0].detach().cpu().numpy()[0, 0]
plt.imshow(pattern)
plt.xlabel("Query position")
plt.ylabel("Key position")
# Now label each cell with its value
for (j,i),label in np.ndenumerate(pattern):
    plt.text(i,j,f'{label:.3f}',ha='center',va='center')
# %%

last_resid = (W_E + W_pos[-1]) # (d_vocab, d_model). Rows = possible residual streams.
key_tok_resid = (W_E + W_pos[0]) # (d_model, d_vocab). Rows = possible residual streams.
q = last_resid @ W_Q[0, 0, :, :] # (d_vocab, d_model).
k = key_tok_resid @ W_K[0, 0, :, :] # (d_vocab, d_model).
x_scores = q @ k.T # (d_vocab, d_vocab).

scores = x_scores.detach().cpu().numpy()
print(f"{scores[19, 18]=}, {scores[19, 19]=}")
# %%
# There's some kind of mismatch between cached scores and the attention influences
# calculated above.

q_cached = cache['q', 0].detach().cpu().numpy()[0, :, 0, :]
q_cached.shape # (n_ctx, d_model)

k_cached = cache['k', 0].detach().cpu().numpy()[0, :, 0, :]
k_cached.shape # (n_ctx, d_model)

scores_cached = q_cached @ k_cached.T / np.sqrt(d_model)
# %%
plt.imshow(scores_cached[-1:, :])
for (j, i), label in np.ndenumerate(scores_cached[-1:, :]):
    plt.text(i, j, f'{label:.3f}', ha='center', va='center')
# %%
plt.imshow(pattern[-1:, :])
for (j, i), label in np.ndenumerate(pattern[-1:, :]):
    plt.text(i, j, f'{label:.3f}', ha='center', va='center')
# %%
