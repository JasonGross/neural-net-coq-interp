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
    compute_slack, plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
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
    print(key_tok_resid.shape)
    print(W_K.shape)
    k = einsum(key_tok_resid, W_K[0, 0, :, :], 'n_ctx d_vocab d_model, d_model d_model_k -> n_ctx d_model_k d_vocab')
    # k = key_tok_resid @ W_K[0, 0, :, :] # (n_ctx, d_model, d_vocab). 
    x_scores = einsum(q, k, 'd_vocab_q d_model, n_ctx d_model d_vocab_k -> n_ctx d_vocab_q d_vocab_k')
    print(k.T.shape)
    print(x_scores.shape)
    # x_scores[pos, qt, kt] is the score from query token qt to key token kt at position pos.
    # q_tok is always in the last position; k_tok can be anywhere before it.
    for q_tok in range(d_vocab):
        for k_tok in range(d_vocab):
            for pos_of_k in range(n_ctx - 1):
                if k_tok < q_tok:
                    attn_q_k = x_scores[pos_of_k, q_tok, k_tok].item() / np.sqrt(d_model)
                    attn_q_q = x_scores[-1, q_tok, q_tok].item() / np.sqrt(d_model)
                    attn_delta = (attn_q_k - attn_q_q)
                    points.append(attn_delta)
                    if attn_delta > 0:
                        print(f"query={q_tok}, key={k_tok}, attn_delta={attn_delta:.2f}, pos_of_k={pos_of_k}, attn_q_k={attn_q_k:.2f}, attn_q_q={attn_q_q:.2f}")
                    # TODO still need to account for y > x case
    # result = 0
    # print(f"Score coefficient: {result:.2f}")
    return min(points)

find_d_score_coeff(model)

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

all_logits, cache = model.run_with_cache(torch.tensor([23, 23, 23, 23, 25]))
logits = all_logits[0, -1, :].detach().cpu().numpy()
print(f"{logits[23]=}, {logits[25]=}")

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
print(f"{scores[25, 23]=}, {scores[25, 25]=}")
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
HookedTransformer