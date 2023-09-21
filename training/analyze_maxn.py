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

from train_max_of_n import get_model


# %%

if __name__ == '__main__':
    
    TRAIN_IF_NECESSARY = False
    model = get_model(train_if_necessary=TRAIN_IF_NECESSARY).to('cpu')
# %%


# %%

# Accuracy of model

# dataset = large_data_gen(n_digits=64, sequence_length=N_CTX, batch_size=128, context="test", device=DEVICE)

# %%
# # Test accuracy of model and get wrong examples
# accs = []
# for i in tqdm.tqdm(range(3000)):
#     batch = dataset.__next__()
#     logits = model(batch)
#     acc_batch = acc_fn(logits, batch, return_per_token=True)
#     acc = acc_batch.mean().item()
#     if acc < 1:
#         # print out wrong examples
#         wrong_indices = torch.where(acc_batch == 0)[0]
#         # print(f"Wrong indices: {wrong_indices}")
#         last_logits = logits[:, -1, :]
#         model_output = torch.argmax(last_logits, dim=1)
#         correct_answers = torch.max(batch, dim=1)[0]
#         # Model logit on correct answers
#         correct_logits = last_logits[torch.arange(len(logits)), correct_answers]
#         model_output_logits = last_logits[torch.arange(len(logits)), model_output]
#         logit_diff = correct_logits - model_output_logits
#         print(f"Logit diff: {logit_diff[wrong_indices].detach().cpu().numpy()}")
#         print(f"Wrong examples: {batch[wrong_indices].cpu().numpy()}, {model_output[wrong_indices].cpu().numpy()}")
#     accs.append(acc) 
# print(f"Accuracy: {np.mean(accs)}")

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
#plt.imshow(x_scores.detach().cpu().numpy())
# Set axis labels
#plt.title("Attention scores")
#plt.xlabel("Key token")
#plt.ylabel("Query token")

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

#last_resid = (W_E + W_pos[-1]) # (d_vocab, d_model). Rows = possible residual streams.
#key_tok_resid = (W_E + W_pos[0]) # (d_model, d_vocab). Rows = possible residual streams.
#q = last_resid @ W_Q[0, 0, :, :] # (d_vocab, d_model).
#k = key_tok_resid @ W_K[0, 0, :, :] # (d_vocab, d_model).
#x_scores = q @ k.T # (d_vocab, d_vocab).

#scores = x_scores.detach().cpu().numpy()
#print(f"{scores[25, 23]=}, {scores[25, 25]=}")
# %%
# There's some kind of mismatch between cached scores and the attention influences
# calculated above.

q_cached = cache['q', 0].detach().cpu().numpy()[0, :, 0, :]
q_cached.shape # (n_ctx, d_model)

k_cached = cache['k', 0].detach().cpu().numpy()[0, :, 0, :]
k_cached.shape # (n_ctx, d_model)

scores_cached = q_cached @ k_cached.T / np.sqrt(model.cfg.d_model)
# %%
plt.imshow(scores_cached[-1:, :])
for (j, i), label in np.ndenumerate(scores_cached[-1:, :]):
    plt.text(i, j, f'{label:.3f}', ha='center', va='center')
# %%
plt.imshow(pattern[-1:, :])
for (j, i), label in np.ndenumerate(pattern[-1:, :]):
    plt.text(i, j, f'{label:.3f}', ha='center', va='center')
# %%

"""
O(n^3) proof for max of n

Case 1:
    All numbers other than max are at most max - gap, so we only care about OV on true max
    - Bound EU and PU effects
    - Bound attention on non-max tokens
    - Bound logit effect of attending to non-max tokens

Case 2a:
    Some numbers i st max - gap < i <= max, and query is max
    - For every query token qt:
        - For each i, get the max attention paid to i and logit effect of attending to i

Case 2b:
    Some numbers i st max - gap < i <= max, and query is not max
    - Get max positional effect on attn for every query token (qt,)
    - For every i, j, qt < j, get max wrong-direction attn effect of query token (qt, i, j)
    - Combine above two steps to get max wrong attention to i when max is j
    - OV analysis to get badness of max wrong attention to i when max is j
    - TODO Convexity argument says worst case is when every non-max token is equal,
      so we can just look at worst i for every j

TODO are there any tokens i that copy to j>i more than i? If so we need to worry about duplicates of max
"""

"""
O(n^2 * gap^(n_ctx - 2) * inference) proof for max of n

Cases 1, 2a from above

Case 2b:
    Run model on all n^2 * (gap+1)^(n_ctx - 2) sequences where we have some
    numbers i st max - gap < i <= max, and query is not max, treating all non-query numbers less than
    max - gap as the worst case.
    e.g. we have "17, 17, small, 18, 10"
"""
 # %%
def find_d_EVOU_PVOUx(model) -> float:
    """
    When x is maximum, the minimum logit effect of copying the correct residual stream.

    Complexity: O(d_vocab * d_model^2 + d_vocab^2 * d_model + ...)
    """
    W_E, W_pos, W_V, W_O, W_U = model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    assert W_U.shape == (d_model, d_vocab)

    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (n_ctx, d_vocab)

    # Worst case over all x of (effect on x - effect on y) where y != x. (could do y < x)
    EVOU_without_diag = EVOU - EVOU.diag().diag() * EVOU.max()
    min_EVOU_effect = (EVOU.diag() - EVOU_without_diag.max(dim=1).values)

    # Worst case over all positions of (effect on x - effect on y) where y <= x.
    PVOU_cummax = PVOU.cummax(dim=1).values # (n_ctx, d_vocab)
    min_PVOU_effect = (PVOU - PVOU_cummax).min(dim=0).values # (d_vocab,)

    # To improve this bound we take into account x-dependence of EVOU and PVOU.
    result = (min_EVOU_effect + min_PVOU_effect).min()
    print(f"Correct copying effect from:")
    print(f"EVOU: {min_EVOU_effect.min().item():.2f}, PVOU: {min_PVOU_effect.min().item():.2f}")
    print(f"Total: {result.item():.2f}")
    return result

if __name__ == '__main__':
    find_d_EVOU_PVOUx(model)
    
# %%
def min_effect_of_EU_PU(model) -> torch.Tensor:
    """
    Calculate the maximum negative effect of the EU and PU paths on the output.
    Complexity: O(d_vocab^2 * n_ctx * d_model)
    Return shape: (q_token,)
    """
    W_E, W_pos, W_U = model.W_E, model.W_pos, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_U.shape == (d_model, d_vocab)

    # The logit effect of token x and position p is given by the vector:
    #   logits(x, p) = (W_E[x] + W_pos[p]) @ W_U
    max_logit_deltas = torch.zeros((d_vocab, n_ctx))
    for x in range(d_vocab): # query token
        for p in range(n_ctx):
            logit_deltas = (W_E[x] + W_pos[p]) @ W_U # (d_vocab,)
            max_logit_deltas[x, p] = logit_deltas.max() - logit_deltas.min()

    result = -max_logit_deltas.max(dim=1).values # (q_token,)
    print(f"EU and PU paths have min effect of {result.min():.2f}")
    return result

if __name__ == '__main__':
    eu_pu = min_effect_of_EU_PU(model)
    
# %%
def find_d_EVOU_PVOUy(model) -> float:
    """
    When x is maximum, the minimum logit effect of copying the incorrect residual stream.
    Basically the max amount that copying y increases z more than x where z < x and y < x.
    """
    W_E, W_pos, W_V, W_O, W_U = model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    assert W_U.shape == (d_model, d_vocab)

    EVOU = W_E @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    EVOU.names = ('qtok', 'ktok')
    PVOU = W_pos @ W_V[0, 0, :, :] @ W_O[0, 0, :, :] @ W_U # (n_ctx, d_vocab)

    # Our reasoning is simpler than for find_d_EVOU_PVOUx: just the largest logit delta from each query token
    EVOU_neg_range = -EVOU.max(dim='ktok').values + EVOU.min(dim='ktok').values # (d_vocab,) for each query token
    # Case 1: y = x - 1 e.g. 37, 38. We want EVOU[37, 38] - max_j EVOU[37, j].
    # EVOU_delta_case_1 = torch.diff(EVOU, dim=1).min(dim='ktok').values # (d_vocab,)
    EVOU_y_yp1 = torch.cat((EVOU.rename(None).diag(1), torch.tensor((1000,)))) # (d_vocab,)
    EVOU_y_smaller = EVOU.cummax(dim='ktok').values.diagonal() # (d_vocab,)
    EVOU_delta_case_1 = (EVOU_y_yp1[:, None] - EVOU_y_smaller) # (d_vocab, d_vocab)

    # Worst case over all positions of (effect on x - effect on y) where y <= x.
    PVOU_cummax_reverse = PVOU.flip(dims=(1,)).cummax(dim=1).values.flip(dims=(1,))
    min_PVOU_effect_case_2 = (PVOU - PVOU_cummax_reverse).min(dim=0).values # (d_vocab,): qtok
    

    result_case_1 = (EVOU_delta_case_1 + min_PVOU_effect_case_2).min()
    result_case_2 = (EVOU_neg_range + min_PVOU_effect_case_2).min()
    print(f"Incorrect copying effect:")
    print(f"Case 1: {result_case_1.item():.2f}, Case 2: {result_case_2.item():.2f}")
    # result_case_1, result_case_2
    return result_case_1, result_case_2

if __name__ == '__main__':
    find_d_EVOU_PVOUy(model)
    
# %%    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def slack(model, biggap=None, smallgap=None):
    """
    Compute the minimum value of logit(x)-logit(y) when x > y.
    If this is >0, the model gets 100% accuracy.
    """
    if biggap is None: 
        for biggap in range(1, model.cfg.d_vocab):
            print(f"Big Gap {biggap}:")
            gaps, currslack = slack(model, biggap = biggap, smallgap = smallgap)
            if currslack > 0: 
                return gaps, currslack
        return (None, smallgap), float('-inf')
    
    if smallgap is None:
        for smallgap in range(1, biggap):
            print(f"Small Gap {smallgap}:")
            gaps, currslack = slack(model, biggap = biggap, smallgap = smallgap)
            if currslack > 0: 
                return gaps, currslack
        return (biggap, None), float('-inf')
            
    d_EU_PU = min_effect_of_EU_PU(model)
    d_score_coeff = find_d_score_coeff(model)
    d_EOVU_POVUx = find_d_EVOU_PVOUx(model)
    d_EOVU_POVUy_c1, d_EOVU_POVUy_c2 = find_d_EVOU_PVOUy(model)

    # d_attn_out_U_case_1 = sigmoid(d_score_coeff) * d_EOVU_POVUx + (1 - sigmoid(d_score_coeff)) * d_EOVU_POVUy_c1
    # d_attn_out_U_case_2 = sigmoid(d_score_coeff * 2) * d_EOVU_POVUx + (1 - sigmoid(d_score_coeff * 2)) * d_EOVU_POVUy_c2
    gap_worst_attn_scores = torch.zeros(model.cfg.d_vocab, model.cfg.n_ctx)
    gap_worst_attn_scores[:, 1] = -d_score_coeff * smallgap# column 0 = attention paid to true max
    gap_worst_attn_scores[:, 2:] = -d_score_coeff * biggap
    gap_worst_attn_pattern = torch.softmax(gap_worst_attn_scores, dim=1)[:, 0] # (d_vocab,) because only first column matters.
    d_attn_out_U_case_2 = gap_worst_attn_pattern * d_EOVU_POVUx + (1 - gap_worst_attn_pattern) * d_EOVU_POVUy_c2

    result = (d_EU_PU + d_attn_out_U_case_2).min().item() # min over query token
    #min_logit_diff = logit_diff_on_gap_1_cases(model).min().item()
    #print(f"Min logit diff on gap 1: {min_logit_diff:.2f}")
    #result = min(result, min_logit_diff)
    print(f"Model slack for case 1 on small gap {smallgap}, big gap {biggap}: {result:.2f}")
    #print(f"Model {'is' if result > 0 else 'is not'} proven 100% accurate.")
    return (biggap, smallgap), result
    
if __name__ == '__main__':
    slack(model)
# %%
