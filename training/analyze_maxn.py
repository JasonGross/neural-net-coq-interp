# %%
import torch
import torch.nn as nn
import numpy as np
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
import circuitsvis as cv
from einops import einsum, repeat
from pathlib import Path
from IPython import get_ipython
import matplotlib.pyplot as plt
import seaborn as sns
import time

from coq_export_utils import strify
from analysis_utils import line, summarize, plot_QK_cosine_similarity, \
    analyze_svd, calculate_OV_of_pos_embed, calculate_attn, calculate_attn_by_pos, \
    calculate_copying, calculate_copying_with_pos, calculate_embed_and_pos_embed_overlap, \
    calculate_rowwise_embed_and_pos_embed_overlap, \
    calculate_embed_overlap, calculate_pos_embed_overlap, check_monotonicity, \
    plot_avg_qk_heatmap, plot_qk_heatmap, plot_qk_heatmaps_normed, plot_unembed_cosine_similarity
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from interp_max_utils import logit_delta, all_EVOU, all_PVOU
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
# calculate_copying(model)



# %%
calculate_rowwise_embed_and_pos_embed_overlap(model)


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
def find_d_EVOU_PVOU_max(model) -> float:
    """
    When x is maximum, the logit effect of copying the correct residual stream.

    Complexity: O(d_vocab * d_model^2 + d_vocab^2 * d_model + ...)
    Return shape: (mt, ot)
    reducing over mp
    """
    W_E, W_pos, W_V, W_O, W_U = model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    assert W_U.shape == (d_model, d_vocab)

    EVOU = all_EVOU(model) # (d_vocab, d_vocab). EVOU[i, j] is how copying i affects j.
    PVOU = all_PVOU(model) # (n_ctx, d_vocab)

    # Values of (effect on x - effect on y) where y != x. (could do y < x)
    EVOU_without_diag = EVOU - EVOU.diag().diag() * EVOU.max()
    EVOU_effect = (EVOU.diag()[:, None] - EVOU) # mt, ot

    EVOU_PVOU = EVOU_effect
    for mt in range(d_vocab):
        for ot in range(d_vocab):
            EVOU_PVOU[mt, ot] += (PVOU[:, mt] - PVOU[:, ot]).min()

    # To improve this bound we take into account x-dependence of EVOU and PVOU.
    # return result
    return EVOU_PVOU.rename('mt', 'ot')

if __name__ == '__main__':
    print(find_d_EVOU_PVOU_max(model))

# %%
def effect_of_EU_PU(model) -> torch.Tensor:
    """
    Calculate the maximum negative effect of the EU and PU paths on the output.
    Complexity: O(d_vocab^2 * n_ctx * d_model)
    Return shape: (qt, mt, ot)
    """
    W_E, W_pos, W_U = model.W_E, model.W_pos, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_U.shape == (d_model, d_vocab)

    # The logit effect of token x and position p is given by the vector:
    #   logits(x, p) = (W_E[x] + W_pos[p]) @ W_U
    max_logit_deltas = torch.zeros((d_vocab, d_vocab, d_vocab))
    for qt in range(d_vocab): # query token
        # for p in range(n_ctx):
        logit_deltas = (W_E[qt] + W_pos[-1]) @ W_U # (d_vocab,)
        max_logit_deltas[qt] = logit_deltas[:, None] - logit_deltas[None, :]

    result = max_logit_deltas # (q_token, max_token, other_token)
    print(f"EU and PU paths have min effect of {result.min():.2f}")
    return result.detach().rename('qt', 'mt', 'ot')

if __name__ == '__main__':
    eu_pu = effect_of_EU_PU(model)

# %%
def find_d_EVOU_PVOU_nonmax(model, out_shape=('kt','mt','ot')) -> torch.Tensor:
    """
    When x is maximum, the minimum logit effect of copying the incorrect residual stream.
    Basically the max amount that copying y increases z more than x where z < x and y < x.
    Return shape: (mt, ot) or (kt, mt, ot)
    """
    assert out_shape in [('kt','mt','ot'), ('mt', 'ot')]
    W_E, W_pos, W_V, W_O, W_U = model.W_E, model.W_pos, model.W_V, model.W_O, model.W_U
    d_model, n_ctx, d_vocab = model.cfg.d_model, model.cfg.n_ctx, model.cfg.d_vocab
    assert W_E.shape == (d_vocab, d_model) and W_pos.shape == (n_ctx, d_model) and W_V.shape == (1, 1, d_model, d_model) and W_O.shape == (1, 1, d_model, d_model) and W_U.shape == (d_model, d_vocab)

    # is this right or do we need to transpose W_O and W_U?
    EVOU = all_EVOU(model) # (d_vocab, d_vocab) = (kt, ot). EVOU[i, j] is how copying i affects j.
    PVOU = all_PVOU(model) # (n_ctx, d_vocab) = (kp, ot)

    # Our reasoning is simpler than for find_d_EVOU_PVOUx: just the largest logit delta from each key token
    dEVOUs = (EVOU[:, None, :] - EVOU[:, :, None]) # (kt, mt, ot)
    if out_shape == ('mt', 'ot'): dEVOUs = dEVOUs.min(dim=0).values # (mt, ot)

    # Worst case over all positions of (effect on x - effect on y) where y <= x.
    PVOU_cummax_reverse = PVOU.flip(dims=(1,)).cummax(dim=1).values.flip(dims=(1,)) # kp, ot
    min_PVOU_effect = (PVOU - PVOU_cummax_reverse).min(dim=0).values # (ot,)

    result = (dEVOUs + (min_PVOU_effect[None, None, :] if 'kt' in out_shape else min_PVOU_effect[None, :])) # (mt, ot)
    return result.rename(*out_shape)

if __name__ == '__main__':
    wrong_attn_dEPVOU = find_d_EVOU_PVOU_nonmax(model)
    print(wrong_attn_dEPVOU)

# %%
def required_attn_frac(model, out_shape=('qt','kt','mt')):
    """
    Returns the minimum percentage of attention that must be paid to the correct token mt
      to prove that mt has a higher logit than some other token ot.
    Return shape: qt, kt, mt
    """
    assert out_shape in [('qt','kt','mt'), ('qt', 'mt')]
    # out_shape = ('qt', 'kt', 'mt')
    d_vocab = model.cfg.d_vocab

    global dEU_PU_above_diag
    dEU_PU = effect_of_EU_PU(model)
    # Minimum of EU+PU effect values where qt <= mt
    dEU_PU_above_diag = dEU_PU.clone() # (qt, mt, ot)
    print(dEU_PU_above_diag.names)
    tril_index = torch.tril_indices(d_vocab, d_vocab, offset=-1)
    dEU_PU_above_diag.rename_(None)
    dEU_PU_above_diag[tril_index[0], tril_index[1], :] = dEU_PU.max() # (qt, mt, ot)
    if 'kt' in out_shape:
        dEU_PU_above_diag = dEU_PU_above_diag.min(dim=2).values # (qt, mt)
        dEU_PU_above_diag = dEU_PU_above_diag.unsqueeze(1).rename('qt', 'kt', 'mt')

    correct_attn_dEPVOU = find_d_EVOU_PVOU_max(model) # (mt, ot)
    # Ignore diagonal; we don't care what happens when mt = ot
    correct_attn_dEPVOU = correct_attn_dEPVOU + correct_attn_dEPVOU.max() * torch.eye(d_vocab) # (mt, ot)

    wrong_attn_dEPVOU = find_d_EVOU_PVOU_nonmax(model, out_shape=('kt', 'mt','ot') if 'kt' in out_shape else ('mt', 'ot'))
    # dEPVOU_delta = (correct_attn_dEPVOU - wrong_attn_dEPVOU).min(dim='ot').values # (kt, mt)

    if 'kt' in out_shape:
        pass
        global wrong_attn_dEPVOU_min
        correct_attn_dEPVOU = correct_attn_dEPVOU.min(dim='ot').values # (mt,)
        wrong_attn_dEPVOU_min = wrong_attn_dEPVOU.min(dim='ot').values # (kt, mt)
        wrong_attn_dEPVOU_max = wrong_attn_dEPVOU.max(dim='ot').values # (kt, mt)

        # where this is true, we get a maximum attention %
        # global wrong_better_than_correct_mask 
        # wrong_better_than_correct_mask = (correct_attn_dEPVOU - wrong_attn_dEPVOU < 0) # (kt, mt, ot)
        # print(wrong_better_than_correct_mask.names)

    # Output logit diff = EUPU effect + x * correct copying EPVOU + (1-x) * incorrect copying EPVOU > 0
    # x (correct EPVOU - incorrect EPVOU) > -(incorrect EPVOU + EUPU)
    # We can divide by correct EPVOU - incorrect EPVOU because it's always positive; correct attention is better than the worst incorrect attention.
    if 'kt' not in out_shape: assert (correct_attn_dEPVOU - wrong_attn_dEPVOU).min() > 0
    # sns.histplot((correct_attn_dEPVOU - wrong_attn_dEPVOU).rename(None).flatten())
    # sns.histplot((wrong_attn_dEPVOU + dEU_PU_above_diag).rename(None).flatten())
    # numerator is how much "slack" to overcome, denominator is goodness of correct attention
    always_good_mask = (wrong_attn_dEPVOU_min + dEU_PU_above_diag > 0).rename(None) & (correct_attn_dEPVOU + dEU_PU_above_diag > 0).rename(None) # (qt, kt, mt)
    always_bad_mask = (wrong_attn_dEPVOU_max + dEU_PU_above_diag < 0).rename(None) & (correct_attn_dEPVOU + dEU_PU_above_diag < 0).rename(None) # (qt, kt, mt)
    raf = - (wrong_attn_dEPVOU_min + dEU_PU_above_diag) / (correct_attn_dEPVOU - wrong_attn_dEPVOU_min) # (qt, kt, mt)
    raf.rename(None)[always_good_mask] = -1
    raf.rename(None)[always_bad_mask] = 2
    if 'ot' in raf.names: raf = raf.max(dim='ot').values
    return raf
raf = required_attn_frac(model)
print(f"{raf.min():.4f}, {raf.median():.4f}, {raf.max():.4f}")
# %%
sns.ecdfplot(raf.rename(None).flatten(), color='blue')
raf2 = required_attn_frac(model, out_shape=('qt', 'mt'))
sns.ecdfplot(raf2.rename(None).flatten(), color='orange')

# %%

def accuracy_bound_prep(model):
    """
    required_attn_frac(model) gives a min attention % paid to the correct token,
      for each qt, mt, ot
    If we take the worst case (max) over ot, then we can prove the model correct
    for some fraction of qt, mt pairs (where there is no kt such that the model
    pays too much attention to kt and not enough to mt).
    Return shape: qt, mt
    """
    d_vocab = model.cfg.d_vocab
    W_E, W_pos, W_Q, W_K = model.W_E, model.W_pos, model.W_Q, model.W_K

    raf = required_attn_frac(model) # (qt, kt, mt) or (qt, mt, ot)
    # raf_reduced = raf.max(dim=2).values.detach().cpu().numpy() # (qt, mt)
    # Now find the worst tokens kt for qt to attend to, and put mt in the worst position with kt everywhere else.
    # Use run_with_cache to get attention score on mt then compare...
    # This is O(d_vocab^2 log d_vocab), choosing qt, mt, kt.
    # But since we don't binary search for kt this implementation is O(d_vocab^3).
    
    EQKP = W_E @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ W_pos.T # (qt, n_ctx)
    worst_positions_by_qt = EQKP.argmin(dim=1)

    n_ok_kts = torch.zeros((d_vocab, d_vocab), dtype=torch.long)
    for qt in range(d_vocab):
        worst_position = worst_positions_by_qt[qt]
        for mt in range(d_vocab):
            input = torch.arange(mt, dtype=torch.long) # set kt
            input = repeat(input, 'kt -> kt ctx', ctx = model.cfg.n_ctx).clone()
            input[:, worst_position] = mt
            input[:, -1] = qt
            # print(input)
            logits, cache = model.run_with_cache(input)
            pattern = cache['pattern', 0].detach().cpu().numpy()[:, 0, -1, :]
            attn_to_mt = pattern[:, worst_position]

            # when qt==mt, we should count attention to qt
            good_attn = attn_to_mt + pattern[:, -1] if (qt == mt and worst_position != -1) else attn_to_mt
            n_ok_kts_qt_mt = (torch.tensor(good_attn) > raf[qt, :mt, mt]).sum()
            n_ok_kts[qt, mt] = n_ok_kts_qt_mt

    return n_ok_kts


# start = time.time()
# result = accuracy_bound_prep(model)
# elapsed = time.time() - start
# print(result)
# print(f"Elapsed: {elapsed:.2f} seconds")

# %%

def accuracy_bound(model):
    d_vocab, n_ctx = model.cfg.d_vocab, model.cfg.n_ctx
    ok_kts_arr = accuracy_bound_prep(model) # (qt, mt)

    # When qt = mt, there are ok_kts^(n_ctx - 1) choices of input
    # When qt != mt, there are ok_kts^(n_ctx - 2) * (n_ctx - 1) choices of input
    total = 0
    for qt in range(d_vocab):
        for mt in range(d_vocab):
            if qt == mt:
                total += ok_kts_arr[qt, mt] ** (n_ctx - 1)
            else:
                # TODO fix for multiplicity of mt
                total += ok_kts_arr[qt, mt] ** (n_ctx - 2) * (n_ctx - 1)
    
    return total / (d_vocab ** n_ctx)

accuracy = accuracy_bound(model)
print(f"Accuracy bound: {accuracy*100:.4f}%")



# %%
# def compute_attention_slack(model: HookedTransformer):
max_copying = torch.zeros_like(correct_copying_effect)
attention_slack = torch.zeros_like(correct_copying_effect)
for mt in range(d_vocab):
    for ot in range(d_vocab):
        if mt == ot: continue
        max_copying[mt, ot] = dEVOU_PVOU[ot, mt] # how much more ot copies itself than mt
        # TODO: we can do a more refined computation of scaling how much various tokens copy ot by the actual attention paid to them
        # solve for x: x * result[mt, ot] - (1-x) * dEVOU_PVOU[ot, mt] = 0
        # x * result[mt, ot] = (1-x) * dEVOU_PVOU[ot, mt]
        # x * result[mt, ot] = dEVOU_PVOU[ot, mt] - x * dEVOU_PVOU[ot, mt]
        # x * result[mt, ot] + x * dEVOU_PVOU[ot, mt] = dEVOU_PVOU[ot, mt]
        # x * (result[mt, ot] + dEVOU_PVOU[ot, mt]) = dEVOU_PVOU[ot, mt]
        # x = dEVOU_PVOU[ot, mt] / (result[mt, ot] + dEVOU_PVOU[ot, mt])
        # x = e^attn_good / (e^attn_good + e^attn_bad) = e^attn_bad * e^(attn_good - attn_bad) / (e^attn_bad * (e^(attn_good - attn_bad) + 1)) = e^(attn_good - attn_bad) / (1 + e^(attn_good - attn_bad))
        # solve for attn_good - attn_bad
        # 1 - x = 1 / (1 + e^(attn_good - attn_bad))
        # 1 / (1 - x) - 1 = e^(attn_good - attn_bad)
        # log(1 / (1 - x) - 1) = attn_good - attn_bad
        attention_slack[mt, ot] = (1 / (1 - dEVOU_PVOU[ot, mt] / (correct_copying_effect[mt, ot] + dEVOU_PVOU[ot, mt])) - 1).log()
print(attention_slack)

# print(compute_attention_slack(model))
#%%

a = torch.arange(36).reshape((3, 3, 4))
print(a)
tril_index = torch.tril_indices(3, 3, offset=-1)
print(tril_index)
print(f"{a[tril_index[0], tril_index[1]]=}")
a[tril_index[0], tril_index[1], :] = 5
print(a)

# %%

# def min_result_of_attn_to_max(model):
dEVOU_PVOU = find_d_EVOU_PVOU_max(model)
dEVOU_PVOU_without_diag = dEVOU_PVOU + dEVOU_PVOU.max() * torch.eye(model.cfg.d_vocab)
sns.histplot(dEVOU_PVOU_without_diag.flatten().detach().cpu().numpy())
dEU_PU = effect_of_EU_PU(model)
dEU_PU_above_diag = dEU_PU.clone()
dEU_PU_above_diag[torch.tril_indices(model.cfg.d_vocab, model.cfg.d_vocab, offset=-1), :] = 0
min_eu_pu = dEU_PU_above_diag.min(dim=0).values
result = (min_eu_pu + dEVOU_PVOU_without_diag) # (mt, ot)
sns.histplot(result.flatten().detach().cpu().numpy())
print(result.min())
# get 2-D index of min
v, r = result.min(dim=0)
v, c = v.min(dim=0)
print((r[c], c))
# indices of negative values
print((result < 0).nonzero())
# return result

# min_result_of_attn_to_max(model)





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

    d_EU_PU = effect_of_EU_PU(model)
    d_score_coeff = find_d_score_coeff(model)
    d_EOVU_POVUx = find_d_EVOU_PVOU_max(model)
    d_EOVU_POVUy_c1, d_EOVU_POVUy_c2 = find_d_EVOU_PVOU_nonmax(model)

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

# Get EVOU matrix (kt, ot)
evou = all_EVOU(model).detach()
sns.heatmap(evou)

def kt_attn_reqd_to_flip(model):
    evou = all_EVOU(model).detach()
    d_vocab = evou.shape[0]
    result = torch.zeros((d_vocab, d_vocab, d_vocab))
    for mt in range(d_vocab):
        for kt in range(d_vocab):
            if kt == mt: continue
            for ot in range(d_vocab):
                mt_ot_delta = evou[mt, mt] - evou[mt, ot]
                kt_ot_delta = evou[kt, mt] - evou[kt, ot]
                result[mt, kt, ot] = kt_ot_delta / mt_ot_delta
    return result
# %%

kt_numbers = kt_attn_reqd_to_flip(model)
# %%

sns.heatmap(kt_numbers[25, :, :])
plt.title("Is it easy to flip 25?")
plt.ylabel("kt")
plt.xlabel("ot")
# %%
