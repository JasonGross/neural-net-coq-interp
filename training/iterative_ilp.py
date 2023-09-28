# %%

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
from scipy.optimize import milp, LinearConstraint, Bounds
import math

from coq_export_utils import strify
from coq_export_utils import coq_export_params
from max_of_n import acc_fn, loss_fn, train_model, large_data_gen
from interp_max_utils import logit_delta, all_EVOU
from training_utils import compute_all_tokens, make_testset_trainset, make_generator_from_data

import os, sys
from importlib import reload

from train_max_of_n import get_model


# %%

if __name__ == '__main__':
    
    TRAIN_IF_NECESSARY = False
    model = get_model(train_if_necessary=TRAIN_IF_NECESSARY).to('cpu')
# %%
evou = all_EVOU(model).detach() # 64, 64

# %%


from scipy.special import binom

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


# %%
"""
n_ctx^2 problems of the form indexed by max m, output o
minimize sum_i (e_im - e_io) x_i subject to
0 <= x_i
sum x_i = n_ctx
for i > m, x_i = 0
for i == m, x_i >= 1

(more constraints from iteration)
"""


def get_excluded_inputs(mt, ot, n_ctx=5, mt_frac = 0.85, verbose=False):
    show = lambda *args: print(*args) if verbose else id
    c = evou[:, mt] - evou[:, ot] # negative = wrong answer
    A = np.ones((1, 64))
    mult = (n_ctx - 1) * mt_frac / (1 - mt_frac) # half of the attention is on true max
    c[mt] *= mult
    b_u = np.array([n_ctx])
    b_l = np.array([n_ctx])
    sum_constraint = LinearConstraint(A, b_l, b_u)

    l = np.zeros(64)
    l[mt] = 1 # at least one max token
    u = np.ones(64, dtype=int) * n_ctx
    for i in range(mt + 1, 64):
        u[i] = 0 # no tokens greater than max
    done = False
    steps=0
    while not done:
        steps += 1
        bounds = Bounds(l, u)
        integrality = np.array([True] * 64)
        res = milp(c=c, bounds=bounds, constraints=sum_constraint, integrality=integrality)
        if res.success == False or res.fun >= 0:
            done = True
        else:
            counts = {x:int(count) for x, count in enumerate(res.x) if count > 0}
            show(f"Optimal solution is {counts} with objective value {res.fun:.3f}")
            # reduce bound of most common value
            counts.pop(mt) # we don't want to remove the true max
            # remove the most common token, breaking ties by the one with the worst coefficient
            bound_to_reduce = max(counts, key=lambda x: (counts[x], -c[x]))
            show(f"Reducing bound of {bound_to_reduce} from {u[bound_to_reduce]} to {counts[bound_to_reduce] - 1}")
            u[bound_to_reduce] = counts[bound_to_reduce] - 1

    if res.fun is not None: show(f"Achieved min logit delta of {res.fun:.3f} in {steps} steps")
    show(f"Bounds: {u}")

    total_inputs = (mt + 1) ** n_ctx - mt ** n_ctx
    n_excluded_inputs = 0
    # count of x can be no greater than u[x], which excludes some inputs with a counting argument
    for kt in range(mt + 1):
        max_count = u[kt]
        # numbers that are not kt have mt choices, from 0 to mt excluding kt
        excluded_by_kt = 0
        for mtn in range(1, n_ctx):
            excluded_by_kt += sum(multinomial((count, mtn, n_ctx - count - mtn)) * mt ** (n_ctx - count - mtn) \
                                for count in range(max_count + 1, n_ctx - mtn + 1))
        n_excluded_inputs += excluded_by_kt

    n_excluded_inputs = min(n_excluded_inputs, total_inputs)

    show(f"Excluded {n_excluded_inputs:.3E} out of {total_inputs:.3E} inputs ({n_excluded_inputs / total_inputs * 100:.1f}%))")
    return n_excluded_inputs, total_inputs

# get_excluded_inputs(0, 19, verbose=True)
# %%
# [(k, get_excluded_inputs(63, k, verbose=True)) for k in range(64)]
# %%
# %%

def calculate_accuracy(d_vocab):
    n_ctx = 5
    excluded_inputs = 0
    for mt in range(d_vocab):
        n_excluded_inputs_n = 0
        for ot in range(d_vocab):
            if mt == ot:
                continue
            n_excluded_inputs_ot, total_inputs = get_excluded_inputs(mt, ot)
            n_excluded_inputs_n += n_excluded_inputs_ot
        print(f"Accuracy for {mt}: {1 - n_excluded_inputs_n / total_inputs:.3f}")
        n_excluded_inputs_n = min(n_excluded_inputs_n, total_inputs)
        excluded_inputs += n_excluded_inputs_n
    overall_accuracy = 1 - excluded_inputs / (d_vocab ** n_ctx)
    print(f"Overall accuracy: {overall_accuracy:.3f}")


calculate_accuracy(64)
# %%
