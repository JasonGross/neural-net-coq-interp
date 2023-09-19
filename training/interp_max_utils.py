# In[ ]:


# from training.Proving_How_A_Transformer_Takes_Max import linear_func


from typing import Optional
import einops
from fancy_einsum import einsum
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
import plotly.express as px
from analysis_utils import summarize
from training_utils import compute_all_tokens


# In[ ]:

def logit_delta(model: HookedTransformer, renderer=None, histogram_all_incorrect_logit_differences=False, return_summary=False):
    """
    Largest difference between logit(true_max) and logit(y) for y != true_max.
    Complexity: O(d_vocab^n_ctx * fwd_pass)
    fwd_pass = (n_ctx^2 * d_vocab * d_model^2) + (n_ctx * d_vocab * d_model^2)
    todo fix complexity.
    """

    all_tokens = compute_all_tokens(model=model)
    predicted_logits = model(all_tokens)[:,-1,:].detach().cpu()

    # Extract statistics for each row
    # Use values in all_tokens as indices to gather correct logits
    indices_of_max = all_tokens.max(dim=-1, keepdim=True).values
    correct_logits = torch.gather(predicted_logits, -1, indices_of_max)
    logits_above_correct = correct_logits - predicted_logits
    # replace correct logit indices with large number so that they don't get picked up by the min
    logits_above_correct[torch.arange(logits_above_correct.shape[0]), indices_of_max.squeeze()] = float('inf')
    min_incorrect_logit = logits_above_correct.min(dim=-1).values

    if histogram_all_incorrect_logit_differences:
        all_incorrect_logits = logits_above_correct[logits_above_correct != float('inf')]
        summarize(all_incorrect_logits, name='all incorrect logit differences', histogram=True, renderer=renderer)

    if return_summary:
        return summarize(min_incorrect_logit, name='min(correct logit - incorrect logit)', renderer=renderer, histogram=True)

    else:
        return min_incorrect_logit.min().item()
# In[ ]:
