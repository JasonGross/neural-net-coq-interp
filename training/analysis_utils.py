# In[ ]:


# from training.Proving_How_A_Transformer_Takes_Max import linear_func


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import torch
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
import plotly.express as px

def linear_func(x, a, b):
    """Linear function: f(x) = a * x + b"""
    return a * x + b


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


def pm_range(values):
    return f"{(values.max().item() + values.min().item()) / 2.0} ± {(values.max().item() - values.min().item()) / 2.0}"


def pm_mean_std(values):
    return f"{values.mean().item()} ± {values.std().item()}"


def summarize(values, name=None, histogram=False, renderer=None, hist_args={}, include_value=False, linear_fit=False, min=True, max=True, mean=True, median=True, range=True, range_size=True, firstn=None, abs_max=True):
    if histogram:
        hist_args = dict(hist_args)
        if 'title' not in hist_args and name is not None: hist_args['title'] = f'Histogram of {name}'
        if 'renderer' not in hist_args and renderer is not None: hist_args['renderer'] = renderer
        if 'xaxis' not in hist_args: hist_args['xaxis'] = name if name is not None else 'Value'
        if 'yaxis' not in hist_args: hist_args['yaxis'] = 'Count'
        hist(values, **hist_args)

    if linear_fit:
        assert len(values.shape) in (1, 2)
        if len(values.shape) == 1:
            x_vals = np.arange(values.shape[0])
            y_vals = utils.to_numpy(values)
            aggregated = ''
        else:
            x_vals = np.tile(np.arange(values.shape[1]), values.shape[0])
            y_vals = utils.to_numpy(values.flatten())
            aggregated = 'Aggregated '
        name_space = '' if name is None else f'{name} '
        lin_title = f"{aggregated}{name_space}Data and Fit"
        resid_title = f"{aggregated}{name_space}Residual Errors"

        # Fit linear regression to the aggregated data
        popt, _ = curve_fit(linear_func, x_vals, y_vals)

        # Scatter plot the data & best fit line
        plt.figure()
        plt.scatter(x_vals, y_vals, label='Data', alpha=0.5, s=1)
        plt.plot(x_vals, linear_func(x_vals, *popt), 'r-', label=f'Fit: y = {popt[0]:.3f}*x + {popt[1]:.3f}')
        plt.title(lin_title)
        plt.legend()
        plt.show()

        # Plot residual errors
        residuals = y_vals - linear_func(x_vals, *popt)
        order_indices = np.argsort(x_vals)
        plt.figure()
        plt.scatter(x_vals[order_indices], residuals[order_indices], c='b', alpha=0.5)
        plt.title(resid_title)
        plt.show()

    res = {}
    if include_value: res['value'] = values.detach().clone()
    if min: res['min'] = values.min().item()
    if max: res['max'] = values.max().item()
    if mean: res['mean'] = pm_mean_std(values.float())
    if median: res['median'] = values.median().item()
    if range: res['range'] = pm_range(values)
    if range_size: res['range_size'] = values.max().item() - values.min().item()
    if firstn is not None: res[f'first {firstn}'] = values[:firstn]
    if abs_max: res['abs(max)'] = values.abs().max().item()
    if linear_fit: res['linear_fit_params'] = popt
    if linear_fit: res['linear_fit_equation'] = f'y = {popt[0]}*x + {popt[1]}'
    if linear_fit: res['range_residuals'] = pm_range(residuals)
    if linear_fit: res['residuals'] = residuals[order_indices]

    return res


#list(zip(all_integers[~correct_idxs], all_integers_ans[~correct_idxs]))


# # Simpler Model Interpretabiltiy

# ## Calculating how much slack we have

# Let's find out what the actual logits are, and how much slack we have on errors

# In[ ]:


def compute_slack(model: HookedTransformer, renderer=None):
    all_tokens = compute_all_tokens(model)
    pred_logits = model(all_tokens)[:,-1].detach()

    # Extract statistics for each row
    # Use values in all_tokens as indices to gather correct logits
    indices_max = all_tokens.max(dim=1).values.unsqueeze(1)
    indices_min = all_tokens.min(dim=1).values.unsqueeze(1)
    correct_logits = torch.gather(pred_logits, 1, indices_max).squeeze()
    incorrect_logits = torch.gather(pred_logits, 1, indices_min).squeeze()

    max_logits = pred_logits.max(dim=1).values
    min_logits = pred_logits.min(dim=1).values

    sorted_logits, _ = pred_logits.sort(dim=1, descending=True)
    diff_max_second = sorted_logits[:, 0] - sorted_logits[:, 1]
    diff_min_second = sorted_logits[:, -1] - sorted_logits[:, -2]
    diff_correct_incorrect = correct_logits - incorrect_logits

    # diagonal entries are zero, so remove them
    non_zero_diff_correct_incorrect = diff_correct_incorrect[diff_correct_incorrect != 0]

    # Sort diff_correct_incorrect and get sorted indices
    sorted_values, sorted_indices = torch.sort(diff_max_second)

    # Use the non-zero indices to get the sorted subtensors
    sorted_all_tokens = all_tokens[sorted_indices]
    sorted_diff_max_second = diff_max_second[sorted_indices]

    statistics = [
        ('Non-Zero Diff Correct-Incorrect', non_zero_diff_correct_incorrect),
        ('Correct Logit', correct_logits),
        ('Incorrect Logit', incorrect_logits),
        ('all tokens sorted by Diff Max-Second Largest', sorted_all_tokens),
        ('Diff Max-Second Largest sorted', sorted_diff_max_second),
        ('Max Logit', max_logits),
        ('Min Logit', min_logits),
        ('Diff Max-Second Largest', diff_max_second),
        ('Diff Min-Second Smallest', diff_min_second),
        ('Diff Correct-Incorrect', diff_correct_incorrect),
    ]

    # display a histogram of the logits
    plt.hist(pred_logits.numpy(), bins=100, edgecolor='black') # choose 100 bins for granularity
    plt.title('Histogram of Logits')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.show()

    # Plot histograms for each statistic
    # for stat_name, stat_values in statistics:
    #     plt.figure()
    #     plt.hist(stat_values.cpu().numpy(), bins=100, edgecolor='black')
    #     plt.title(f'Histogram of {stat_name}')
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    #     plt.show()

    # Return summary maps
    summary_maps = {}
    for stat_name, stat_values in statistics:
        summary = summarize(stat_values, name=stat_name, renderer=renderer, histogram=True, firstn=10)
        summary_maps[stat_name] = summary

    return summary_maps


# ## Negligibility of W_E @ W_U

# In[ ]:


def calculate_embed_overlap(model: HookedTransformer, renderer=None):
    W_U, W_E = model.W_U, model.W_E
    d_model, d_vocab = model.cfg.d_model, model.cfg.d_vocab
    assert W_U.shape == (d_model, d_vocab)
    assert W_E.shape == (d_vocab, d_model)
    res = (W_E @ W_U).detach()
    self_overlap = res.diag()
    imshow(res, renderer=renderer)
    line(self_overlap, renderer=renderer)
    statistics = [
        ('overlap', res),
        ('self-overlap', self_overlap),
        ('self-overlap after 0', self_overlap[1:]),
    ]
    return {name: summarize(value, name=name, include_value=True) for name, value in statistics}


# ## Negligibility of W_pos @ W_U

# In[ ]:


def calculate_pos_embed_overlap(model: HookedTransformer, renderer=None):
    W_U, W_pos = model.W_U, model.W_pos
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    res = (W_pos @ W_U).detach()
    imshow(res, renderer=renderer)

    statistics = [
        ('pos_embed_overlap', res),
        ('pos_embed_overlap (pos -1)', res[-1,:]),
    ]
    return {name: summarize(value, name=name, include_value=True, linear_fit=True, renderer=renderer) for name, value in statistics}


# ## Negligibility of (W_E + W_pos[-1]) @ W_U

# In[ ]:


def calculate_embed_and_pos_embed_overlap(model: HookedTransformer, renderer=None):
    W_U, W_E, W_pos = model.W_U, model.W_E, model.W_pos
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    res = ((W_E + W_pos[-1,:]) @ W_U).detach()
    self_overlap = res.diag()
    centered = res - self_overlap
    imshow(res, renderer=renderer)
    line(self_overlap, renderer=renderer)
    imshow(centered, renderer=renderer)
    imshow(centered[:,1:], renderer=renderer)
    statistics = [
        ('overlap (incl pos)', res),
        ('self-overlap (incl pos)', self_overlap),
        ('self-overlap after 0 (incl pos)', self_overlap[1:]),
        ('centered overlap (incl pos)', centered),
        ('centered overlap after 0 (incl pos)', centered[:,1:]),
    ]
    return {name: summarize(value, name=name, renderer=renderer) for name, value in statistics}


# ## Negligibility of W_pos @ W_V @ W_O @ W_U

# In[ ]:


def calculate_OV_of_pos_embed(model: HookedTransformer, renderer=None):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_pos @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    imshow(res, renderer=renderer)
    return summarize(res, name='W_pos @ W_V @ W_O @ W_U', renderer=renderer, linear_fit=True)