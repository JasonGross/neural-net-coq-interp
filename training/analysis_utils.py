# In[ ]:


# from training.Proving_How_A_Transformer_Takes_Max import linear_func


from typing import Callable, Iterable, List, Optional, Tuple
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
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from inspect import signature
import itertools

# %%

def linear_func(x, a, b):
    """Linear function: f(x) = a * x + b"""
    return a * x + b
linear_func.equation = lambda popt: f'y = {popt[0]:.3f}*x + {popt[1]:.3f}'

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c
quadratic_func.equation = lambda popt: f'y = {popt[0]:.3e}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}'

def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d
cubic_func.equation = lambda popt: f'y = {popt[0]:.3e}*x^3 + {popt[1]:.3e}*x^2 + {popt[2]:.3f}*x + {popt[3]:.3f}'

def quartic_func(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e
quartic_func.equation = lambda popt: f'y = {popt[0]:.3e}*x^4 + {popt[1]:.3e}*x^3 + {popt[2]:.3e}*x^2 + {popt[3]:.3f}*x + {popt[4]:.3f}'

def quintic_func(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
quintic_func.equation = lambda popt: f'y = {popt[0]:.3e}*x^5 + {popt[1]:.3e}*x^4 + {popt[2]:.3e}*x^3 + {popt[3]:.3e}*x^2 + {popt[4]:.3f}*x + {popt[5]:.3f}'

def absolute_shift_func(x, a, b, c):
    return a * np.abs(x - b) + c
absolute_shift_func.equation = lambda popt: f'y = {popt[0]:.3f}*|x - {popt[1]:.3f}| + {popt[2]:.3f}'

def linear_sinusoid_func(x, a, b, c, d):
    return (a * x + b) * np.sin(c * x + d)
linear_sinusoid_func.equation = lambda popt: f'y = ({popt[0]:.3f}*x + {popt[1]:.3f}) * sin({popt[2]:.3f}*x + {popt[3]:.3f})'

def quadratic_sinusoid_func(x, a, b, c, d, e):
    return (a * x**2 + b * x + c) * np.sin(d * x + e)
quadratic_sinusoid_func.equation = lambda popt: f'y = ({popt[0]:.3f}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}) * sin({popt[3]:.3f}*x + {popt[4]:.3f})'

def absolute_shift_sinusoid_func(x, a, b, c, d, e):
    return (a * np.abs(x - b) + c) * np.sin(d * x + e)
absolute_shift_sinusoid_func.equation = lambda popt: f'y = ({popt[0]:.3f}*|x - {popt[1]:.3f}| + {popt[2]:.3f}) * sin({popt[3]:.3f}*x + {popt[4]:.3f})'

def linear_abs_sinusoid_func(x, a, b, c, d):
    return (a * x + b) * np.abs(np.sin(c * x + d))
linear_abs_sinusoid_func.equation = lambda popt: f'y = ({popt[0]:.3f}*x + {popt[1]:.3f}) * |sin({popt[2]:.3f}*x + {popt[3]:.3f})|'

def quadratic_abs_sinusoid_func(x, a, b, c, d, e):
    return (a * x**2 + b * x + c) * np.abs(np.sin(d * x + e))
quadratic_abs_sinusoid_func.equation = lambda popt: f'y = ({popt[0]:.3f}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}) * |sin({popt[3]:.3f}*x + {popt[4]:.3f})|'

def absolute_shift_abs_sinusoid_func(x, a, b, c, d, e):
    return (a * np.abs(x - b) + c) * np.abs(np.sin(d * x + e))
absolute_shift_abs_sinusoid_func.equation = lambda popt: f'y = ({popt[0]:.3f}*|x - {popt[1]:.3f}| + {popt[2]:.3f}) * |sin({popt[3]:.3f}*x + {popt[4]:.3f})|'

def sigmoid_func(x, K, B, M):
    return K / (1 + np.exp(-B * (x - M)))
sigmoid_func.equation = lambda popt: f'y = {popt[0]:.3f} / (1 + exp(-{popt[1]:.3f} * (x - {popt[2]:.3f})))'

def inv_sigmoid_func(y, K, B, M):
    return M - np.log(K / y - 1) / B
inv_sigmoid_func.equation = lambda popt: f'x = {popt[2]:.3f} - ln({popt[0]:.3f} / y - 1) / {popt[1]:.3f}'

def fit_name_of_func(fit_function):
    fit_name = fit_function.__name__
    if fit_name is not None and fit_name.endswith('_func'): fit_name = fit_name[:-len('_func')]
    return fit_name

def imshow(tensor, renderer=None, xaxis="", yaxis="", colorscale="RdBu", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale=colorscale, labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", line_labels=None, showlegend=None, hovertemplate=None, **kwargs):
    fig = px.line(utils.to_numpy(tensor), labels={"index":xaxis, "value":yaxis}, y=line_labels, **kwargs)
    if showlegend is not None: fig.update_layout(showlegend=showlegend)
    if hovertemplate is not None: fig.update_traces(hovertemplate=hovertemplate)
    fig.show(renderer)


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


def summarize(values, name=None, histogram=False, renderer=None, hist_args={},
              imshow_args=None, include_value=False, linear_fit=False,
              fit_function=None, fit_equation=None, fit_name=None,
              min=True, max=True, mean=True, median=True, range=True, range_size=True, firstn=None, abs_max=True):
    if histogram:
        hist_args_list = hist_args if isinstance(hist_args, list) else [hist_args]
        for hist_args in hist_args_list:
            hist_args = dict(hist_args)
            if 'title' not in hist_args and name is not None: hist_args['title'] = f'Histogram of {name}'
            if 'renderer' not in hist_args and renderer is not None: hist_args['renderer'] = renderer
            if 'xaxis' not in hist_args: hist_args['xaxis'] = name if name is not None else 'Value'
            if 'yaxis' not in hist_args: hist_args['yaxis'] = 'Count'
            hist(values, **hist_args)

    if imshow_args is not None:
        imshow_args = dict(imshow_args)
        if 'title' not in imshow_args and name is not None: imshow_args['title'] = name
        if 'renderer' not in imshow_args and renderer is not None: imshow_args['renderer'] = renderer
        if 'xaxis' not in imshow_args and name is not None: imshow_args['xaxis'] = f'({name}).shape[1]'
        if 'yaxis' not in imshow_args and name is not None: imshow_args['yaxis'] = f'({name}).shape[0]'
        if len(values.shape) == 1:
            line(values, **imshow_args)
        else:
            imshow(values, **imshow_args)

    if fit_function is None and linear_fit: fit_function = linear_func
    if fit_equation is None and fit_function is not None: fit_equation = fit_function.equation
    if fit_function is not None:
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
        if fit_name is None:
            fit_name = fit_function.__name__
            if fit_name is not None and fit_name.endswith('_func'): fit_name = fit_name[:-len('_func')]
        fit_name_space = '' if not fit_name else f'{fit_name} '
        fit_title = f"{aggregated}{name_space}Data and {fit_name_space}Fit"
        resid_title = f"{aggregated}{name_space}Residual Errors"

        # Fit linear regression to the aggregated data
        popt, _ = curve_fit(fit_function, x_vals, y_vals)

        # Create a subplot with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figure size to your liking

        # Scatter plot the data & best fit line on the first subplot
        axs[0].scatter(x_vals, y_vals, label='Data', alpha=0.5, s=1)
        axs[0].plot(x_vals, fit_function(x_vals, *popt), 'r-', label=f'Fit: {fit_equation(popt)}')
        axs[0].set_title(fit_title)
        axs[0].legend()

        # Plot residual errors on the second subplot
        residuals = y_vals - fit_function(x_vals, *popt)
        order_indices = np.argsort(x_vals)
        axs[1].scatter(x_vals[order_indices], residuals[order_indices], c='b', alpha=0.5)
        axs[1].set_title(resid_title)

        # Adjust the layout
        plt.tight_layout()
        plt.show()

    res = {}
    if include_value: res['value'] = values.detach().clone().cpu()
    if min: res['min'] = values.min().item()
    if max: res['max'] = values.max().item()
    if mean: res['mean'] = pm_mean_std(values.float())
    if median: res['median'] = values.median().item()
    if range: res['range'] = pm_range(values)
    if range_size: res['range_size'] = values.max().item() - values.min().item()
    if firstn is not None: res[f'first {firstn}'] = values[:firstn]
    if abs_max: res['abs(max)'] = values.abs().max().item()
    if fit_function is not None: res['fit_equation'] = f'y = {popt[0]}*x + {popt[1]}'
    if fit_function is not None: res['range_residuals'] = pm_range(residuals)
    if fit_function is not None: res['residuals'] = residuals[order_indices]
    if fit_function is not None: res['fit_params'] = popt

    return res


#list(zip(all_integers[~correct_idxs], all_integers_ans[~correct_idxs]))

def center_by_mid_range(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    maxv, minv = tensor.max(dim=dim, keepdim=True).values, tensor.min(dim=dim, keepdim=True).values
    return tensor - (maxv + minv) / 2.0

# # Simpler Model Interpretabiltiy

# ## Calculating how much slack we have

# Let's find out what the actual logits are, and how much slack we have on errors

# In[ ]:


def analyze_svd(M, descr='', scale_by_singular_value=True, colorscale='Picnic_r', singular_color='blue', renderer=None):
    U, S, Vh = torch.linalg.svd(M)
    V = Vh.T
    if scale_by_singular_value:
        U = U * S[None, :].sqrt()
        V = V * S[:, None].sqrt()
    if descr: descr = f' for {descr}'

    fig = make_subplots(rows=1, cols=3, subplot_titles=["U", "Singular Values", "V"])
    uzmax, vzmax = U.abs().max().item(), V.abs().max().item()
    fig.add_trace(go.Heatmap(z=utils.to_numpy(U), zmin=-uzmax, zmax=uzmax, colorscale=colorscale,
                             showscale=False,
                            hovertemplate="U: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>"
                            ),
                row=1, col=1)
    fig.add_trace(go.Heatmap(z=utils.to_numpy(V), colorscale=colorscale, zmin=-vzmax, zmax=vzmax,
                             showscale=False,
                            hovertemplate="V: %{x}<br>Singular Index: %{y}<br>Value: %{z}<extra></extra>",
                            ),
                row=1, col=3)
    fig.add_trace(go.Scatter(x=np.arange(S.shape[0]), y=utils.to_numpy(S),
                            mode='lines+markers',
                            marker=dict(color=singular_color),
                            line=dict(color=singular_color),
                            hovertemplate="Singular Value: %{y}<br>Singular Index: %{x}<extra></extra>",
                            ), row=1, col=2)
    fig.update_layout(title=f"SVD{descr}") #, margin=dict(l=150, r=150))


    fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_layout(yaxis_scaleanchor="x")
    fig.update_yaxes(scaleanchor='x', autorange='reversed', row=1, col=1)
    fig.update_yaxes(scaleanchor='x', autorange='reversed', row=1, col=3)

    # fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[0, U.shape[0]], row=1, col=1)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[0, U.shape[1]], row=1, col=1)

    # fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[0, None], row=1, col=2)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[0, S.shape[0]], row=1, col=2)

    # fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[0, Vh.shape[0]], row=1, col=3)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[0, Vh.shape[1]], row=1, col=3)

    # fig.update_xaxes(range=[0, None], row=1, col=1)
    # fig.update_xaxes(range=[0, None], row=1, col=2)
    # fig.update_xaxes(range=[0, None], row=1, col=3)

    # fig.update_yaxes(range=[0, None], row=1, col=1)
    # fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_yaxes(range=[0, None], row=1, col=3)

    # fig.update_yaxes(title_text="Query Token", row=1, col=1)
    fig.update_yaxes(range=[0, None], row=1, col=2)
    # fig.update_yaxes(title_text="Key Token", row=1, col=3)

    fig.show(renderer)


    # line(S, title=f"Singular Values{descr}")
    # imshow(U, title=f"Principal Components on U{descr}")
    # imshow(Vh, title=f"Principal Components on Vh{descr}")



# %%
@torch.no_grad()
def make_fit(values: torch.Tensor, fit_function, exclude_count=None):
    assert len(values.shape) in (1, 2)
    if len(values.shape) == 1:
        x_vals = np.arange(values.shape[0])
        y_vals = utils.to_numpy(values)
    else:
        x_vals = np.tile(np.arange(values.shape[1]), values.shape[0])
        y_vals = utils.to_numpy(values.flatten())

    x_vals_fit, y_vals_fit = x_vals, y_vals
    if exclude_count is not None: x_vals_fit, y_vals_fit = x_vals[exclude_count:-exclude_count], y_vals[exclude_count:-exclude_count]
    popt, _ = curve_fit(fit_function, x_vals_fit, y_vals_fit)

    residuals = y_vals - fit_function(x_vals, *popt)
    order_indices = np.argsort(x_vals)

    return popt, (x_vals, y_vals), (x_vals, fit_function(x_vals, *popt)), (x_vals[order_indices], residuals[order_indices])


def make_fit_traces(values: torch.Tensor, fit_function, exclude_count=None, fit_equation: Optional[Callable] = None, reference_lines: Optional[List[Tuple[str, float]]] = None, reference_colors=plotly.colors.qualitative.Dark24):
    popt, points, fit, resid = make_fit(values, fit_function, exclude_count=exclude_count)
    if fit_equation is None: fit_equation = fit_function.equation
    if reference_lines is None: reference_lines = []
    reference_line_traces = \
        [go.Scatter(x=np.arange(points[0].shape[0]), y=np.full(points[0].shape, val), name=name, mode='lines', line=dict(color=color, dash='dash'),
                hovertemplate=f'{val}<extra>{name}</extra>',
                showlegend=False, legendgroup=fit_function.__name__)
        for (name, val), color in zip(reference_lines, itertools.cycle(reference_colors))]
    # , size=1
    return popt, \
            [go.Scatter(x=points[0], y=points[1], name='Data', mode='markers', marker=dict(color='red', opacity=0.5), showlegend=True, legendgroup=fit_function.__name__),
            go.Scatter(x=fit[0], y=fit[1], name=f'Fit: {fit_equation(popt)}', mode='lines', line=dict(color='blue'), showlegend=True, legendgroup=fit_function.__name__),
            go.Scatter(x=resid[0], y=resid[1], name='Residuals', mode='markers', marker=dict(color='red', opacity=0.5), showlegend=False)], \
            reference_line_traces

def show_fits(values: torch.Tensor, name: str, fit_funcs: Iterable[Callable], do_exclusions=True, renderer=None, **kwargs):
    assert len(values.shape) == 1
    fit_funcs = list(fit_funcs)
    fig = make_subplots(rows=len(fit_funcs), cols=2,
                        subplot_titles=[title
                                        for fit_func in fit_funcs
                                        for title in (f"{fit_name_of_func(fit_func)} Fit", f"Residuals")])
    for i, fit_func in enumerate(fit_funcs):
        popt, (points, fit, resid), reference_line_traces = make_fit_traces(values, fit_func, exclude_count=None, **kwargs)
        fig.add_trace(points, row=i+1, col=1)
        fig.add_trace(fit, row=i+1, col=1)
        fig.add_trace(resid, row=i+1, col=2)
        for trace in reference_line_traces:
            fig.add_trace(trace, row=i+1, col=1)
    fig.update_layout(
        title=f"{name} Data & Fit",
        legend=dict(
            bgcolor='rgba(255,255,255,0.5)',
            yanchor="middle",
            y=0.5,  # Y=1 anchors the legend to the top of the plot area
            xanchor="left",
            x=0
        ),
        height=300 * len(fit_funcs) + 100,
    )

    if do_exclusions:
        max_param_count = max([len(signature(fit_func).parameters) for fit_func in fit_funcs])
        frames = [go.Frame(data=[trace
                                for fit_func in fit_funcs
                                for trace_list in make_fit_traces(values, fit_func, exclude_count=exclude_count, **kwargs)[1:]
                                for trace in trace_list],
                            name=(str(exclude_count) if exclude_count is not None else "0"),
        ) for exclude_count in [None] + list(range(1, (values.shape[0] - max_param_count) // 2))]

        fig.frames = frames

        sliders = [dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(font=dict(size=20), prefix='# End Points to Exclude:', visible=True, xanchor='right'),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(args=[[frame.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                        method='animate',
                        label=frame.name) for frame in fig.frames]
        )]

        fig.update_layout(sliders=sliders)

    fig.show(renderer)

# %%

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
    centered_by_mid_range = center_by_mid_range(res, dim=-1)
    centered = res - self_overlap[:,None]
    centered_triu = centered.triu()
    centered_tril = centered.tril()
    centered_no_diag = centered.clone()
    centered_no_diag.diagonal().fill_(-1000000)
    centered_no_diag_after_0 = centered_no_diag[:,1:]
    centered_no_diag = centered_no_diag[centered_no_diag != -1000000]
    centered_no_diag_after_0 = centered_no_diag_after_0[centered_no_diag_after_0 != -1000000]
    statistics = [
        ('centered overlap (incl pos)', centered),
        ('centered overlap (incl pos) triu', centered_triu),
        ('centered overlap (incl pos) tril', centered_tril),
        ('centered overlap after 0 (incl pos)', centered[:,1:]),
        ('centered overlap after 0 (incl pos) no diag', centered_no_diag_after_0),
        ('centered overlap only 0 (incl pos)', centered[:,0]),
        ('centered overlap only 0 (incl pos) no diag', centered[1:,0]),
        ('overlap (incl pos)', res),
        ('self-overlap (incl pos)', self_overlap),
        ('self-overlap after 0 (incl pos)', self_overlap[1:]),
        ('centered overlap (incl pos) no diag', centered_no_diag),
        ('centered by mid_range overlap (incl pos)', centered_by_mid_range),
        ('centered by mid_range overlap after 0 (incl pos)', centered_by_mid_range[:,1:]),
        ('centered by mid_range overlap only 0 (incl pos)', centered_by_mid_range[:,0]),
        ('centered by mid_range overlap only 0 (incl pos) no diag', centered_by_mid_range[1:,0]),
    ]
    return {name: summarize(value, include_value=False, name=name, renderer=renderer, linear_fit=True,
                            imshow_args={'yaxis':'input token', 'xaxis':'output token'},
                            ) for name, value in statistics}


def calculate_rowwise_embed_and_pos_embed_overlap(model: HookedTransformer, renderer=None):
    """
    For `(W_E + W_pos[-1,:]) @ W_U`, we compute for each row the maximum absolute value of the following quantity:
    - the largest negative value a number to the right of the diagonal is below the diagonal
    - the largest positive value a number to the left  of the diagonal is above the diagonal
    This is the exact value of the largest absolute error introduced in a given row.
    """
    W_U, W_E, W_pos = model.W_U, model.W_E, model.W_pos
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    res = ((W_E + W_pos[-1,:]) @ W_U).detach()
    self_overlap = res.diag()
    centered = res - self_overlap[:,None]
    centered_triu = centered.triu()
    centered_tril = centered.tril()
    diffs = centered_tril - centered_triu
    imshow(res)
    imshow(centered)
    imshow(diffs)
    # # max of positive differences to the right of the diagonal
    # max_pos_diffs = torch.max(centered_triu, dim=-1).values
    # # max of negative differences to the left of the diagonal
    # max_neg_diffs = torch.min(centered_tril, dim=-1).values
    # # stack the diffs
    # diffs = torch.stack([max_neg_diffs, max_pos_diffs], dim=-1)
    # summarize(diffs, name='rowwise diffs (positive and negative)', renderer=renderer)
    # # take the max of the diffs
    max_diffs = torch.max(diffs, dim=-1).values
    return summarize(max_diffs, name='rowwise max absolute diffs', include_value=True, linear_fit=True, renderer=renderer)



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
    imshow(res, title='W_pos @ W_V @ W_O @ W_U', xaxis='logit affected', yaxis='position', renderer=renderer)
    return summarize(res, name='W_pos @ W_V @ W_O @ W_U', renderer=renderer, linear_fit=True)
# %%
def analyze_PVOU(model: HookedTransformer, colorscale='RdBu', renderer=None):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_pos @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    pos_indices = torch.arange(n_ctx)
    fig = px.imshow(utils.to_numpy(res), title='W_pos @ W_V @ W_O @ W_U',
                    labels={"x":'logit affected', "y":'position'},
                    color_continuous_midpoint=0.0, color_continuous_scale=colorscale)
    fig.update_yaxes(tickvals=pos_indices, ticktext=pos_indices)
    fig.show(renderer)

# %%
def analyze_PU(model: HookedTransformer, colorscale='RdBu', renderer=None):
    W_U, W_pos = model.W_U, model.W_pos
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    res = (W_pos[-1, :] @ W_U).detach()
    line(res, title='W_pos[-1] @ W_U', xaxis='output token', showlegend=False, hovertemplate='Logit for %{x}: %{y}', renderer=renderer)

# %%
def analyze_EU(model: HookedTransformer, colorscale='RdBu', renderer=None):
    W_U, W_E = model.W_U, model.W_E
    d_model, d_vocab = model.cfg.d_model, model.cfg.d_vocab
    assert W_U.shape == (d_model, d_vocab)
    assert W_E.shape == (d_vocab, d_model)
    res = (W_E @ W_U).detach()
    imshow(res, title='W_E @ W_U', renderer=renderer,
           xaxis="logit affected", yaxis="input token", colorscale=colorscale)




# ## Copying: W_E @ W_V @ W_O @ W_U

# %%
def analyze_EVOU(model: HookedTransformer, colorscale='RdBu', renderer=None, scale_by_singular_value=True):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_E @ W_V @ W_O @ W_U).detach().cpu()[0,0,:,:]
    imshow(res, title='W_E @ W_V @ W_O @ W_U', renderer=renderer,
           xaxis="logit affected", yaxis="input token", colorscale=colorscale)
    analyze_svd(res, descr='W_E @ W_V @ W_O @ W_U', colorscale=colorscale, scale_by_singular_value=scale_by_singular_value, renderer=renderer)
    line(res.diag(), title='(W_E @ W_V @ W_O @ W_U).diag()', xaxis='input token', showlegend=False, hovertemplate='Input Token: %{x}<br>Logit on %{x}: %{y}', renderer=renderer)

# In[ ]:


def calculate_copying(model: HookedTransformer, colorscale='RdBu', renderer=None, scale_by_singular_value=True):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_E @ W_V @ W_O @ W_U).detach().cpu()[0,0,:,:]
    res_diag = res.diag()
    res_off_diagonal = res[torch.eye(d_vocab) == 0]
    centered = -res + res.diag()[:, None]
    nonzero_centered = centered[torch.eye(d_vocab) == 0]
    imshow(res, title='W_E @ W_V @ W_O @ W_U', renderer=renderer,
           xaxis="logit affected", yaxis="input token")
    analyze_svd(res, descr='W_E @ W_V @ W_O @ W_U', colorscale=colorscale, scale_by_singular_value=scale_by_singular_value, renderer=renderer)
    # imshow(centered, title='copying.diag()[:,None] - copying', renderer=renderer)
    line(res.diag(), title='copying.diag()', xaxis='input token', renderer=renderer)
    # take svd of res
    u, s, vh = torch.linalg.svd(res)
    v = vh.T
    # plot singular values
    line(s, title='singular values of copying', renderer=renderer)
    # plot u, v
    imshow(u, title='u', renderer=renderer)
    imshow(v, title='v', renderer=renderer)

    # 1. We already have u, s, and v from torch.linalg.svd(res)
    u1 = u[:, 0]
    v1 = v[:, 0]

    # 2. Fit linear models to u1 and v1
    # Fit for u's first column
    x_vals_u = np.arange(d_vocab)
    y_vals_u = u[:, 0].numpy()
    popt_u, _ = curve_fit(linear_func, x_vals_u, y_vals_u)

    # Fit for v's first column
    x_vals_v = np.arange(d_vocab)
    y_vals_v = v[0, :].numpy()
    popt_v, _ = curve_fit(linear_func, x_vals_v, y_vals_v)

    # Plot u's column against its linear fit
    plt.figure()
    plt.scatter(x_vals_u, y_vals_u, alpha=0.5, label='Data')
    plt.plot(x_vals_u, linear_func(x_vals_u, *popt_u), 'r-', label=f'u: y = {popt_u[0]:.4f}x + {popt_u[1]:.4f}')
    plt.title("First Column of u vs Linear Fit")
    plt.legend()
    plt.show()

    # Plot residuals for u
    plt.figure()
    residuals_u = y_vals_u - linear_func(x_vals_u, *popt_u)
    plt.scatter(x_vals_u, residuals_u, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals of u's First Column Fit")
    plt.show()

    # Plot v's row against its linear fit
    plt.figure()
    plt.scatter(x_vals_v, y_vals_v, alpha=0.5, label='Data')
    plt.plot(x_vals_v, linear_func(x_vals_v, *popt_v), 'r-', label=f'v: y = {popt_v[0]:.4f}x + {popt_v[1]:.4f}')
    plt.title("First Row of v vs Linear Fit")
    plt.legend()
    plt.show()

    # Plot residuals for v
    plt.figure()
    residuals_v = y_vals_v - linear_func(x_vals_v, *popt_v)
    plt.scatter(x_vals_v, residuals_v, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals of v's First Row Fit")
    plt.show()

    # Subtract impact of lines
    u_prime = linear_func(x_vals_u, *popt_u)
    v_prime = linear_func(x_vals_v, *popt_v)
    impact = s[0] * u_prime[:, None] @ v_prime[None, :]
    adjusted_res = res - impact
    imshow(impact, title="adjustment", renderer=renderer)

    # adjusted_res = res - s[0] * (u[:, 0:1] @ v[:,0:1].T) * (popt_u[0] * x_vals_u[:, None] + popt_v[0] * x_vals_v[None, :])

    imshow(adjusted_res, title='Adjusted res', renderer=renderer)

    # SVD of adjusted_res
    u_adj, s_adj, vh_adj = torch.linalg.svd(adjusted_res)
    line(s_adj, title='Singular Values of Adjusted res', renderer=renderer)
    imshow(u_adj, title='u of residuals', renderer=renderer)
    imshow(vh_adj.T, title='v of residuals', renderer=renderer)

    # Extracting diagonal and off-diagonal entries
    diagonal_entries = torch.diag(adjusted_res)
    off_diagonal_entries = adjusted_res - torch.diag_embed(diagonal_entries)
    off_diagonal_entries = off_diagonal_entries[off_diagonal_entries != 0]

    # Finding the smallest diagonal entry and the largest off-diagonal entry
    min_diagonal_entry = diagonal_entries.min().item()
    max_off_diagonal_entry = off_diagonal_entries.max().item()

    # Printing the results
    print(f"Smallest diagonal entry: {min_diagonal_entry} ({pm_range(diagonal_entries)})")
    print(f"Largest off-diagonal entry: {max_off_diagonal_entry} ({pm_range(off_diagonal_entries)})")

    line(diagonal_entries, title='Diagonal Entries', renderer=renderer)

    off_diagonal_entries = off_diagonal_entries.flatten()
    # Histogram plot
    plt.hist(diagonal_entries.numpy(), bins=50, color='blue', alpha=0.7, label='Diagonal entries')
    plt.hist(off_diagonal_entries.numpy(), bins=50, color='red', alpha=0.5, label='Off-diagonal entries')
    plt.legend(loc='upper right')
    plt.title('Histogram of Diagonal and Off-diagonal Entries')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # Histogram plot
    plt.hist(diagonal_entries.numpy(), bins=50, color='blue', alpha=0.7, label='Diagonal entries', density=True)
    plt.hist(off_diagonal_entries.numpy(), bins=50, color='red', alpha=0.5, label='Off-diagonal entries', density=True)
    plt.legend(loc='upper right')
    plt.title('Density Histogram of Diagonal and Off-diagonal Entries')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()


    centered_adjusted_res = -adjusted_res + adjusted_res.diag()[:, None]
    nonzero_centered_adjusted_res = centered_adjusted_res[centered_adjusted_res != 0]

    imshow(centered_adjusted_res, title='adjusted copying.diag()[:,None] - adjusted copying', renderer=renderer)
    print(f"range on nonzero centered adjusted res: {pm_range(nonzero_centered_adjusted_res)}")

    statistics = [
        ('copying', res),
        ('diag', res_diag),
        ('off-diag', res_off_diagonal),
        ('centered', centered),
        ('nonzero centered', nonzero_centered),
    ]

    summaries = {name: summarize(value, name=name, renderer=renderer, histogram=False) for name, value in statistics}
    for k, v in summaries.items():
        print(k, v)
    return res


# In[ ]:


def calculate_copying_with_pos(model: HookedTransformer, renderer=None):
    W_U, W_E, W_pos, W_V, W_O = model.W_U, model.W_E, model.W_pos, model.W_V, model.W_O
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_V.shape == (1, 1, d_model, d_model)
    assert W_O.shape == (1, 1, d_model, d_model)
    res = (W_E @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    res_pos = (W_pos @ W_V @ W_O @ W_U).detach()[0,0,:,:]
    res_pos_min, res_pos_max = res_pos.min(dim=0).values, res_pos.max(dim=0).values
    res_diag = res.diag() + res_pos_min
    res_above_diag = -(res + res_pos_max[None,:]) + res_diag[:, None]
    imshow(res_above_diag, title='(W_E + worst(W_pos)) @ W_V @ W_O @ W_U', renderer=renderer,
              xaxis="logit affected", yaxis="input token")
    res_above_diag_off_diag = res_above_diag[torch.eye(d_vocab) == 0]
    first_diagonal = res.diag(diagonal=1) + res_pos_min[:-1]
    res_above_first_diagonal = -(res[:-1,:] + res_pos_max[None,:]) + first_diagonal[:, None]
    statistics = [
       ('res_above_diag_off_diag', res_above_diag_off_diag),
         ('res_above_first_diagonal', res_above_first_diagonal),
    ]
    for name, value in statistics:
        print(name, summarize(value, name=name, renderer=renderer, histogram=True))


# ## Attention Scaling Factor

# In[ ]:


def calculate_attn(model: HookedTransformer, pos: Optional[int] = None, renderer=None):
    W_U, W_E, W_pos, W_Q, W_K = model.W_U, model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    if pos is None:
        return [calculate_attn(model, pos=i, renderer=renderer) for i in range(n_ctx)]
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_Q.shape == (1, 1, d_model, d_model)
    assert W_K.shape == (1, 1, d_model, d_model)
    residm1 = (W_E + W_pos[-1,:][None,:])
    resid = (W_E + W_pos[pos,:][None,:])
    q = (residm1 @ W_Q)[0,0,:,:]
    k = (resid @ W_K)[0,0,:,:]
    res = (q @ k.T).detach()
    # imshow(res, title=f'(W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos[{pos}]).T', renderer=renderer)
    centered = res - res.mean(dim=-1, keepdim=True)
    imshow(centered, title=f'centered (W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos[{pos}]).T', renderer=renderer,
           xaxis="Key token", yaxis="Query token")
    return centered


# %%

# check for monotonicity violations
def check_monotonicity(model: HookedTransformer, renderer=None):
    count = 0
    centered_scores = calculate_attn(model, renderer=renderer)
    for pos, centered_score in enumerate(centered_scores):
        for row_n, row in enumerate(centered_score):
            for i in range(row.shape[0] - 1):
                for j in range(i + 1, row.shape[0]):
                    if row[i] > row[j]:
                        count += 1
                        print(f"{i, j} at row {row_n} pos {pos}, magnitude {row[i] - row[j]:.3f}")
    return count


# %%

def calculate_attn_by_pos(model: HookedTransformer, pos=False, renderer=None):
    W_U, W_E, W_pos, W_Q, W_K = model.W_U, model.W_E, model.W_pos, model.W_Q, model.W_K
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_U.shape == (d_model, d_vocab)
    assert W_pos.shape == (n_ctx, d_model)
    assert W_E.shape == (d_vocab, d_model)
    assert W_Q.shape == (1, 1, d_model, d_model)
    assert W_K.shape == (1, 1, d_model, d_model)
    residm1 = (W_E + W_pos[-1,:][None,:])

    resid = (W_E if not pos else W_pos[0,:][None,:] - W_pos[1, :][None,:])
    resid_name = 'W_E' if not pos else f'(W_pos[0] - W_pos[1])'
    q = (residm1 @ W_Q)[0,0,:,:]
    k = (resid @ W_K)[0,0,:,:]
    res = (q @ k.T).detach()
    # imshow(res, title=f'(W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos[{pos}]).T', renderer=renderer)
    centered = res - res.mean(dim=-1, keepdim=True) if not pos else res
    imshow(centered, title=f'centered (W_E + W_pos[-1]) @ W_Q @ W_K.T @ {resid_name}.T', renderer=renderer,
           xaxis="Key token", yaxis="Query token")
    #print(centered.shape)
    return summarize(centered, name=f'centered (W_E + W_pos[-1]) @ W_Q @ W_K.T @ {resid_name}.T',
                     renderer=renderer,
                     include_value=True)

def replace_nans_with_row_max(tensor):
    # Step 1: Identify the nan values
    nan_mask = torch.isnan(tensor)

    # Step 2: Compute the maximum value for each row, ignoring nans
    non_nan_tensor = torch.where(nan_mask, torch.tensor(float('-inf')).to(tensor.device), tensor)
    row_max, _ = torch.max(non_nan_tensor, dim=1, keepdim=True)

    # Replace nan with the max value of the respective row
    tensor[nan_mask] = row_max.expand_as(tensor)[nan_mask]

    return tensor

def calculate_rowwise_attn_by_pos_near(model: HookedTransformer, pos=False, renderer=None, max_offset=1):
    def pad_diagonal_with(shape, diag, offset, val=100000):
        before_padding = torch.zeros(list(shape[:-2]) + [np.max([0, -offset])], device=diag.device)
        after_padding  = torch.zeros(list(shape[:-2]) + [np.max([0,  offset])], device=diag.device)
        before_padding.fill_(val)
        after_padding.fill_(val)
        return torch.cat([before_padding, diag, after_padding], dim=-1)

    points = []
    centered_score = calculate_attn_by_pos(model, renderer=renderer, pos=pos)['value']
    centered_diag = centered_score.diag()
    centered_score = centered_score - centered_diag[:, None]
    res = torch.stack([np.sign(offset) * pad_diagonal_with(centered_score.shape, centered_score.diag(diagonal=offset), offset, val=float('nan'))
                       for offset in range(-max_offset, max_offset + 1) if offset != 0], dim=-1)
    imshow(centered_score, renderer=renderer)
    imshow(res, renderer=renderer)
    res = replace_nans_with_row_max(res)
    min_right_attn = res.min(dim=-1).values
    return summarize(min_right_attn, name=f'min right attn by pos near {max_offset}', renderer=renderer, include_value=True, fit_function=quadratic_func)
    #return min(points)

def calculate_min_attn_by_pos_far(model: HookedTransformer, pos=False, renderer=None, min_offset=2):
    points = []
    centered_score = calculate_attn_by_pos(model, renderer=renderer, pos=pos)['value']
    for row_n, row in enumerate(centered_score):
        for i in range(row.shape[0]):
            if i != row_n and abs(i - row_n) >= min_offset:
                points.append((row[i].item() - row[row_n].item())  / (i - row_n))
    # histogram
    plt.hist(points, bins=100, edgecolor='black')
    return min(points)

# ## Attention Patterns

# In[ ]:


def calculate_qk_attn_heatmap(model, keypos=-1, querypos=-1, do_layernorm=True):
    attn = model.blocks[0].attn
    all_token_embeddings = model.embed(range(model.cfg.d_vocab))
    positional_embeddings = model.pos_embed(all_token_embeddings)

    token_embeddings_at_keypos = all_token_embeddings + positional_embeddings[:,keypos,:] if keypos > -1 else all_token_embeddings
    token_embeddings_at_querypos = all_token_embeddings + positional_embeddings[:,querypos,:] if querypos > -1 else all_token_embeddings

    # layernorm before attention
    if do_layernorm:
        token_embeddings_at_keypos = model.blocks[0].ln1(token_embeddings_at_keypos)
        token_embeddings_at_querypos = model.blocks[0].ln1(token_embeddings_at_querypos)

    embeddings_key = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                            token_embeddings_at_keypos, attn.W_K)
    embeddings_query = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
                            token_embeddings_at_querypos, attn.W_Q)

    qk_circuit_attn_heatmap = einsum(
        "n_heads d_vocab_q d_head, n_heads d_vocab_k d_head -> ... d_vocab_q d_vocab_k",
        embeddings_query, embeddings_key
        ).detach().cpu().numpy()

    plt.rcParams['figure.figsize'] = [20, 10]
    return qk_circuit_attn_heatmap


def calculate_qk_attn_heatmap_normed(model, querypos=-1, do_layernorm=True, skip_var=True):
    all_token_embeddings = model.embed(range(model.cfg.d_vocab))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    all_heatmaps = torch.stack([torch.tensor(calculate_qk_attn_heatmap(model, cur_keypos, querypos, do_layernorm=do_layernorm)) for cur_keypos in range(positional_embeddings.shape[-2])])
    avg = einops.reduce(all_heatmaps, "keypos d_vocab_q d_vocab_k -> d_vocab_q ()", 'mean')
    var = einops.reduce(all_heatmaps, "keypos d_vocab_q d_vocab_k -> d_vocab_q ()", torch.var)
    #print(all_heatmaps.shape, avg.shape)
    #print(avg)
    res = (all_heatmaps - avg)
    if not skip_var: res = res * (var ** -0.5)
    return res


def plot_qk_heatmap(model, keypos=-1, querypos=-1, do_layernorm=True):
  qk_attn_heatmap = calculate_qk_attn_heatmap(model, keypos=keypos, querypos=querypos, do_layernorm=do_layernorm)

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(qk_attn_heatmap, cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()


def plot_qk_heatmaps_normed(model, keypositions=None, querypos=-1, do_layernorm=True, skip_var=True):
    if keypositions is None:
        all_token_embeddings = model.embed(range(model.cfg.d_vocab))
        positional_embeddings = model.pos_embed(all_token_embeddings)
        keypositions = range(positional_embeddings.shape[-2])

    heatmaps = calculate_qk_attn_heatmap_normed(model, querypos=querypos, do_layernorm=do_layernorm, skip_var=skip_var)
    for keypos in keypositions:
        fig, ax = plt.subplots(figsize=(8, 8))
        qk_attn_heatmap = heatmaps[keypos]
        graph = ax.imshow(qk_attn_heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar(graph)
        plt.tight_layout()
        plt.show()
    print(heatmaps.shape) # torch.Size([2, 64, 64]), keypos d_vocab_q d_vocab_k


def plot_avg_qk_heatmap(model, keypositions, querypos=-1, do_layernorm=True):
  heatmaps = []

  for keypos in keypositions:
    heatmaps.append(calculate_qk_attn_heatmap(model, keypos=keypos, querypos=querypos, do_layernorm=do_layernorm))

  qk_circuit_attn_heatmap = np.mean(heatmaps, axis=0)

  fig, ax = plt.subplots(figsize=(8, 8))
  graph = ax.imshow(qk_circuit_attn_heatmap, cmap="hot", interpolation="nearest")
  plt.colorbar(graph)
  plt.tight_layout()


#list(zip(all_integers[~correct_idxs], all_integers_ans[~correct_idxs]))


# # Interpretability

# ## Unembed

# In[ ]:


def plot_unembed_cosine_similarity(model):
    all_token_embeddings = model.embed(range(model.cfg.d_vocab))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    all_token_pos_embed = all_token_embeddings[:,None,:] + positional_embeddings
    #print(model.W_U.shape, all_token_embeddings.shape, positional_embeddings.shape)
    # torch.Size([32, 64]) torch.Size([64, 32]) torch.Size([64, 2, 32])
    avg = F.normalize(all_token_embeddings.sum(dim=0), dim=-1)
    # overlap between model.W_U and token embedings
    input_overlap = all_token_pos_embed @ model.W_U
    print(f"Definition max_input_output_overlap := {input_overlap.abs().max()}.")
    line(F.cosine_similarity(avg[None,:], all_token_embeddings, dim=-1))


# In[ ]:


def count_monotonicity_violations_line(result_tensor, m):
    # Count the number of pairs of indices (i, j), i != j, for which
    # (result_tensor[i] + m*i - result_tensor[j] + m*j) / (i - j) is negative
    count = 0
    for i in range(len(result_tensor)):
        for j in range(i + 1, len(result_tensor)):
            if ((result_tensor[i] + m*i - result_tensor[j] + m*j) / (i - j)) < 0:
                count += 1
    return count


def reorder_tensor_greedy(tensor, m):
    # Convert to numpy for easier handling
    tensor_np = tensor.detach().clone().numpy()

    # Initialize the result with the maximum positive value
    result = [np.max(tensor_np)]
    tensor_np = np.delete(tensor_np, np.argmax(tensor_np))

    while len(tensor_np) > 0:
        # Find values that maintain the condition
        candidates = tensor_np[tensor_np - result[-1] < -m]

        if len(candidates) > 0:
            # If such values exist, select the maximum
            next_value = np.max(candidates)
        else:
            # Otherwise, select the maximum of the remaining values
            next_value = np.max(tensor_np)

        # Add the selected value to the result
        result.append(next_value)

        # Remove the selected value from the list of remaining values
        tensor_np = np.delete(tensor_np, np.where(tensor_np == next_value)[0][0])

    # Convert the result back to a tensor
    result_tensor = torch.tensor(result)

    # Count the number of indices for which the difference between
    # successive elements in the result is less than -m
    # diff = result_tensor[1:] - result_tensor[:-1]
    # count = torch.sum(diff < -m).item()

    count = count_monotonicity_violations_line(result_tensor, m)

    return result_tensor, count


def compute_best_fit_and_error(direction_dot_embed):
    n_head, d_vocab = direction_dot_embed.shape

    coefficients = torch.empty((n_head, 2))  # To store the coefficients a, b for each row
    max_abs_errors = torch.empty(n_head)  # To store the max abs error for each row
    errors = torch.empty((n_head, d_vocab))
    predicted = torch.empty((n_head, d_vocab))
    negative_pairs = []
    diff_values = []

    x_values = np.arange(d_vocab)

    # Create a meshgrid of indices
    idxi, idxj = np.meshgrid(x_values, x_values)
    # Exclude the diagonal (i == j)
    mask = idxi != idxj
    pairs = list(zip(idxi[mask], idxj[mask]))  # create a list of pairs (i, j)

    for i in range(n_head):
        row = direction_dot_embed[i].detach().numpy()

        # Use curve_fit to find a, b that best fit the data in this row
        coeff, _ = curve_fit(linear_func, x_values, row)
        coefficients[i] = torch.from_numpy(coeff)

        # Compute the predicted y values using these coefficients
        y_pred = coeff[0] * x_values + coeff[1]

        # Compute the absolute error for each value, and take the maximum
        cur_errors = row - y_pred
        max_abs_errors[i] = np.abs(cur_errors).max()
        errors[i] = torch.from_numpy(cur_errors)
        predicted[i] = torch.from_numpy(y_pred)

        # Compute (pos[i] - pos[j]) / (i - j) for all pairs (i, j)
        values = (row[idxi] - row[idxj]) / (idxi - idxj)

        # Select only the values where i != j
        values = values[mask]
        negative_pairs.append([pair for pair, value in zip(pairs, values) if value < 0])

        diff_values.append(values)

    return coefficients, max_abs_errors, errors, predicted, diff_values, negative_pairs


def plot_QK_cosine_similarity(model, keypos=-1, querypos=-1, do_layernorm=True):
    attn = model.blocks[0].attn
    all_token_embeddings = model.embed(range(model.cfg.d_vocab))
    positional_embeddings = model.pos_embed(all_token_embeddings)
    normed_all_token_embeddings = F.normalize(all_token_embeddings, dim=-1)

    token_embeddings_at_keypos = all_token_embeddings + positional_embeddings[:,keypos,:] if keypos > -1 else all_token_embeddings
    token_embeddings_at_querypos = all_token_embeddings + positional_embeddings[:,querypos,:] if querypos > -1 else all_token_embeddings

    # layernorm before attention
    if do_layernorm:
        token_embeddings_at_keypos = model.blocks[0].ln1(token_embeddings_at_keypos)
        token_embeddings_at_querypos = model.blocks[0].ln1(token_embeddings_at_querypos)

    #embeddings_key = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
    #                        token_embeddings_at_keypos, attn.W_K)
    #embeddings_query = einsum("d_vocab d_model, n_heads d_model d_head -> n_heads d_vocab d_head",
    #                        token_embeddings_at_querypos, attn.W_Q)
    embeddings_query_waiting_for_key = einsum("d_vocab_query d_model_query, n_heads d_model_query d_head, n_heads d_model_key d_head -> n_heads d_vocab_query d_model_key",
                            token_embeddings_at_querypos, attn.W_Q, attn.W_K)

    QK = einsum("n_heads d_model_query d_head, n_heads d_model_key d_head -> n_heads d_model_query d_model_key",
                            attn.W_Q, attn.W_K)

    analyze_svd(embeddings_query_waiting_for_key[0], descr="embeddings_query_waiting_for_key")
    analyze_svd(QK[0], descr="QK")
    U, S, Vh = torch.linalg.svd(embeddings_query_waiting_for_key[0])
    print(Vh.T[0])
    print(Vh[0])
    print((U @ torch.diag(S) @ Vh)[0])
    print((U @ torch.diag(S) @ Vh).T[0])
    imshow(U @ torch.diag(S) @ Vh, title="tmp")
    #qk_circuit_attn_heatmap = einsum(
    #    "n_heads d_vocab_q d_head, n_heads d_vocab_k d_head -> ... d_vocab_q d_vocab_k",
    #    embeddings_query, embeddings_key
    #    ).detach().cpu().numpy()

    imshow(embeddings_query_waiting_for_key[0])


    direction = embeddings_query_waiting_for_key
    #direction = direction / direction.norm(dim=-1, keepdim=True)
    direction = direction.sum(dim=1) / direction.shape[1]
    print(f"Definition size_direction := {direction}.")
    direction = direction / direction.norm(dim=-1)
    print(f"Definition normed_size_direction := {direction}.")
    print(all_token_embeddings.shape, direction.shape)
    proj_direction_scale = einsum("n_head d_model_key, n_head d_vocab_query d_model_key -> n_head d_vocab_query",
                                  direction,
                                  embeddings_query_waiting_for_key)[:,:,None]
    print(proj_direction_scale.shape)
    proj_direction = proj_direction_scale * einops.rearrange(direction, "n_head d_model -> n_head () d_model")
    print(proj_direction.shape)
    remaining_directions = embeddings_query_waiting_for_key - proj_direction
    print(remaining_directions.shape)
    remaining_directions = remaining_directions.norm(dim=-1)
    print(remaining_directions.shape)
    direction_key_overlap = einsum("n_head d_model_key, n_head d_vocab_query d_model_key -> d_vocab_query n_head",
                direction,
                embeddings_query_waiting_for_key)
    print(direction_key_overlap.shape)
    print(f"Definition min_attention_query_size_direction_overlap := {direction_key_overlap.min()}.")
    direction_dot_embed = einsum("n_head d_model, d_vocab d_model -> n_head d_vocab", direction, normed_all_token_embeddings)
    direction_dot_pos_embed = einsum("n_head d_model, pos d_model -> n_head pos", direction, positional_embeddings[0])
    print(f"Definition max_direction_dot_pos_embed := {direction_dot_pos_embed.abs().max()}.")
    # linear fit of direction_dot_embed
    direction_dot_embed_coefficients, direction_dot_embed_max_abs_errors, direction_dot_embed_error, direction_dot_embed_predicted, direction_dot_embed_diff_values, direction_dot_embed_neg_values = \
          compute_best_fit_and_error(direction_dot_embed)

    direction_dot_embed_diffs = direction_dot_embed[...,1:] - direction_dot_embed[...,:-1]
    #direction_dot_embed_coef = direction_dot_embed_diffs.mean(dim=-1, keepdim=True)
    #direction_dot_embed_offset = direction_dot_embed.mean(dim=-1, keepdim=True)
    #direction_dot_embed_diff_error = direction_dot_embed_diffs - torch.arange(direction_dot_embed_diffs.shape[-1]) * direction_dot_embed_coef + direction_dot_embed_offset)
    print(direction_dot_embed_diffs)
    print(direction_dot_embed_diffs.abs())
    line(direction_dot_embed_diffs.T, title="direction_dot_embed_diffs")
    line(direction_dot_embed_diffs.T.abs(), title="direction_dot_embed_diffs abs")
    #line(direction_dot_embed_diff_error.T, title="direction_dot_embed_diff_error")
    #print(direction_dot_embed_coef, direction_dot_embed_offset)


    #direction_dot_embed_coef_better, _ = curve_fit(constant_function, np.arange(direction_dot_embed_diffs.shape[-1]), direction_dot_embed_diffs[0].detach().numpy())
    #direction_dot_embed_diff_error_better = direction_dot_embed_diffs - (torch.arange(direction_dot_embed_diffs.shape[-1]) * direction_dot_embed_coef + direction_dot_embed_offset)
    #line(direction_dot_embed_diffs.T, title="direction_dot_embed_diffs")
    #line(direction_dot_embed_diff_error.T, title="direction_dot_embed_diff_error")
    #print(direction_dot_embed_coef, direction_dot_embed_offset)


    # indices = np.argsort(direction_dot_embed_error[0].numpy() / np.arange(1, len(direction_dot_embed_error[0]) + 1))

    # Use these indices to sort 'direction_dot_embed_error'
    # sorted_direction_dot_embed_error = direction_dot_embed_error[:,indices]
    print(direction_dot_embed_error.mean(), direction_dot_embed_error.var())
    # randomly reorder direction_dot_embed_error, put in tmp
    tmp = direction_dot_embed_error[0].detach().clone().numpy()
    np.random.shuffle(tmp)
    print(tmp)
    line(tmp)
    print(count_monotonicity_violations_line(torch.tensor(tmp), direction_dot_embed_coefficients[0, 0].item()))
    print(f"Definition ")
    sorted_direction_dot_embed_error, bad_count = reorder_tensor_greedy(direction_dot_embed_error[0], direction_dot_embed_coefficients[0, 0].item())
    sorted_direction_dot_embed_error = sorted_direction_dot_embed_error[None,:]

    # sorted_direction_dot_embed_error, _ = direction_dot_embed_error.sort(dim=-1, descending=True)
    print(direction_dot_embed_coefficients, direction_dot_embed_max_abs_errors, direction_dot_embed)
    line(direction_key_overlap, title="direction @ query_waiting_for_key")
    line(remaining_directions.T, title="norm of remaining direction")
    line(F.cosine_similarity(direction, all_token_embeddings, dim=-1), title="cos_sim(direction, embed)")
    print(positional_embeddings.shape)
    line(direction_dot_embed.T, title="direction @ normed embed")
    line(torch.cat([direction_dot_embed, direction_dot_embed_predicted], dim=0).T, title="direction @ normed embed, + fit")
    print(bad_count)
    line(torch.cat([direction_dot_embed_predicted, direction_dot_embed_predicted + sorted_direction_dot_embed_error], dim=0).T, title="direction @ normed embed bad fit")

    # Plot the histogram
    print(len(direction_dot_embed_neg_values[0]) // 2)
    print(list(sorted([p for p in direction_dot_embed_neg_values[0] if p[0] < p[1]])))
    plt.hist(direction_dot_embed_diff_values[0], bins=30, edgecolor='black')
    plt.title("Distribution of (pos[i] - pos[j]) / (i - j)")
    plt.xlabel("(pos[i] - pos[j]) / (i - j)")
    plt.ylabel("Frequency")
    plt.show()

    line(direction_dot_embed_error.T, title="direction_dot_normed_embed_error")
    line(direction_dot_pos_embed.T, title="direction @ pos_embed")


# %%
def make_local_tqdm(tqdm):
    if tqdm is None:
        return lambda arg, **kwargs: arg
    else:
        return tqdm

# %%
@torch.no_grad()
def layernorm_noscale(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(axis=-1, keepdim=True)

# %%
@torch.no_grad()
def layernorm_scales(x: torch.Tensor, eps: float = 1e-5, recip: bool = True) -> torch.Tensor:
    x = layernorm_noscale(x)
    scale = (x.pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    if recip: scale = 1 / scale
    return scale

# %%
@torch.no_grad()
def compute_singular_contribution(M: torch.Tensor, plot_heatmaps=True, yaxis=None, xaxis=None, title=None, renderer=None, description=None, singular_value_count=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(M)
    U[:, singular_value_count:], S[singular_value_count:], Vh[singular_value_count:, :] = 0, 0, 0
    contribution = U @ torch.diag(S) @ Vh
    if plot_heatmaps:
        singular_value_str = f"first {singular_value_count} singular values" if singular_value_count != 1 else f"first singular value"
        to_description = f" to {description}" if description is not None else ""
        description = f"{description} " if description is not None else ""
        diff_zmax = (M - contribution).abs().max().item()
        zmax = np.max([contribution.abs().max().item(), diff_zmax])
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Contribution", f"Residual", f"Residual (rescaled)"])
        fig.add_trace(go.Heatmap(z=utils.to_numpy(contribution), zmin=-zmax, zmax=zmax, showscale=True,
                                 colorbar=dict(x=-0.15, y=0.5),
                                **kwargs),
                    row=1, col=1)
        fig.add_trace(go.Heatmap(z=utils.to_numpy(M - contribution), zmin=-zmax, zmax=zmax, showscale=False,
                                **kwargs),
                    row=1, col=2)
        fig.add_trace(go.Heatmap(z=utils.to_numpy(M - contribution), zmin=-diff_zmax, zmax=diff_zmax, showscale=True,
                                **kwargs),
                    row=1, col=3)
        if title is None: title = f"Contribution of the {singular_value_str}{to_description}"
        fig.update_layout(title=title, margin=dict(l=100))
        for col in range(3):
            if yaxis is not None: fig.update_yaxes(title_text=yaxis, row=1, col=col+1)
            if xaxis is not None: fig.update_xaxes(title_text=xaxis, row=1, col=col+1)
    fig.show(renderer)
    return M - contribution, contribution
# %%

def display_size_direction_stats(size_direction: torch.Tensor, query_direction: torch.Tensor, QK: torch.Tensor, U: torch.Tensor, Vh: torch.Tensor, S: torch.Tensor,
                                 size_direction_resid: Optional[torch.Tensor] = None, size_direction_QK: Optional[torch.Tensor] = None,
                                 query_direction_resid: Optional[torch.Tensor] = None, query_direction_QK: Optional[torch.Tensor] = None,
                                 do_exclusions: bool = True,
                                 include_contribution: bool = True,
                                 scale_by_singular_value: bool = True,
                                 renderer=None,
                                 fit_funcs: Iterable = (cubic_func, quintic_func),
                                 delta_fit_funcs: Iterable = (quadratic_func, quartic_func),
                                 colorscale='Plasma_r', **kwargs):
    if scale_by_singular_value:
        U = U * S[None, :].sqrt()
        Vh = Vh * S[:, None].sqrt()
    imshow(QK, title="Attention<br>(W_E + W_pos[-1]) @ W_Q @ W_K.T @ (W_E + W_pos.mean(dim=0)).T", xaxis="Key Token", yaxis="Query Token", renderer=renderer, colorscale=colorscale, **kwargs)
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Query-Side SVD", "Singular Values", "Key-Side SVD"])
    uzmax, vzmax = U.abs().max().item(), Vh.abs().max().item()
    fig.add_trace(go.Heatmap(z=utils.to_numpy(U), colorscale=colorscale, zmin=-uzmax, zmax=uzmax,
                             showscale=False,
                            #  colorbar=dict(x=-0.15, # https://community.plotly.com/t/colorbar-ticks-left-aligned/60473/4
                            #             ticklabelposition='inside',
                            #             ticksuffix='     ',
                            #             ticklabeloverflow='allow',
                            #             tickfont_color='darkslategrey',),
                            hovertemplate="Query: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>",
                            ),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(S.shape[0]), y=utils.to_numpy(S),
                            mode='lines+markers',
                            marker=dict(color='blue'),
                            line=dict(color='blue'),
                            hovertemplate="Singular Value: %{y}<br>Singular Index: %{x}<extra></extra>",
                            ), row=1, col=2)
    fig.add_trace(go.Heatmap(z=utils.to_numpy(Vh.T), colorscale=colorscale, zmin=-vzmax, zmax=vzmax,
                             showscale=False,
                            #  colorbar=dict(x=1.15),
                            hovertemplate="Key: %{y}<br>Singular Index: %{x}<br>Value: %{z}<extra></extra>",
                            ),
                row=1, col=3)
    fig.update_layout(title="Attention SVD") #, margin=dict(l=150, r=150))
    fig.update_yaxes(title_text="Query Token", row=1, col=1)
    fig.update_yaxes(range=[0, None], row=1, col=2)
    fig.update_yaxes(title_text="Key Token", row=1, col=3)
    fig.show(renderer)

    contribution_diff = None
    if include_contribution:
        contribution_diff, _ = compute_singular_contribution(
            QK, description="Attention", colorscale=colorscale, renderer=renderer, singular_value_count=1,
            xaxis='Key Token', yaxis='Query Token',
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Value: %{z}<extra></extra>",
            **kwargs)

    # imshow(U, title="Query-Side SVD", yaxis="Query Token", renderer=renderer, **kwargs)
    # imshow(Vh.T, title="Key-Side SVD", yaxis="Key Token", renderer=renderer, **kwargs)
    # px.line({'singular values': utils.to_numpy(S)}, title="Singular Values of QK Attention").show(renderer)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Size", "Query"])
    fig.add_trace(go.Scatter(x=np.arange(size_direction.shape[0]), y=utils.to_numpy(size_direction),
                            mode='lines+markers',
                            marker=dict(color='blue'),
                            line=dict(color='blue'),
                            hovertemplate="Token: %{x}<br>Size: %{y}<extra></extra>",
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(query_direction.shape[0]), y=utils.to_numpy(query_direction),
                            mode='lines+markers',
                            marker=dict(color='blue'),
                            line=dict(color='blue'),
                            hovertemplate="Token: %{x}<br>Query Value: %{y}<extra></extra>",
                            ), row=1, col=2)
    fig.update_layout(title="Directions in Token Space", showlegend=False)
    fig.show(renderer)

    # px.line({'size direction': utils.to_numpy(size_direction)}, title="size direction in token space").show(renderer)
    # px.line({'query direction': utils.to_numpy(query_direction)}, title="query direction in token space").show(renderer)
    if size_direction_resid is not None: line(size_direction_resid, title="size direction in residual space", renderer=renderer)
    if query_direction_resid is not None: line(query_direction_resid, title="query direction in residual space", renderer=renderer)
    if size_direction_QK is not None: line(size_direction_QK, title="size direction in QK space", renderer=renderer)
    if query_direction_QK is not None: line(query_direction_QK, title="query direction in QK space", renderer=renderer)

    reference_lines = []
    if contribution_diff is not None:
        # we make some reference lines for the plots of size[i+1] - size[i]
        # since we'll eventually multiply these by the singular value and the query direction entry, we want to divide by this product when comparing to values from the non-size-direction contributions
        # we compute the mean and worst-case behavior, and a more fine-grained worst-case adjacent difference
        singular_scale = S[0].item()
        scale_per_query = query_direction * singular_scale
        resid_diffs = contribution_diff[:, :-1] - contribution_diff[:, 1:]
        resid_max_diff = contribution_diff.max().item() - contribution_diff.min().item()
        resid_max_diff_per_query = contribution_diff.max(dim=1).values - contribution_diff.min(dim=1).values
        scale_mean, scale_min = scale_per_query.mean(dim=0).item(), scale_per_query.min().item()
        resid_mean_diff = (contribution_diff[:, :, None, None] - contribution_diff[None, None, :, :]).abs().mean().item()
        resid_mean_diff_per_query = (contribution_diff[:, :, None] - contribution_diff[:, None, :]).abs().mean(dim=(-2, -1))
        reference_lines = [
            ("resid.max - resid.min (worst-case independent query)", resid_max_diff / scale_min),
            ("resid.max - resid.min (average-case independent query)", resid_max_diff / scale_mean),
            ("resid.max - resid.min (worst-case query)", (resid_max_diff_per_query / scale_per_query).max().item()),
            ("(resid[i] - resid[i+1]).max (worst-case independent query)", (resid_diffs / scale_min).max().item()),
            ("(resid[i] - resid[i+1]).max (worst-case query)", (resid_diffs / scale_per_query[:, None]).max().item()),
            ("(resid[i] - resid[i+1]).abs.mean (average-case independent query)", (resid_diffs / scale_mean).abs().mean().item()),
            ("(resid[i] - resid[j]).abs.mean (average-case independent query)", resid_mean_diff / scale_mean),
            ("(resid[i] - resid[j]).abs.mean (average-case query)", (resid_mean_diff_per_query / scale_per_query).abs().mean().item()),
        ]

    size_direction_differences = size_direction[1:] - size_direction[:-1]
    show_fits(size_direction, name='Size Direction', fit_funcs=(fit_func for fit_func in fit_funcs if fit_func is not sigmoid_func),
              do_exclusions=do_exclusions, renderer=renderer)
    show_fits(size_direction_differences, name='Size Direction Δ', reference_lines=reference_lines, fit_funcs=(fit_func for fit_func in delta_fit_funcs if fit_func is not sigmoid_func),
              do_exclusions=do_exclusions, renderer=renderer)

    y_data = size_direction.detach().cpu().numpy()
    x_data = np.linspace(1, len(y_data), len(y_data))

    for fit_func in fit_funcs:
        fit_func_name = fit_func.__name__
        if fit_func_name.endswith("_func"): fit_func_name = fit_func_name[:-len("_func")]

        if fit_func is sigmoid_func:
            # fit to sigmoid
            y_transposed = np.linspace(1, len(x_data), len(x_data))
            initial_params_transposed = [max(y_transposed), 1/np.mean(y_data), np.median(y_data)]

            # Fit the curve with initial parameters

            params_transposed, covariance_transposed = curve_fit(sigmoid_func, y_data, y_transposed, p0=initial_params_transposed, maxfev=10000)

            # Generate predicted y values with parameters
            y_pred_transposed = sigmoid_func(y_data, *params_transposed)
            # Calculating residuals
            residuals = y_transposed - y_pred_transposed

            # Creating subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            fig.suptitle('Fitting a Sigmoid to the Size Vector Components and Residuals Analysis', fontsize=16)

            # Plotting the original data and fitted curve
            axs[0].scatter(y_data, y_transposed, label='Data', color='blue')
            axs[0].plot(y_data, y_pred_transposed, color='red',
                    label=rf'{inv_sigmoid_func.equation(params_transposed)}')
            axs[0].set_xlabel('Component in Normalized Size Vector')
            axs[0].set_ylabel('Input Token')
            axs[0].legend()
            axs[0].grid(True)

            # Plotting residuals
            axs[1].scatter(y_data, residuals, color='green', label='Residuals')
            axs[1].axhline(y=0, color='r', linestyle='--', label='y=0')
            axs[1].set_xlabel('Component in Normalized Size Vector')
            axs[1].set_ylabel('Residual')
            axs[1].legend()
            axs[1].grid(True)

            # Displaying the plots
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # To prevent overlap between suptitle and subplots
            plt.show()


@torch.no_grad()
def find_size_and_query_direction(model: HookedTransformer, plot_heatmaps=False, renderer=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximates the size direction of the model.
    """
    W_pos, W_Q, W_K, W_E = model.W_pos, model.W_Q, model.W_K, model.W_E
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_pos.shape == (n_ctx, d_model), f"W_pos.shape = {W_pos.shape} != {(n_ctx, d_model)} = (n_ctx, d_model)"
    assert W_Q.shape == (1, 1, d_model, d_model), f"W_Q.shape = {W_Q.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
    assert W_K.shape == (1, 1, d_model, d_model), f"W_K.shape = {W_K.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
    assert W_E.shape == (d_vocab, d_model), f"W_E.shape = {W_E.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"

    QK = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_E + W_pos.mean(dim=0)).T
    assert QK.shape == (d_vocab, d_vocab), f"QK.shape = {QK.shape} != {(d_vocab, d_vocab)} = (d_vocab, d_vocab)"

    # take SVD:
    U, S, Vh = torch.linalg.svd(QK)
    # adjust the free parameter of sign
    sign = torch.sign(U[:, 0].mean())
    U, Vh = U * sign, Vh * sign

    # the size direction is the first column of Vh, normalized
    # query direction is the first column of U, normalized
    size_direction, query_direction = Vh[0, :], U[:, 0]
    size_query_singular_value = S[0] * size_direction.norm() * query_direction.norm()
    size_direction, query_direction = size_direction / size_direction.norm(), query_direction / query_direction.norm()

    if plot_heatmaps:
        size_direction_resid, query_direction_resid = size_direction @ W_E + W_pos[-1], query_direction @ W_E + W_pos.mean(dim=0)
        size_direction_QK, query_direction_QK = size_direction_resid @ W_Q[0, 0, :, :], query_direction_resid @ W_K[0, 0, :, :]

        display_size_direction_stats(size_direction, query_direction, QK, U, Vh, S,
                                    # size_direction_resid=size_direction_resid, size_direction_QK=size_direction_QK,
                                    # query_direction_resid=query_direction_resid, query_direction_QK=query_direction_QK,
                                    renderer=renderer, **kwargs)

    return size_direction, query_direction, size_query_singular_value.item()


@torch.no_grad()
def find_size_direction(model: HookedTransformer, **kwargs):
    """
    Approximates the size direction of the model.
    """
    return find_size_and_query_direction(model, **kwargs)[0]

@torch.no_grad()
def find_query_direction(model: HookedTransformer, **kwargs):
    """
    Approximates the query direction of the model.
    """
    return find_size_and_query_direction(model, **kwargs)[1]

# %%
@torch.no_grad()
def find_backwards_attention(model: HookedTransformer):
    W_pos, W_Q, W_K, W_E = model.W_pos, model.W_Q, model.W_K, model.W_E
    d_model, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_vocab, model.cfg.n_ctx
    assert W_pos.shape == (n_ctx, d_model), f"W_pos.shape = {W_pos.shape} != {(n_ctx, d_model)} = (n_ctx, d_model)"
    assert W_Q.shape == (1, 1, d_model, d_model), f"W_Q.shape = {W_Q.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
    assert W_K.shape == (1, 1, d_model, d_model), f"W_K.shape = {W_K.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
    assert W_E.shape == (d_vocab, d_model), f"W_E.shape = {W_E.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"

    QK = (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T @ (W_E + W_pos[:, None, :]).transpose(-1, -2)
    assert QK.shape == (n_ctx, d_vocab, d_vocab), f"QK.shape = {QK.shape} != {(n_ctx, d_vocab, d_vocab)} = (n_ctx, d_vocab, d_vocab)"
    # diffs0 = QK[:, :, :-1].max(dim=0).values - QK[:, :, 1:].min(dim=0).values
    diffs = QK[:, :, :-1] - QK[:, :, 1:].flip(dims=(0,))
    return torch.nonzero(diffs >= 0).squeeze()

# if __name__ == '__main__':
#     from train_max_of_2 import get_model
#     from tqdm.auto import tqdm

#     TRAIN_IF_NECESSARY = False
#     model = get_model(train_if_necessary=TRAIN_IF_NECESSARY)

#     print(find_backwards_attention(model))

# %%
# %%
## %%
# @torch.no_grad()
# def find_size_and_query_direction_by_parts(model: HookedTransformer, plot_heatmaps=False, renderer=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Approximates the size direction of the model.
#     """
#     W_pos, W_Q, W_K, W_E = model.W_pos, model.W_Q, model.W_K, model.W_E
#     d_model, d_head, d_vocab, n_ctx = model.cfg.d_model, model.cfg.d_head, model.cfg.d_vocab, model.cfg.n_ctx
#     assert W_pos.shape == (n_ctx, d_model), f"W_pos.shape = {W_pos.shape} != {(n_ctx, d_model)} = (n_ctx, d_model)"
#     assert W_Q.shape == (1, 1, d_model, d_model), f"W_Q.shape = {W_Q.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
#     assert W_K.shape == (1, 1, d_model, d_model), f"W_K.shape = {W_K.shape} != {(1, 1, d_model, d_model)} = (1, 1, d_model, d_model)"
#     assert W_E.shape == (d_vocab, d_model), f"W_E.shape = {W_E.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"

#     QE, Q, KT, KET = W_E + W_pos[-1], W_Q[0, 0, :, :], W_K[0, 0, :, :].T, (W_E + W_pos.mean(dim=0)).T
#     assert QE.shape == (d_vocab, d_model), f"QE.shape = {QE.shape} != {(d_vocab, d_model)} = (d_vocab, d_model)"
#     assert Q.shape == (d_model, d_head), f"Q.shape = {Q.shape} != {(d_model, d_head)} = (d_model, d_head)"
#     assert KT.shape == (d_head, d_model), f"KT.shape = {KT.shape} != {(d_head, d_model)} = (d_head, d_model)"
#     assert KET.shape == (d_model, d_vocab), f"KET.shape = {KET.shape} != {(d_model, d_vocab)} = (d_model, d_vocab)"

#     # take SVD:
#     analyze_svd(QE, descr="W_E + W_pos[-1]")
#     analyze_svd(Q, descr="W_Q")
#     analyze_svd(KT, descr="W_K.T")
#     analyze_svd(KET, descr="(W_E + W_pos.mean(dim=0)).T")
#     UQE, SQE, VQEh = torch.linalg.svd(QE)
#     UQ, SQ, VQh = torch.linalg.svd(Q)
#     UKT, SKT, VKTh = torch.linalg.svd(KT)
#     UKET, SKET, VKETh = torch.linalg.svd(KET)
#     # # adjust the free parameter of sign
#     # sign = torch.sign(U[:, 0].mean())
#     # U, Vh = U * sign, Vh * sign

#     # # the size direction is the first column of Vh, normalized
#     # size_direction = Vh[0, :]
#     # size_direction = size_direction / size_direction.norm()

#     # # query direction is the first column of U, normalized
#     # query_direction = U[:, 0]
#     # query_direction = query_direction / query_direction.norm()

#     # if plot_heatmaps:
#     #     size_direction_resid, query_direction_resid = size_direction @ W_E + W_pos[-1], query_direction @ W_E + W_pos.mean(dim=0)
#     #     size_direction_QK, query_direction_QK = size_direction_resid @ W_Q[0, 0, :, :], query_direction_resid @ W_K[0, 0, :, :]

#     #     display_size_direction_stats(size_direction, query_direction, QK, U, Vh, S,
#     #                                 # size_direction_resid=size_direction_resid, size_direction_QK=size_direction_QK,
#     #                                 # query_direction_resid=query_direction_resid, query_direction_QK=query_direction_QK,
#     #                                 renderer=renderer, **kwargs)

#     # return size_direction, query_direction


# if __name__ == '__main__':
#     from train_max_of_2 import get_model
#     from tqdm.auto import tqdm

#     TRAIN_IF_NECESSARY = False
#     model = get_model(train_if_necessary=TRAIN_IF_NECESSARY)

# #     find_size_and_query_direction_by_parts(model, plot_heatmaps=True)
# #
#     print(find_size_and_query_direction(model, plot_heatmaps=True, colorscale='Picnic_r'))#, renderer='png')
#     size_direction, query_direction = find_size_and_query_direction(model)
#     W_pos, W_Q, W_K, W_E = model.W_pos, model.W_Q, model.W_K, model.W_E
#     line(query_direction @ (W_E + W_pos[-1]) @ W_Q[0, 0, :, :] @ W_K[0, 0, :, :].T)
# %%
