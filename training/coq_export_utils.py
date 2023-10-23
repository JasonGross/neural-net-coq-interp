# %%
import dataclasses
from typing import Iterable
import numpy as np
import torch
from transformer_lens import HookedTransformer, FactoredMatrix


def strify(v, ty=None, description=None, parens_if_space=False):
    tymap = {'int': 'Z', 'float':'Q', 'str':'string', 'bool':'bool', 'NormalizationType':'NormalizationType', 'ActivationKind':'ActivationKind'}
    def wrap_parens(s):
        return f'({s})' if parens_if_space else s
    if v is None:
        assert ty is not None
        assert ty.startswith('Optional[')
        ty = ty[len('Optional['):-1]
        resty = '%s'
        while ty.startswith('List['):
            resty = '(list %s)' % resty
            ty = ty[len('List['):-1]
        return wrap_parens(f'@None {resty % tymap[ty]}')
    if ty is not None and ty.startswith('Optional['):
        ty = ty[len('Optional['):-1]
        return wrap_parens(f'Some {strify(v, ty=ty, description=description, parens_if_space=True)}')
    if isinstance(v, bool): return 'true' if v else 'false'
    if isinstance(v, str) and ty in ('NormalizationType', 'ActivationKind'): return v
    if isinstance(v, str): return '"' + repr(v)[1:-1] + '"'
    if isinstance(v, torch.Tensor): return strify(v.detach().cpu().numpy(), ty=ty, description=description, parens_if_space=parens_if_space)
    if isinstance(v, np.ndarray):
        if len(v.shape) > 1:
            return '[' + "\n;".join(map(strify, v)) + ']'
        else:
            return '[' + ";".join(map(strify, v)) + ']'
    if isinstance(v, list):
        if isinstance(v[0], list):
            return '[' + "\n;".join(map(strify, v)) + ']'
        else:
            return '[' + ";".join(map(strify, v)) + ']'
    if isinstance(v, FactoredMatrix): return strify(v.AB, ty=ty, description=description, parens_if_space=parens_if_space)
    if any(isinstance(v, ty) for ty in (np.float64, float)): return v.hex()
    if isinstance(v, np.float32): return strify(float(v), ty=ty, description=description, parens_if_space=parens_if_space)
    if isinstance(v, torch.dtype): return f'"{str(v)}"'
    if any(isinstance(v, ty) for ty in (int, )): return wrap_parens(f'{v}%Z') if v < 0 else f'{v}%N'
    raise ValueError(f"unknown type {type(v)}" + (f" ({description})" if description is not None else ""))


# # Exporting the Simpler Model to Coq

# In[ ]:



def coq_export_params_lines(model: HookedTransformer) -> Iterable[str]:
    yield 'Module cfg <: CommonConfig.'
    for f in dataclasses.fields(model.cfg):
        val = dataclasses.asdict(model.cfg)[f.name]
        ty = f.type
        for (name, expty, newty) in [('attn_types', 'Optional[List]', 'Optional[List[str]]'),
                                    ('normalization_type', 'Optional[str]', 'Optional[NormalizationType]'),
                                    ('act_fn', 'Optional[str]', 'Optional[ActivationKind]')]:
            if f.name == name:
                assert ty == expty, f'{f.name}.ty == {ty} != {expty}'
                ty = newty
                break
        yield f'  Definition {f.name} := {strify(val, ty=ty, description=f.name)}.'
    yield 'End cfg.'

    for name in (#'OV',
    #'QK',
    #'T_destination',
    'W_E',
    #'W_E_pos',
    'W_K',
    'W_O',
    'W_Q',
    'W_U',
    'W_V',
    #'W_in',
    #'W_out',
    'W_pos', 'b_K',
    'b_O',
    'b_Q',
    'b_U',
    'b_V',):
    #'b_in',
    #'b_out'):
        yield f'Definition {name} :='
        yield strify(getattr(model, name))
        yield '.'


    for layer, block in enumerate(model.blocks):
        for module, names in (('ln1', ('b', 'w')), ('attn', ('W_Q', 'W_K', 'W_O', 'W_V', 'b_Q', 'b_K', 'b_O', 'b_V')), ):
            if hasattr(block, module):
                for name in names:
                    if hasattr(getattr(block, module), name):
                        yield f'Definition L{layer}_{module}_{name} :='
                        yield strify(getattr(getattr(block, module), name))
                        yield '.'

    for module, names in (('ln_final', ('b', 'w')), ):
        if hasattr(model, module):
            for name in names:
                yield f'Definition {module}_{name} :='
                yield strify(getattr(getattr(model, module), name))
                yield '.'
# %%
def coq_export_params(model: HookedTransformer):
    return '\n'.join(coq_export_params_lines(model))
