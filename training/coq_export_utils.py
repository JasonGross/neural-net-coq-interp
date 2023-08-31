# %%
import dataclasses
import numpy as np
import torch
import transformer_lens
from transformer_lens import HookedTransformer


def strify(v, ty=None, description=None):
    tymap = {'int': 'Z', 'float':'Q', 'str':'string', 'bool':'bool'}
    if v is None:
        assert ty is not None
        assert ty.startswith('Optional[')
        ty = ty[len('Optional['):-1]
        resty = '%s'
        while ty.startswith('List['):
            resty = '(list %s)' % resty
            ty = ty[len('List['):-1]
        return f'@None {resty % tymap[ty]}'
    if isinstance(v, bool): return 'true' if v else 'false'
    if isinstance(v, torch.Tensor): return strify(v.detach().numpy())
    if isinstance(v, np.ndarray):
        if len(v.shape) > 1:
            return '[' + "\n;".join(map(strify, v)) + ']'
        else:
            return '[' + ";".join(map(strify, v)) + ']'
    if isinstance(v, str): return '"' + repr(v)[1:-1] + '"'
    if isinstance(v, list):
        if isinstance(v[0], list):
            return '[' + "\n;".join(map(strify, v)) + ']'
        else:
            return '[' + ";".join(map(strify, v)) + ']'
    if isinstance(v, transformer_lens.FactoredMatrix): return strify(v.AB)
    if any(isinstance(v, ty) for ty in (np.float64, float)): return v.hex()
    if isinstance(v, np.float32): return float(v).hex()
    if isinstance(v, torch.dtype): return f'"{str(v)}"'
    if any(isinstance(v, ty) for ty in (int, )): return f'{v}%Z' if v < 0 else f'{v}%N'
    raise ValueError(f"unknown type {type(v)}" + (f" ({description})" if description is not None else ""))


# # Exporting the Simpler Model to Coq

# In[ ]:



def coq_export_params(model: HookedTransformer):
    print('Module cfg.')
    for f in dataclasses.fields(model.cfg):
        val = dataclasses.asdict(model.cfg)[f.name]
        ty = f.type
        if f.name == 'attn_types' and ty == 'Optional[List]': ty = 'Optional[List[str]]'
        print(f'  Definition {f.name} := {strify(val, ty=ty, description=f.name)}.')
    print('End cfg.')

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
        print(f'Definition {name} :=')
        print(strify(getattr(model, name)))
        print('.')

    
    for layer, block in enumerate(model.blocks):
        for module, names in (('ln1', ('b', 'w')), ('attn', ('W_Q', 'W_K', 'W_O', 'W_V', 'b_Q', 'b_K', 'b_O', 'b_V')), ):
            if hasattr(block, module):
                for name in names:
                    if hasattr(getattr(block, module), name):
                        print(f'Definition L{layer}_{module}_{name} :=')
                        print(strify(getattr(getattr(block, module), name)))
                        print('.')

    for module, names in (('ln_final', ('b', 'w')), ):
        if hasattr(model, module):
            for name in names:
                print(f'Definition {module}_{name} :=')
                print(strify(getattr(getattr(model, module), name)))
                print('.')
# %%
