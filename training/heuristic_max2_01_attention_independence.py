#!/usr/bin/env python
# %%
import gc
from interp_max_utils_heuristic import compute_heuristic_independence_attention_copying, print_independence_attention_copying_stats
from train_max_of_2 import get_model
from tqdm.auto import tqdm


# %%

if __name__ == '__main__':
    TRAIN_IF_NECESSARY = False
    model = get_model(train_if_necessary=TRAIN_IF_NECESSARY)

# %%
if __name__ == '__main__':
    results = compute_heuristic_independence_attention_copying(model, tqdm=tqdm)
    gc.collect()
# %%
if __name__ == '__main__':
    print_independence_attention_copying_stats(model, tqdm=tqdm, results=results)
# %%
