# %%
import sys
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm.auto as tqdm
from training_utils import train_or_load_model, make_testset_trainset, make_generator_from_data
from max_of_n import acc_fn, loss_fn, large_data_gen
from training_utils import make_testset_trainset, make_generator_from_data, DEFAULT_WANDB_ENTITY
from coq_export_utils import coq_export_params

# %%

DETERMINISTIC = False # @param
DEVICE = "cuda" if torch.cuda.is_available() and not DETERMINISTIC else "cpu"
N_LAYERS = 1 # @param
N_HEADS = 1 # @param
D_MODEL = 32 # @param
D_HEAD = 32 # @param
D_MLP = None # @param
D_VOCAB = 64 # @param
SEED = 123 # @param
N_EPOCHS = 50000 # @param
N_CTX = 5 # @param
ADJACENT_FRACTION=0.3 # @param
BATCH_SIZE = 128 # @param
FAIL_IF_CANT_LOAD = '--fail-if-cant-load' in sys.argv[1:] # @param

ALWAYS_TRAIN_MODEL = False # @param
SAVE_IN_GOOGLE_DRIVE = False # @param
OVERWRITE_DATA = False # @param
TRAIN_MODEL_IF_CANT_LOAD = True # @param


# %%

simpler_cfg = HookedTransformerConfig(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_head=D_HEAD,
    n_ctx=N_CTX,
    d_vocab=D_VOCAB,
    seed=SEED,
    device=DEVICE,
    attn_only=True,
    normalization_type=None,
)
# %%

model = HookedTransformer(simpler_cfg).to(DEVICE)

for name, param in model.named_parameters():
    if "b_" in name:
        param.requires_grad = False
        
model_is_trained = False


# %%

def train(fail_if_cant_load=FAIL_IF_CANT_LOAD, train_if_cant_load=TRAIN_MODEL_IF_CANT_LOAD, overwrite_data=OVERWRITE_DATA,
          always_train_model=ALWAYS_TRAIN_MODEL,
          wandb_entity=DEFAULT_WANDB_ENTITY,
          save_in_google_drive=SAVE_IN_GOOGLE_DRIVE):
    
    global model_is_trained
    train_data_gen = large_data_gen(n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx, batch_size=BATCH_SIZE, context="train", device=DEVICE, adjacent_fraction=ADJACENT_FRACTION)
    test_data_gen = large_data_gen(n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx, batch_size=BATCH_SIZE * 20, context="test", adjacent_fraction=ADJACENT_FRACTION)
    data_test = next(test_data_gen)

    training_losses, model_pth_path = train_or_load_model(
        f'neural-net-coq-interp-max-{model.cfg.n_ctx}-epochs-{N_EPOCHS}',
        model,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        train_data_gen_maybe_lambda=train_data_gen,
        train_data_gen_is_lambda=False,
        data_test=data_test,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        adjacent_fraction=1,
        use_complete_data=True,
        batches_per_epoch=10,
        wandb_project=f'neural-net-coq-interp-max-{model.cfg.n_ctx}-epochs-{N_EPOCHS}',
        deterministic=DETERMINISTIC,
        save_in_google_drive=save_in_google_drive,
        overwrite_data=overwrite_data,
        train_model_if_cant_load=train_if_cant_load,
        model_description=f"trained max of {model.cfg.n_ctx} model on {DEVICE}",
        save_model=True,
        force_train=always_train_model,
        wandb_entity=wandb_entity,
        fail_if_cant_load=fail_if_cant_load,
    )
    
    model_is_trained = True
    return training_losses, model_pth_path

# %%

def get_model(train_if_necessary = False,  **kwargs):
    
    train(fail_if_cant_load = not train_if_necessary, train_if_cant_load = train_if_necessary, **kwargs)
    
    return model
    

# %%
if __name__ == '__main__':
    training_losses, model_pth_path = train()
    print(coq_export_params(model))

# %%
