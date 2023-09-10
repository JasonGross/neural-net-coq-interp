# # Data Generation

# Helper functions for generating data and splitting it into batches for training.

# In[ ]:

import datetime
import os, os.path
from pathlib import Path
from typing import List, Any, Iterable, Optional
import numpy as np
import tqdm
from transformer_lens import HookedTransformer
import torch
import itertools

import wandb

# In[ ]:
def default_device(deterministic: bool = False) -> str:
   return "cuda" if torch.cuda.is_available() and not deterministic else "cpu"

# In[ ]:

def in_colab() -> bool:
    """
    Returns True if running in Google Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except:
        return False

# In[ ]:

def get_pth_base_path(save_in_google_drive: bool = False, create: bool = True) -> Path:
    """
    Returns the base path for saving models. If `save_in_google_drive` is True, returns the path to the Google Drive
    folder where models are saved. Otherwise, returns the path to the local folder where models are saved.
    """
    if in_colab():
        if save_in_google_drive:
            from google.colab import drive
            drive.mount('/content/drive/')
            pth_base_path = Path('/content/drive/MyDrive/Colab Notebooks/')
        else:
            pth_base_path = Path("/workspace/_scratch/")
    else:
        pth_base_path = Path(os.getcwd())

    pth_base_path = pth_base_path / 'trained-models'

    if create and not os.path.exists(pth_base_path):
        os.makedirs(pth_base_path)

    return pth_base_path

# In[ ]:

def generate_all_sequences(n_digits, sequence_length=2):
  data = list(itertools.product(range(n_digits), repeat=sequence_length))
  data = torch.tensor(data)
  return data

# In[ ]:

def compute_all_tokens(model: HookedTransformer):
    return generate_all_sequences(n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx)

# In[ ]:

def shuffle_data(data):
  indices = np.array(range(len(data)))
  np.random.shuffle(indices)
  data = data[indices]
  return data

# In[ ]:

def make_testset_trainset(
    model: HookedTransformer,
    training_ratio=0.7,
    force_adjacent=False):
  """
  Generate a train and test set of tuples containing `sequence_length` integers with values 0 <= n < n_digits.

  Args:
      sequence_length (int): The length of each tuple in the dataset.
      n_digits (int): The number of possible values for each element in the tuple.
      training_ratio (float): The ratio of the size of the training set to the full dataset.
      force_adjacent (bool): Whether to make training adversarial (force to include all (x, x +- 1))

  Returns:
      Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]: A tuple containing the training set and test set.
          The training set contains `training_ratio` percent of the full dataset, while the test set contains the
          remaining data. Each set is a list of tuples containing `sequence_length` integers with values 0 <= n < n_digits.
          The tuples have been shuffled before being split into the train and test sets.
  """
  data = compute_all_tokens(model)

  data = shuffle_data(data)

  if force_adjacent:
    idxs = (data[:,0] - data[:,1]).abs() == 1
    data, extra_data = data[~idxs], data[idxs]
    data = torch.cat([extra_data, data], dim=0)

  split_idx = int(len(data) * training_ratio)

  data_train = data[:split_idx]
  data_test = data[split_idx:]

  if force_adjacent:
    data_train = shuffle_data(data_train)
    data_test = shuffle_data(data_test)

  return data_train, data_test

# In[ ]:

def make_generator_from_data(data: List[Any], batch_size: int = 128) -> Iterable[List[Any]]:
  """
  Returns a generator that yields slices of length `batch_size` from a list.

  Args:
      data: The input list to be split into batches.
      batch_size: The size of each batch.

  Yields:
      A slice of the input list of length `batch_size`. The final slice may be shorter if the
      length of the list is not evenly divisible by `batch_size`.
  """
  data = shuffle_data(data)
  for i in range(0,len(data), batch_size):
    yield data[i:i+batch_size]

# In[ ]:

def make_wandb_config(
    model:HookedTransformer,
    optimizer_kwargs: dict,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    adjacent_fraction=0,
    use_complete_data=True,
    device=None,
    **kwargs):
  return {
      'model.cfg':model.cfg.to_dict(),
      'optimizer.cfg':optimizer_kwargs,
      'n_epochs':n_epochs,
      'batch_size':batch_size,
      'batches_per_epoch':batches_per_epoch,
      'adjacent_fraction':adjacent_fraction,
      'use_complete_data':use_complete_data,
      'device':device,
    }

def load_model(model: HookedTransformer, model_pth_path: str):
  try:
    cached_data = torch.load(model_pth_path)
    model.load_state_dict(cached_data['model'])
    #model_checkpoints = cached_data["checkpoints"]
    #checkpoint_epochs = cached_data["checkpoint_epochs"]
    #test_losses = cached_data['test_losses']
    train_losses = cached_data['train_losses']
    #train_indices = cached_data["train_indices"]
    #test_indices = cached_data["test_indices"]
    return train_losses, model_pth_path
  except Exception as e:
    print(f'Could not load model from {model_pth_path}:\n', e)

def train_or_load_model(
      model_name:str,
      model:HookedTransformer,
      loss_fn,
      acc_fn,
      train_data_gen_maybe_lambda,
      data_test,
      n_epochs=100,
      batches_per_epoch=10,
      device=None,
      wandb_project=None,
      save_model=True,
      model_pth_path=None,
      deterministic: bool = False,
      optimizer=torch.optim.Adam,
      optimizer_kwargs={'lr':1e-3, 'betas': (.9, .999)},
      train_data_gen_is_lambda: bool = False,
      loss_fn_kwargs={'return_per_token':True},
      print_every: Optional[int] = 10,
      log_acc: bool = False,
      force_train: bool = False,
      overwrite_data: bool = False,
      model_description: str = "trained model",
      wandb_entity:str = 'tkwa-team',
      fail_if_cant_load: bool = False,
      save_in_google_drive: bool = False,
      **kwargs, # kwargs for **locals() below
  ):
  if force_train and fail_if_cant_load: raise ValueError(f"force_train is {force_train} and fail_if_cant_load is {fail_if_cant_load}")
  if device is None: device = default_device(deterministic=deterministic)

  pth_base_path = get_pth_base_path(save_in_google_drive=save_in_google_drive, create=True)
  if model_pth_path is None:
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_pth_path = pth_base_path / f'{model_name}-{model.cfg.n_ctx}-epochs-{n_epochs}-{datetime_str}.pth'

  if not force_train and os.path.exists(model_pth_path):
    res = load_model(model, model_pth_path)
    if res is not None: return res

  if wandb_project is not None:
    wandb_model_path = f"{wandb_entity}/{wandb_project}/{model_name}:latest"
    if not force_train:
      model_dir = None
      try:
        api = wandb.Api()
        model_at = api.artifact(wandb_model_path)
        model_dir = Path(model_at.download())
      except Exception as e:
        print(f'Could not load model {wandb_model_path} from wandb:\n', e)
      if model_dir is not None:
        for model_path in model_dir.glob('*.pth'):
          res = load_model(model, model_path)
          if res is not None: return res

  assert not fail_if_cant_load, f"Couldn't load model from {model_pth_path}{f' or wandb ({wandb_model_path})' if wandb_project is not None else ''}, and fail_if_cant_load is {fail_if_cant_load}"

  if wandb_project is not None:
    config_info = make_wandb_config(**locals())
    run = wandb.init(project=wandb_project, entity=wandb_entity, config=config_info, job_type="train")

  optimizer = optimizer(model.parameters(), **optimizer_kwargs)
  train_data_gen_lambda = (lambda: train_data_gen_maybe_lambda) if not train_data_gen_is_lambda else train_data_gen_maybe_lambda

  train_losses = []

  pbar = tqdm.tqdm(range(n_epochs))
  for epoch in pbar:
    train_data_gen = train_data_gen_lambda()
    epoch_losses = []
    for _ in range(batches_per_epoch):
      tokens = next(train_data_gen)
      logits = model(tokens)
      losses = loss_fn(logits, tokens, **loss_fn_kwargs)
      losses.mean().backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_losses.extend(losses.detach().cpu().numpy())

    train_losses.append(np.mean(epoch_losses))

    if print_every and epoch % print_every == 0:
      pbar.set_description(f'Epoch {epoch} train loss: {train_losses[-1]:.5e}')

    if wandb_project is not None:
      log_data = {'train_loss': train_losses[-1]}
      if log_acc: log_data['train_acc'] = acc_fn(model(tokens), tokens)
      wandb.log(log_data)

  model.eval()
  logits = model(data_test)
  acc = acc_fn(logits, data_test)

  print(f"Test accuracy after training: {acc}")

  if save_model:
    data = {
       "model":model.state_dict(),
       "config": model.cfg,
       "train_losses": train_losses,
       }
    if overwrite_data or not os.path.exists(model_pth_path):
      torch.save(data, model_pth_path)
      if wandb_project is not None:
        trained_model_artifact = wandb.Artifact(
            model_name, type="model", description=model_description, metadata=model.cfg.to_dict())
        trained_model_artifact.add_file(model_pth_path)
        run.log_artifact(trained_model_artifact)
    elif wandb_project is not None:
      print(f"Warning: {model_pth_path} already exists, saving model directly")
      run.log_artifact(data)

  if wandb_project is not None:
    run.finish()

  return train_losses, model_pth_path
