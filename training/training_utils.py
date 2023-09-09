# # Data Generation

# Helper functions for generating data and splitting it into batches for training.

# In[ ]:

from typing import List, Any, Iterable
import numpy as np
from transformer_lens import HookedTransformer
import torch
import itertools

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    n_digits,
    sequence_length=2,
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
  data = generate_all_sequences(n_digits=n_digits, sequence_length=sequence_length)

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




