# # Data Generation

# Helper functions for generating data and splitting it into batches for training.

# In[ ]:


import numpy as np
from transformer_lens import HookedTransformer


import torch


import itertools

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_data(n_digits, sequence_length=2):
  data = list(itertools.product(range(n_digits), repeat=sequence_length))
  data = torch.tensor(data)
  return data


# In[ ]:

# somewhat redundant with generate_data
def compute_all_tokens(model: HookedTransformer):
    return generate_data(n_digits=model.cfg.d_vocab, sequence_length=model.cfg.n_ctx)
    # d_vocab, n_ctx = model.cfg.d_vocab, model.cfg.n_ctx
    # one_tokens = torch.arange(0, d_vocab)
    # # take n_ctx cartesian_prod copies of one_tokens
    # all_tokens = torch.cartesian_prod(*[one_tokens for _ in range(n_ctx)])

    # # Reshape the tensor to the required shape (d_vocab^n_ctx, n_ctx)
    # all_tokens = all_tokens.reshape(d_vocab**n_ctx, n_ctx)
    # return all_tokens


# In[ ]:


def shuffle_data(data):
  indices = np.array(range(len(data)))
  np.random.shuffle(indices)
  data = data[indices]
  return data


# In[ ]:


def make_generator_from_data(data, batch_size=128):
  """
  Returns a generator that yields slices of length `batch_size` from a list.

  Args:
      data (List[Any]): The input list to be split into batches.
      batch_size (int): The size of each batch.

  Yields:
      List[Any]: A slice of the input list of length `batch_size`. The final slice may be shorter if the
      length of the list is not evenly divisible by `batch_size`.
  """
  data = shuffle_data(data)
  for i in range(0,len(data), batch_size):
    yield data[i:i+batch_size]


# In[ ]:


def get_data(
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
  data = generate_data(n_digits=n_digits, sequence_length=sequence_length)

  data = shuffle_data(data)

  split_idx = int(len(data) * training_ratio)

  if force_adjacent:
    idxs = (data[:,0] - data[:,1]).abs() == 1
    data, extra_data = data[~idxs], data[idxs]
    data = torch.cat([extra_data, data], dim=0)

  data_train = data[:split_idx]
  data_test = data[split_idx:]

  if force_adjacent:
    data_train = shuffle_data(data_train)
    data_test = shuffle_data(data_test)

  return data_train, data_test


# In[ ]:


def large_data_gen(n_digits, sequence_length=6, batch_size=128, context="train", device=DEVICE):
  if context == "train":
    seed = 5
  else:
    seed = 6
  torch.manual_seed(seed)
  while True:
    yield torch.randint(0, n_digits, (batch_size, sequence_length)).to(device)


# # Loss Function

# The loss is the cross entropy between the prediction for the final token and the true maximum of the sequence.

# In[ ]:


