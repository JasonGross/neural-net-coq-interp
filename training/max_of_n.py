import numpy as np
import torch
from transformer_lens import HookedTransformer
from training_utils import get_data, make_generator_from_data
import tqdm.auto as tqdm
import wandb



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def loss_fn(
    logits, # [batch, pos, d_vocab]
    tokens, # [batch, pos]
    return_per_token=False
  ):
  logits = logits[:, -1, :]
  true_maximum = torch.max(tokens, dim=1)[0]
  log_probs = logits.log_softmax(-1)
  correct_log_probs = log_probs.gather(-1, true_maximum.unsqueeze(-1))
  if return_per_token:
    return -correct_log_probs.squeeze()
  return -correct_log_probs.mean()


# In[ ]:


def acc_fn(
    logits, # [batch, pos, d_vocab]
    tokens, # [batch, pos]
    return_per_token=False
  ):
  pred_logits = logits[:, -1, :]
  pred_tokens = torch.argmax(pred_logits, dim=1)
  true_maximum = torch.max(tokens, dim=1)[0]
  if return_per_token:
    return (pred_tokens == true_maximum).float()
  return (pred_tokens == true_maximum).float().mean().item()


def large_data_gen(n_digits, sequence_length=6, batch_size=128, context="train", device=DEVICE, adjacent_fraction=0):
  if context == "train":
    seed = 5
  else:
    seed = 6
  torch.manual_seed(seed)
  while True:
    result = torch.randint(0, n_digits, (batch_size, sequence_length)).to(device)
    if adjacent_fraction == 0: yield result
    else:
      adjacent = torch.randint(0, n_digits, (batch_size,))
      adjacent = adjacent.unsqueeze(1).repeat(1, sequence_length)
      # in half the rows, replace a random column with n+1
      rows_to_change = torch.randperm(batch_size)[:batch_size // 2]
      cols_to_change = torch.randint(0, sequence_length, (batch_size // 2,))
      adjacent[rows_to_change, cols_to_change] += 1
      adjacent %= n_digits
      adjacent = adjacent.to(device)
      mask = torch.rand(batch_size) < adjacent_fraction
      result[mask] = adjacent[mask]
      yield result


def train_model(
    model:HookedTransformer,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    adjacent_fraction=0,
    use_complete_data=True,
    device=DEVICE,
    use_wandb=False,
  ):
  lr = 1e-3
  betas = (.9, .999)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
  n_digits, sequence_length = model.cfg.d_vocab, model.cfg.n_ctx
  train_losses = []

  if use_complete_data:
    data_train, data_test = get_data(n_digits=n_digits, sequence_length=sequence_length, force_adjacent=adjacent_fraction > 0)
    train_data_gen_gen = lambda: make_generator_from_data(data_train, batch_size=batch_size)
  else:
    train_data_gen = large_data_gen(n_digits=n_digits, sequence_length=sequence_length, batch_size=batch_size, context="train", device=device, adjacent_fraction=adjacent_fraction)
    test_data_gen = large_data_gen(n_digits=n_digits, sequence_length=sequence_length, batch_size=batch_size * 20, context="test", adjacent_fraction=adjacent_fraction)
    data_test = next(test_data_gen)


  for epoch in tqdm.tqdm(range(n_epochs)):
    if use_complete_data:
      train_data_gen = train_data_gen_gen()
    epoch_losses = []
    for _ in range(batches_per_epoch):
      tokens = next(train_data_gen)
      logits = model(tokens)
      losses = loss_fn(logits, tokens, return_per_token=True)
      losses.mean().backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_losses.extend(losses.detach().cpu().numpy())

    train_losses.append(np.mean(epoch_losses))

    if epoch % 10 == 0:
      print(f'Epoch {epoch} train loss: {train_losses[-1]}')

    if use_wandb:
      wandb.log({'train_loss': train_losses[-1]})

  model.eval()
  logits = model(data_test)
  acc = acc_fn(logits, data_test)

  print(f"Test accuracy after training: {acc}")


  return train_losses