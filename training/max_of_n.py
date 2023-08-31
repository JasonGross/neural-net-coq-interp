import numpy as np
import torch
from transformer_lens import HookedTransformer
from training_utils import get_data, make_generator_from_data, large_data_gen
import tqdm.auto as tqdm



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


def train_model(
    model:HookedTransformer,
    n_epochs=100,
    batch_size=128,
    batches_per_epoch=10,
    force_adjacent=False,
    use_complete_data=True,
    device=DEVICE
  ):
  lr = 1e-3
  betas = (.9, .999)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
  n_digits, sequence_length = model.cfg.d_vocab, model.cfg.n_ctx
  train_losses = []

  if use_complete_data:
    data_train, data_test = get_data(n_digits=n_digits, sequence_length=sequence_length, force_adjacent=force_adjacent)
    train_data_gen_gen = lambda: make_generator_from_data(data_train, batch_size=batch_size)
  else:
    train_data_gen = large_data_gen(n_digits=n_digits, sequence_length=sequence_length, batch_size=batch_size, context="train", device=device)
    test_data_gen = large_data_gen(n_digits=n_digits, sequence_length=sequence_length, batch_size=batch_size * 20, context="test")
    data_test = next(test_data_gen)


  for epoch in tqdm.tqdm(range(n_epochs)):
    train_data_gen = train_data_gen_gen()
    epoch_losses = []
    for _ in range(batches_per_epoch):
      tokens = next(train_data_gen)
      logits = model(tokens)
      losses = loss_fn(logits, tokens, return_per_token=True)
      losses.mean().backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_losses.extend(losses.detach())

    train_losses.append(np.mean(epoch_losses))

    if epoch % 10 == 0:
      print(f'Epoch {epoch} train loss: {train_losses[-1]}')

  model.eval()
  logits = model(data_test)
  acc = acc_fn(logits, data_test)

  print(f"Test accuracy after training: {acc}")

  return train_losses