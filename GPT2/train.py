import torch
import torch.nn as nn
import tiktoken
from utils import generate_text_simple, text_to_token_ids

def train_step(model, 
               train_loader, 
               loss_func, 
               optimizer, 
               device):
  losses = []
  model.train()
  for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = loss_func(logits.flatten(0,1), y.flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().cpu().item())
  return torch.tensor(losses).mean().item()


def val_step(model,
             val_loader,
             loss_func,
             device):
  losses = []
  model.eval()
  for x, y in val_loader:
    x, y = x.to(device), y.to(device)
    with torch.inference_mode():
      logits = model(x)
    loss = loss_func(logits.flatten(0,1), y.flatten())
    losses.append(loss.detach().cpu().item())
  return torch.tensor(losses).mean().item()


def train(model, 
          train_loader, 
          val_loader, 
          loss_func, 
          optimizer, 
          device, 
          num_epochs,
          text,
          max_new_tokens, 
          context_size, 
          tokenizer
          ):
  ids = text_to_token_ids(text, tokenizer)
  train_losses, val_losses = [], []
  for epoch in range(num_epochs):
    train_loss = train_step(model, train_loader, loss_func, optimizer, device)
    val_loss = val_step(model, val_loader, loss_func, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch: {epoch+1} Train Loss: {train_loss} Val Loss: {val_loss}")
    generate_text_simple(model, ids, max_new_tokens, context_size, tokenizer)

  return train_losses, val_losses

