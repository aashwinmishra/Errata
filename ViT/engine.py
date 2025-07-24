"""
Functions to train and validate ViTs for image classification tasks.
"""
import torch
import torch.nn as nn


def train_step(model:torch.nn.Module,
               dl: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer,
               metric_fn: callable,
               device: torch.device) -> dict:
  """
  Executes an epoch of training of the model on the dataloader.

  Args:
    model: Model to be trained.
    dl: DataLoader with training data.
    loss_fn: Differentiable criterion for gradients.
    opt: Optimizer.
    metric_fn: Metric function to evaulate model.
    device: torch device where model is stored.

  Returns:
    dictionary with train_loss and train_metric
  """
  model.train()
  losses = 0.0
  metrics = 0.0
  batch_count = 0
  for inputs, labels in dl:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    opt.zero_grad()
    loss = loss_fn(outputs, labels)
    loss.backward()
    opt.step()
    losses += loss.detach().item()
    metrics += metric_fn(outputs, labels)
    batch_count += 1

  return {"train_loss": losses / batch_count,
          "train_metric": metrics / batch_count}


def val_step(model: torch.nn.Module,
             dl: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             metric_fn: callable,
             device: torch.device) -> dict:
  """
  Executes an epoch of evaluation of the model on the dataloader.

  Args:
    model: Model to be evaluated.
    dl: DataLoader with validation data.
    loss_fn: Differentiable criterion.
    metric_fn: Metric function to evaulate model.
    device: torch device where model is stored.

  Returns:
    dictionary with val_loss and val_metric
  """
  model.eval()
  with torch.inference_mode():
    losses = 0.0
    metrics = 0.0
    batch_count = 0
    for inputs, labels in dl:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      losses += loss_fn(outputs, labels).item()
      metrics += metric_fn(outputs, labels)
      batch_count += 1
    return {"val_loss": losses / batch_count,
            "val_metric": metrics / batch_count}


def train(model: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          val_dl: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          opt: torch.optim.Optimizer,
          metric_fn: callable,
          device: torch.device,
          num_epochs: int) -> dict:
  """
  Trains and evaluates model on dataloaders for given epochs.

  Args:
    model: Model to be trained and evaluated.
    train_dl: DataLoader for training dataset.
    val_dl: DataLoader for validation/testing dataset.
    loss_fn: Differentiable criterion to train model
    opt: Optimizer.
    metric_fn: Evaluation metric.
    device: torch device where model is stored.
    num_epochs: Number of epochs to train for.

  Returns:
    dictionary with train and validation losses and metric histories as lists.
  """
  train_loss, train_metric, val_loss, val_metric = [], [], [], []
  for epoch in range(num_epochs):
    train_history = train_step(model, train_dl, loss_fn, opt, metric_fn, device)
    val_history = val_step(model, val_dl, loss_fn, metric_fn, device)
    train_loss.append(train_history["train_loss"])
    train_metric.append(train_history["train_metric"])
    val_loss.append(val_history["val_loss"])
    val_metric.append(val_history["val_metric"])
    print(f"Epoch: {epoch+1}\t"
    f"Train Loss: {train_loss[-1]:.4f} Train Metric: {train_metric[-1]:.4f}\t"
    f"Val Loss: {val_loss[-1]:.4f} Val Metric: {val_metric[-1]:.4f}"
    )

