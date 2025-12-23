import torch


def train_step(model, 
               train_dl, 
               loss_fn, 
               opt, 
               metric_fn, 
               device):
  model.train()
  losses, metrics = 0.0, 0.0
  for images, labels in train_dl:
    images, labels = images.to(device), labels.to(device)
    opt.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    opt.step()
    losses += loss.item() * len(labels)
    metrics += metric_fn(outputs, labels) * len(labels)
  return losses / len(train_dl.dataset), metrics / len(train_dl.dataset) #Loss, metrics per sample


def val_step(model, 
             val_dl, 
             loss_fn, 
             metric_fn, 
             device):
  model.eval()
  losses, metrics = 0.0, 0.0
  with torch.no_grad():
    for images, labels in val_dl:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      losses += loss_fn(outputs, labels).item() * len(labels)
      metrics += metric_fn(outputs, labels) * len(labels)
  return losses / len(val_dl.dataset), metrics / len(val_dl.dataset)


def train(model, 
          train_dl, 
          val_dl, 
          loss_fn, 
          opt, 
          metric_fn, 
          device, 
          num_epochs):
  train_losses, val_losses, train_metrics, val_metrics = [], [], [], []
  for epoch in range(num_epochs):
    train_loss, train_metric = train_step(model, train_dl, loss_fn, opt, metric_fn, device)
    val_loss, val_metric = val_step(model, val_dl, loss_fn, metric_fn, device)
    train_losses.append(train_loss)
    train_metrics.append(train_metric)
    val_losses.append(val_loss)
    val_metrics.append(val_metric)
    print(f"Epoch: {epoch+1} Train Loss: {train_loss:.5f} Train Metric: {train_metric:.5f} Val Loss: {val_loss:.5f} Val Metric: {val_metric:.5f}")
  return {"train_loss": train_losses, 
          "train_metric": train_metrics, 
          "val_loss": val_losses, 
          "val_metric": val_metrics}

