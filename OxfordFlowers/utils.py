import torch


def get_devices():
  return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seeds(seed: int=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def accuracy(outputs: torch.tensor, labels: torch.tensor)->float:
  preds = torch.argmax(outputs, dim=-1) #[bs, classes] -> [bs,]
  return (preds == labels).sum().item() / len(labels)


def save_model(model, file_name, save_dir):
  if file_name.endswith(".pt"):
    file_name += ".pt"
  path = save_dir + file_name
  torch.save(model.state_dict(), path)

