import torch 
import tiktoken 
import os


def get_devices() -> torch.device:
  """
  Returns gpu device if available, else cpu
  """
  return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seeds(seed: int = 42):
  """
  Sets torch seeds to ensure reproducability.
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)


def save_model(model_dir: str, 
               model_name: str, 
               model: torch.nn.Module):
  """
  Saves pytorch model in model_dir with model_name.
  Args:
    model_dir: Directory to save model in.
    model_name: name of file to store model.
    model: model to be saved.
  Returns:
    None
  """
  os.makedirs(model_dir, exist_ok=True)
  if not model_name.endswith("pt"):
    model_name += ".pt"
  torch.save(model.state_dict(), os.path.join(model_dir, model_name))


def generate_text_simple(model: torch.nn.Module, 
                         ids: torch.tensor, 
                         max_new_tokens: int, 
                         context_size: int, 
                         tokenizer=None):
  model.eval()
  for _ in range(max_new_tokens):
    with torch.inference_mode():
      logits = model(ids[:, -context_size:])
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=-1)

  if tokenizer:
    print(tokenizer.decode(ids[0].tolist()))
  return ids


def generate_single_word(model: torch.nn.Module, 
                         input: torch.tensor):
  """
  Takes a torch tensor of ids shaped [batch_size, sequence_length]. 
  Passes this through the GPT model. Appends max of last outputs logits.
  """
  assert len(input.shape) == 2
  model.eval()
  with torch.inference_mode():
    outputs = model(input)[:, -1, :]
    next_ids = torch.argmax(outputs, dim=-1, keepdim=True)
  return torch.cat([input, next_ids], dim=-1)


def generate_words_simple(model: torch.nn.Module, 
                          tokenizer, 
                          text: str, 
                          length: int):
  ids = tokenizer.encode(text)
  input = torch.tensor(ids).unsqueeze(0) #batch_dim
  for _ in range(length):
    input = generate_single_word(model, input)
  return tokenizer.decode(input[0].tolist())

