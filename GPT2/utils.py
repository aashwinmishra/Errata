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


def text_to_token_ids(text, tokenizer):
  return torch.tensor(tokenizer.encode(text, allowed_special={'<|endoftext|>'})).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
  return tokenizer.decode(token_ids.squeeze(0).tolist())


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


def generate_text_probabilistic(model: torch.nn.Module, 
                         ids: torch.tensor, 
                         max_new_tokens: int, 
                         context_size: int, 
                         tokenizer=None):
  model.eval()
  for _ in range(max_new_tokens):
    with torch.inference_mode():
      logits = model(ids[:, -context_size:])
    dist = torch.distributions.categorical.Categorical(logits=logits[:, -1, :])
    next_id = dist.sample()
    ids = torch.cat([ids, next_id], dim=-1)

  if tokenizer:
    print(tokenizer.decode(ids[0].tolist()))
  return ids


def generate_text(model: torch.nn.Module, 
                                ids: torch.tensor, 
                                max_new_tokens: int, 
                                context_size: int,
                                temperature=1.0, 
                                top_k=25,
                                tokenizer=None):
  model.eval()
  for _ in range(max_new_tokens):
    with torch.inference_mode():
      logits = model(ids[:, -context_size:])
    logits = logits[:, -1, :]
    top_logits, _ = torch.topk(logits, top_k)
    min_val = top_logits[:, -1]
    logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits) / temperature 
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
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



def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]


    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)


    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return


            block_size = 1024  # 1 Kilobyte
            progress_bar_description = os.path.basename(url)  # Extract filename from URL
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # Open the destination file in binary write mode
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar
    except urllib.error.HTTPError:
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
    return params
