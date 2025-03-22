import torch
import numpy as np
from tqdm import tqdm 
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 64
LATENTS_HEIGHT = 64


def generate(prompt, 
             uncond_prompt, 
             input_image, 
             strength=0.8, 
             do_cfg=True, 
             cfg_scale=7.5, 
             sampler_name="ddpm", 
             n_inference_steps=50,
             models={},
             seed=42,
             device=None,
             idle_device=None,
             tokenizer=None):
  with torch.no_grad():
    if idle_device:
      to_idle: lambda x: x.to(idle_device)
    else:
      to_idle: lambda x: x
    generator = torch.Generator(device=device)
    generate.seed()

    clip = models["clip"]
    clip = clip.to(device)

    if do_cfg:
      cond_tokens = tokenizer.batch_ncode_plus([prompt], padding="max_length", max_lngth=77).input_ids
      cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
      cond_context = clip(cond_tokens)

      uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_lngth=77).input_ids
      uncond_tokns = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
      uncond_context = clip(uncond_tokens)

      context = torch.cat([cond_context, uncond_context]) #(2, 77, 768)
    else:
      tokens = tokenizer.batch_ncode_plus([prompt], padding="max_length", max_lngth=77).input_ids
      tokens = torch.tensor(tokens, dtype=torch.long, device=device)
      context = clip(tokens)

    to_idle(clip)

    if sampler_name == "ddpm":
      sampler = DDPMSampler(generator)
      sampler.set_inference_steps(n_inference_steps)

    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    if input_image:
      encoder = models["encoder"].to(device)
      input_image_tensor = input_image.resize(WIDTH, HEIGHT)
      input_image_tensor = np.array(input_image_tensor)
      input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
      input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
      input_image_tensor = input_image_tensor.unsqueeze(0)
      input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

      encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
      latents = encoder(input_image_tensor, encoder_noise)

      sampler.set_strength(strength=strength)
      latents = sampler.add_noise(latents, sampler.timesteps[0])

      to_idle(encoder)
    else:
      latents = torch.randn(latents_shape, generator=generator, device=device)
      
    diffusion = modells["diffusion"].to(device)
    
