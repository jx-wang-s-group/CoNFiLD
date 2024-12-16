## Imports
import torch
import sys
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from ConditionalNeuralField.scripts.train import trainer, ri
from basicutility import ReadInput as ri

## Setup
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
  
device = torch.device(dev)  

torch.manual_seed(42)
np.random.seed(42)

inp = ri.basic_input(sys.argv[1])

print(f"sys.argv[1]: {sys.argv[1]}")

## Hyperparams
test_batch_size = inp.test_batch_size # num of samples to generate
time_length = inp.time_length
latent_length = inp.latent_length

image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_heads= inp.num_heads
num_head_channels = inp.num_head_channels
attention_resolutions = inp.attention_resolutions

steps= inp.steps
noise_schedule= inp.noise_schedule

## Create model and diffusion
unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions
                        )

unet_model.load_state_dict(torch.load(inp.ema_path, map_location=torch.device('cpu')))
unet_model.to(device)


diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule
                                    )

## Unconditional sample

sample_fn = diff_model.p_sample_loop
gen_latents = sample_fn(unet_model, (test_batch_size, 1, time_length, latent_length))[:, 0]

## Denormalizing the latents (load the max and min of your training latent data)
max_val, min_val = np.load(inp.max_val), np.load(inp.min_val)
max_val, min_val = torch.tensor(max_val), torch.tensor(min_val)
gen_latents = (gen_latents + 1)*(max_val - min_val)/2. + min_val

## Decode the latents
yaml = ri.basic_input(inp.cnf_case_file_path)
fptrainer = trainer(yaml, infer_mode=False) # WARNING: To use infer_mode=True, please define your custom query points using the coord variable below
fptrainer.load(-1, siren_only=True)
fptrainer.nf.to(device)

coord = None # Define your query points here, by default None will result into using training query points

batch_size = 1 # if you are limited by your GPU Memory, please change the batch_size variable accordingly
n_samples = gen_latents.shape[0]
gen_fields = []

for sample_index in range(n_samples):
  for i in range(gen_latents.shape[1]//batch_size):
    gen_fields.append(fptrainer.infer(coord, gen_latents[sample_index, i*batch_size:(i+1)*batch_size]).detach().cpu().numpy())

gen_fields = np.concatenate(gen_fields)

np.save(inp.save_path, gen_fields)
