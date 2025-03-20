import torch
import numpy as np
from functools import partial
import sys
from einops import rearrange

sys.path.append("../../../..") #four levels up from script's current location (Instead of modifying sys.path dynamically, consider using absolute paths or configuring the PYTHONPATH environment variable.)

from ConditionalDiffusionGeneration.src.guided_diffusion.unet import create_model
from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import get_conditioning_method
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise, get_operator
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import create_sampler
from ConditionalNeuralField.cnf.inference_function import ReconstructFrame, decoder

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

device = torch.device(dev)
print(f"Running on device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# Conditioning Data Loader
no_of_sensors = 10  # 1, 10, 100, 1000
true_measurement = torch.from_numpy(np.load(f'input/random_sensor/{no_of_sensors}/measures.npy')).to(device)

# Load trained unconditional model
u_net_model = create_model(
    image_size=384,
    num_channels=128,
    num_res_blocks=2,
    channel_mult="1, 1, 2, 2, 4, 4",
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    model_path='./input/diff_model/ema_0.9999_390000.pt'
)

u_net_model.to(device)
u_net_model.eval()

# Operator and Noise
operator = get_operator(
    device=device,
    name='case4',
    coords_path=f'input/random_sensor/{no_of_sensors}/coords.npy',
    max_val_path="input/data_scale/data_max.npy",
    min_val_path="input/data_scale/data_min.npy",
    normalizer_params_path="input/cnf_model/Hirachical2-11-11_normalizer_params.pt",
    ckpt_path="input/cnf_model/checkpoint_20000.pt",
    batch_size=384
)

noiser = get_noise(sigma=0.0, name='gaussian')

# For masking in time
# start, stop, step = 0, 257, 5
mask = torch.ones_like(true_measurement, device=device)
# mask[start:stop:step] = 1

# Conditioning Method
cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='ps', scale=1.)
measurement_cond_fn = partial(cond_method.conditioning)

# Sampler
sampler = create_sampler(
    sampler='ddpm',
    steps=1000,
    noise_schedule="cosine",
    model_mean_type="epsilon",
    model_var_type="fixed_large",
    dynamic_threshold=False,
    clip_denoised=True,
    rescale_timesteps=False,
    timestep_respacing=""
)

sample_fn = partial(sampler.p_sample_loop, model=u_net_model, measurement_cond_fn=measurement_cond_fn)

# Generate Samples
no_of_samples = 10
time_length = 384
latent_size = 384

x_start = torch.randn(no_of_samples, 1, time_length, latent_size, device=device)
samples = [sample_fn(x_start=x_start[i:i+1], measurement=true_measurement, record=False, save_root=None) for i in range(x_start.shape[0])]

gen_latents = torch.cat(samples)
gen_latents = operator._unnorm(gen_latents)
gen_latents = gen_latents[:, 0]

# Decoding latents to flow fields
info = np.load("input/cnf_model/infos.npz")
coords = torch.tensor(np.load("input/cnf_model/coords.npy"), device=device, dtype=torch.float32)

xnorm = operator.x_normalizer
ynorm = operator.y_normalizer
model = operator.model
gen_latents_cnf_input = rearrange(gen_latents, "s t l -> (s t) l")

gen_fields = decoder(coords, gen_latents_cnf_input, model, xnorm, ynorm, batch_size=16, device=device)
gen_fields = rearrange(gen_fields, "(s t) co c -> s t co c", t=time_length)

# Post-processing the flow fields for visualization
pred_data_list = []
for ss in range(no_of_samples):
    for kk in range(time_length):
        pred_data = gen_fields[ss, kk].cpu().numpy()
        pred_data = ReconstructFrame(pred_data, mask=info['Mask'], shape=info['reduced_shape'], fill_value=0.)
        pred_data_list.append(pred_data)

pred_data_list = rearrange(np.stack(pred_data_list), "(s t) x y z c -> s t x y z c", t=time_length)

# Save operator, samples, and pred_data_list
output_dir = "output_files/"
torch.save(operator, f"{output_dir}/operator.pt")
torch.save(samples, f"{output_dir}/samples.pt")
np.save(f"{output_dir}/pred_data_list.npy", pred_data_list)
torch.save(gen_fields, f"{output_dir}/gen_fields.pt")  # Save gen_fields

print("Files saved successfully!")