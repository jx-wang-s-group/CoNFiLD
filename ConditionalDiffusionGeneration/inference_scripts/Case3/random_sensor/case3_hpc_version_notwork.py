import torch
import numpy as np
from functools import partial
import sys
import os
from einops import rearrange

# Update this path to your project structure if needed
sys.path.append("../../..")

from ConditionalDiffusionGeneration.src.guided_diffusion.unet import create_model
from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import get_conditioning_method
from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise, get_operator
from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import create_sampler
from ConditionalNeuralField.cnf.inference_function import decoder

# ------------------------------------------------------------------
# 1) Device Setup
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# 2) Load Measurement Data (Case 3)
# ------------------------------------------------------------------
# Suppose you have a measurement file with shape (time, sensors, 3) for (u, v, w)
no_of_sensors = 10
measurements_path = f"input_case3/random_sensor/{no_of_sensors}/measures.npy"
true_measurement = torch.from_numpy(np.load(measurements_path)).to(device)

# Select only the first 2 velocity components (u, v)
true_measurement = true_measurement[:, :, :2]  # This ensures (384, 10, 2)
print("DEBUG: true_measurement.shape =", true_measurement.shape)

# ------------------------------------------------------------------
# 3) Load U-Net Diffusion Model (Case 3)
# ------------------------------------------------------------------
# * Replace 'channel_mult' and 'model_path' with the actual values used in your training *
u_net_model = create_model(
    image_size=256,             # Example: your domain resolution for Case 3
    num_channels=128,
    num_res_blocks=2,
    channel_mult=" ",     # Or whichever you used in training for Case 3
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    model_path="input_case3/diff_model/ema_0.9999_340000.pt"  # Put your actual checkpoint path
)

u_net_model.to(device)
u_net_model.eval()

# ------------------------------------------------------------------
# 4) Operator and Noise (Case 3)
# ------------------------------------------------------------------
# * name='case3' so it uses your operator definition for periodic hill
# * coords_path, max_val_path, min_val_path must match your new 3-channel data setup
operator = get_operator(
    device=device,
    name='case3',
    coords=np.load(f"input_case3/random_sensor/{no_of_sensors}/coords.npy"),  # If coords is 2D array for the domain
    max_val=np.load("input_case3/data_scale/data_max.npy"),  # Should be shape (3,) for (u, v, w)
    min_val=np.load("input_case3/data_scale/data_min.npy"),  # Also shape (3,)
    normalizer_params_path="input_case3/cnf_model/normalizer_params.pt",  # If you have a normalizer .pt file
    ckpt_path="input_case3/cnf_model/checkpoint_4800.pt",    # Your trained CNF model for Case 3
    batch_size=384
)

noiser = get_noise(sigma=0.0, name='gaussian')

# ------------------------------------------------------------------
# 5) Conditioning Method
# ------------------------------------------------------------------
cond_method = get_conditioning_method(operator=operator, noiser=noiser, name='ps', scale=1.)
measurement_cond_fn = partial(cond_method.conditioning)

# ------------------------------------------------------------------
# 6) Sampler
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 7) Generate Latent Samples
# ------------------------------------------------------------------
# * Adjust time_length, latent_size if your Case 3 training differs
no_of_samples = 10
time_length = 256
latent_size = 256

x_start = torch.randn(no_of_samples, 1, time_length, latent_size, device=device)
print("DEBUG: x_start.shape =", x_start.shape)

samples = []
for i in range(x_start.shape[0]):
    sample_out = sample_fn(
        x_start=x_start[i:i+1],
        measurement=true_measurement,
        record=False,
        save_root=None
    )
    samples.append(sample_out)

gen_latents = torch.cat(samples, dim=0)
print("DEBUG: gen_latents.shape =", gen_latents.shape)

# Unnormalize from operator
gen_latents = operator._unnorm(gen_latents)
print("DEBUG: gen_latents (after _unnorm) =", gen_latents.shape)

# Usually we select gen_latents[:, 0] if the model outputs multiple channels in dim=1
gen_latents = gen_latents[:, 0]
print("DEBUG: gen_latents (squeezed) =", gen_latents.shape)

# ------------------------------------------------------------------
# 8) Decode Latents to Flow Fields (No Masking)
# ------------------------------------------------------------------
coords = torch.tensor(np.load("input_case3/cnf_model/coords.npy"), device=device, dtype=torch.float32)
print("DEBUG: coords.shape =", coords.shape)

xnorm = operator.x_normalizer
ynorm = operator.y_normalizer
model = operator.model

# Reshape latents from [s, t, l] -> [(s*t), l]
gen_latents_cnf_input = rearrange(gen_latents, "s t l -> (s t) l")
print("DEBUG: gen_latents_cnf_input.shape =", gen_latents_cnf_input.shape)

gen_fields = decoder(coords, gen_latents_cnf_input, model, xnorm, ynorm, batch_size=16, device=device)
# Now reshape back to [s, t, ..., c]
gen_fields = rearrange(gen_fields, "(s t) co c -> s t co c", t=time_length)
print("DEBUG: gen_fields.shape =", gen_fields.shape)

# ------------------------------------------------------------------
# 9) Post-Processing (No Masking)
# ------------------------------------------------------------------
# For a full field, we don't need ReconstructFrame or mask logic.
pred_data_list = gen_fields.cpu().numpy()

# ------------------------------------------------------------------
# 10) Save outputs
# ------------------------------------------------------------------
output_dir = "output_files_case3/"
os.makedirs(output_dir, exist_ok=True)

torch.save(operator, f"{output_dir}/operator_case3.pt")
torch.save(samples, f"{output_dir}/samples_case3.pt")
np.save(f"{output_dir}/pred_data_list_case3.npy", pred_data_list)
torch.save(gen_fields, f"{output_dir}/gen_fields_case3.pt")

print("Case 3 inference completed. Files saved successfully!")