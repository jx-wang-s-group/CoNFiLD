import numpy as np
import torch
# Number of sensors
no_of_sensors = 10  # 1, 10, 100, 1000

# 1) Load the sensor measurements
measures_path = f"input_case3/random_sensor/{no_of_sensors}/measures.npy"
measures = np.load(measures_path)
print(f"Loaded measurements from: {measures_path}")
print("DEBUG: measures.shape =", measures.shape)

# (Optional) Print a small sample of the data
print("DEBUG: measures[0, 0, :] =", measures[0, 0, :], "(first time step, first sensor)")

# 2) Now that we know the shape, let's see how to compute max/min
#    If measures.shape == (time_length, num_sensors, channels), for example (384, 10, 3),
#    you likely want the max over the first two axes to get a (channels,) shape.

# This code calculates the global max/min across all time steps & sensors:
try:
    max_val = np.max(measures, axis=(0, 1))  # shape -> (channels,)
    min_val = np.min(measures, axis=(0, 1))
except ValueError as e:
    print("ERROR computing global max/min. Check if measures have at least 2 dims!")
    raise e

print(f"Max values: {max_val}")
print(f"Min values: {min_val}")

# 3) Save them to new files
out_dir = "input_case3/data_scale/"
np.save(f"{out_dir}/data_max.npy", max_val)
np.save(f"{out_dir}/data_min.npy", min_val)

print("data_max.npy and data_min.npy saved successfully!")

checkpoint_path = "/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com/My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case3/random_sensor/input_case3/cnf_model/checkpoint_4800.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

for key in checkpoint.keys():
    print(key)  # Look for something related to 'channel_mult'

# Extract model state dict
state_dict = checkpoint["model_state_dict"]

# Print all layer names
print("Model layers:", state_dict.keys())

# Look for the final convolution layer (it usually starts with 'out' or 'conv_out')
for key, value in state_dict.items():
    print(f"Layer: {key}, Shape: {value.shape}")

# Find the last layer
last_layer = list(state_dict.keys())[-1]
print(f"Final layer: {last_layer}, Shape: {state_dict[last_layer].shape}")