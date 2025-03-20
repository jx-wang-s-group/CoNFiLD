import numpy as np
import torch

# Conditioning Data Loader
no_of_sensors = 1  # 1, 10, 100, 1000

# Load the sensor coordinates
measures = np.load(f'/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com'
                 f'/My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case4/'
                 f'random_sensor/input/random_sensor/{no_of_sensors}/measures.npy')
#coords = np.load(f'input/random_sensor/{no_of_sensors}/coords.npy')

# Compute the maximum and minimum values for each dimension (x, y, z)
max_val = np.max(measures)  # Max value for each dimension
min_val = np.min(measures)  # Min value for each dimension


# Print the max and min values
print(f"Max values (x, y, z): {max_val}")
print(f"Min values (x, y, z): {min_val}")


# Load the max_val.npy file
max_val = np.load(f'/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com/'
                  f'My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case4/'
                  f'random_sensor/input/data_scale/data_max.npy')

min_val = np.load(f'/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com/'
                  f'My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case4/'
                  f'random_sensor/input/data_scale/data_min.npy')

# Print the contents
print(f"Max values (x, y, z) from the provided input folder: {max_val}")
print(f"Min values (x, y, z) from the provided input folder: {min_val}")
# Save them to new files
#np.save('input/data_scale/data_max.npy', max_val)
#np.save('input/data_scale/data_min.npy', min_val)


data = np.load(f'/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com'
                 f'/My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case4/'
                 f'random_sensor/input/cnf_model/infos.npz')
print(data.files)  # Lists all keys stored in infos.npz

for key in data.files:
    print(f"{key}: {data[key]}")


checkpoint_path = "/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com/My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case4/random_sensor/input/cnf_model/checkpoint_20000.pt"
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