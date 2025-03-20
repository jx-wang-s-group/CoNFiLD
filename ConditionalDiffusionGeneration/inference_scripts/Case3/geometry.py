import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import os

# Load the coordinates
coords = np.load("/Users/minghan/Library/CloudStorage/GoogleDrive-john.chuicandoit@gmail.com/My Drive/CoNFiLD/ConditionalDiffusionGeneration/inference_scripts/Case3/random_sensor/input_case3/cnf_model/coords.npy")

# Check the shape
print(f"Shape of coords: {coords.shape}")

# Print first few points
print(coords[:5])



# Extract x and y
x, y = coords[:, 0], coords[:, 1]


# Create results folder if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=1, alpha=0.7, marker="s")  # s controls point size
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Geometry Visualization")
plt.axis("equal")  # Keep aspect ratio
plt.grid(True)

save_path = os.path.join(results_dir, "coords_plot.png")
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")

plt.show()