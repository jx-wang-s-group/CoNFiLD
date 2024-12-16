import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ema_path', type=str, required=True, help='Path to the model checkpoint file')
inp = parser.parse_args()

print(f"ema_path: {inp.ema_path}")  # Add this line to debug