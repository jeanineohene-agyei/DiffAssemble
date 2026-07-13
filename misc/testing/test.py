import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

PC_PATH = "datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_point_cloud.npy"
BD_PATH = "datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_depth_boundary.tif"

import numpy as np
import pandas as pd

npy_path = "datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_point_cloud.npy"

csv_path = "output.csv"

data = np.load(npy_path, allow_pickle=True)

# If it's a saved dictionary/object
if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
    obj = data.item()

    if isinstance(obj, dict):
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in obj.items()]))
    else:
        df = pd.DataFrame(obj)

# If it's a regular NumPy array
else:
    if data.ndim == 1:
        df = pd.DataFrame(data, columns=["value"])
    else:
        df = pd.DataFrame(data)

df.to_csv(csv_path, index=False)

print(f"Saved CSV to: {csv_path}")