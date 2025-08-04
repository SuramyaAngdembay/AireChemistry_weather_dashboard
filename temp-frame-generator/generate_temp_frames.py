# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# DATA_DIR = "/data"
# FRAME_DIR = os.path.join(DATA_DIR, "temperature_frames")
# os.makedirs(FRAME_DIR, exist_ok=True)

# # Load forecast: shape (num_hours, NY, NX)
# temp = np.load(os.path.join(DATA_DIR, "temperature_forecast.npy"))
# vmin, vmax = np.nanmin(temp), np.nanmax(temp)
# cmap = plt.get_cmap("coolwarm")

# for i in range(temp.shape[0]):
#     arr = temp[i]
#     arr_norm = (arr - vmin) / (vmax - vmin + 1e-6)
#     rgba = (cmap(arr_norm) * 255).astype(np.uint8)
#     img = Image.fromarray(rgba[..., :3])  # Drop alpha if needed
#     img.save(os.path.join(FRAME_DIR, f"temp{i}.gif"), "GIF")
#     print(f"Saved {os.path.join(FRAME_DIR, f'temp{i}.gif')}")

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

DATA_DIR = "/data"
FRAME_DIR_BASE = os.path.join(DATA_DIR, "frames")
os.makedirs(FRAME_DIR_BASE, exist_ok=True)

# Variable config: {variable: filename in .npy, colormap, pretty name}
VARIABLES = {
    "temperature":   ("temperature_forecast.npy", "coolwarm", "°C"),
    "precipitation": ("rain_forecast.npy", "Blues", "mm"),
    "relative_humidity": ("relative_humidity_forecast.npy", "YlGnBu", "%"),
    "cloud_cover":   ("cloud_cover_forecast.npy", "Greys", "%"),
    "visibility":    ("visibility_forecast.npy", "cividis", "m"),
    "wind_speed":    ("wind_speed_forecast.npy", "viridis", "m/s"),
    # "wind_direction": ("wind_direction_forecast.npy", "twilight", "degrees"),  # Unusual to visualize directly, but you can
}

# BOUNDS: update as needed!
bounding_box = [
    [16.45, -22.13],  # top-left (west, north)
    [32.89, -22.13],  # top-right
    [32.89, -34.83],  # bottom-right
    [16.45, -34.83],  # bottom-left
]

for var, (npy_fname, cmap_name, units) in VARIABLES.items():
    print(f"\nProcessing {var}")
    arr = np.load(os.path.join(DATA_DIR, npy_fname))  # shape (T, NY, NX)
    num_frames, NY, NX = arr.shape
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    print(f"{var}: min={vmin:.2f}, max={vmax:.2f}")

    # Pick size proportional to grid (or upscale to match frontend)
    target_size = (NX * 10, NY * 10)
    cmap = plt.get_cmap(cmap_name)
    FRAME_DIR = os.path.join(FRAME_DIR_BASE, var)
    os.makedirs(FRAME_DIR, exist_ok=True)
    frame_info = []

    for t in range(num_frames):
        layer = arr[t]
        arr_norm = (layer - vmin) / (vmax - vmin + 1e-6)# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# DATA_DIR = "/data"
# FRAME_DIR = os.path.join(DATA_DIR, "temperature_frames")
# os.makedirs(FRAME_DIR, exist_ok=True)

# # Load forecast: shape (num_hours, NY, NX)
# temp = np.load(os.path.join(DATA_DIR, "temperature_forecast.npy"))
# vmin, vmax = np.nanmin(temp), np.nanmax(temp)
# cmap = plt.get_cmap("coolwarm")

# for i in range(temp.shape[0]):
#     arr = temp[i]
#     arr_norm = (arr - vmin) / (vmax - vmin + 1e-6)
#     rgba = (cmap(arr_norm) * 255).astype(np.uint8)
#     img = Image.fromarray(rgba[..., :3])  # Drop alpha if needed
#     img.save(os.path.join(FRAME_DIR, f"temp{i}.gif"), "GIF")
#     print(f"Saved {os.path.join(FRAME_DIR, f'temp{i}.gif')}")

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

DATA_DIR = "/data"
FRAME_DIR_BASE = os.path.join(DATA_DIR, "frames")
os.makedirs(FRAME_DIR_BASE, exist_ok=True)

# Variable config: {variable: filename in .npy, colormap, pretty name}
VARIABLES = {
    "temperature":   ("temperature_forecast.npy", "coolwarm", "°C"),
    "precipitation": ("rain_forecast.npy", "Blues", "mm"),
    "relative_humidity": ("relative_humidity_forecast.npy", "YlGnBu", "%"),
    "cloud_cover":   ("cloud_cover_forecast.npy", "Greys", "%"),
    "visibility":    ("visibility_forecast.npy", "cividis", "m"),
    "wind_speed":    ("wind_speed_forecast.npy", "viridis", "m/s"),
    # "wind_direction": ("wind_direction_forecast.npy", "twilight", "degrees"),  # Unusual to visualize directly, but you can
}

# BOUNDS: update as needed!
bounding_box = [
    [16.45, -22.13],  # top-left (west, north)
    [32.89, -22.13],  # top-right
    [32.89, -34.83],  # bottom-right
    [16.45, -34.83],  # bottom-left
]

for var, (npy_fname, cmap_name, units) in VARIABLES.items():
    print(f"\nProcessing {var}")
    arr = np.load(os.path.join(DATA_DIR, npy_fname))  # shape (T, NY, NX)
    num_frames, NY, NX = arr.shape
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    print(f"{var}: min={vmin:.2f}, max={vmax:.2f}")

    # Pick size proportional to grid (or upscale to match frontend)
    target_size = (NX * 10, NY * 10)
    cmap = plt.get_cmap(cmap_name)
    FRAME_DIR = os.path.join(FRAME_DIR_BASE, var)
    os.makedirs(FRAME_DIR, exist_ok=True)
    frame_info = []

    for t in range(num_frames):
        layer = arr[t]
        arr_norm = (layer - vmin) / (vmax - vmin + 1e-6)
        rgba = (cmap(arr_norm) * 255).astype(np.uint8)
        img = Image.fromarray(rgba[..., :3])
        img = img.resize(target_size, resample=Image.BICUBIC)
        frame_path = os.path.join(FRAME_DIR, f"{var}{t}.gif")
        img.save(frame_path, "GIF")
        frame_info.append({"filename": f"{var}{t}.gif"})
        if t % 24 == 0 or t == num_frames - 1:
            print(f"Saved {frame_path}")

    # Metadata for each variable
    metadata = {
        "bounding_box": bounding_box,
        "interval_hours": 1,
        "pixel_width": target_size[0],
        "pixel_height": target_size[1],
        "vmin": vmin,
        "vmax": vmax,
        "colormap": cmap_name,
        "units": units,
        "frames": frame_info
    }
    with open(os.path.join(FRAME_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata.json for {var}")

        rgba = (cmap(arr_norm) * 255).astype(np.uint8)
        img = Image.fromarray(rgba[..., :3])
        img = img.resize(target_size, resample=Image.BICUBIC)
        frame_path = os.path.join(FRAME_DIR, f"{var}{t}.gif")
        img.save(frame_path, "GIF")
        frame_info.append({"filename": f"{var}{t}.gif"})
        if t % 24 == 0 or t == num_frames - 1:
            print(f"Saved {frame_path}")

    # Metadata for each variable
    metadata = {
        "bounding_box": bounding_box,
        "interval_hours": 1,
        "pixel_width": target_size[0],
        "pixel_height": target_size[1],
        "vmin": vmin,
        "vmax": vmax,
        "colormap": cmap_name,
        "units": units,
        "frames": frame_info
    }
    with open(os.path.join(FRAME_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata.json for {var}")
