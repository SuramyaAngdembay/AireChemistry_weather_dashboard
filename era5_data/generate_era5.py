import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import rasterio
from rasterio.transform import from_origin
import os

nc_path = "data_0.nc"
rasters_dir = "../era5_rasters_bins"
geotiffs_dir = "../era5_geotiffs"
os.makedirs(rasters_dir, exist_ok=True)
os.makedirs(geotiffs_dir, exist_ok=True)

# Set your temp variable and fixed bounds
temp_var = 't2m'
temp_vmin, temp_vmax = 273.15, 303.15  # 0°C to 30°C

n_bins = 12

ds = xr.open_dataset(nc_path)
lat = ds.latitude.values
lon = ds.longitude.values
transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

for var in ds.data_vars:
    arr_full = ds[var].values  # [time, lat, lon]
    for t in range(arr_full.shape[0]):
        arr = arr_full[t]
        arr = np.nan_to_num(arr, nan=0)

        # -- Color bin logic --
        if var == temp_var:
            vmin, vmax = temp_vmin, temp_vmax
        else:
            vmin, vmax = np.percentile(arr, [5, 95])

        # (protect against zero-range)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6

        bounds = np.linspace(vmin, vmax, n_bins + 1)
        cmap = plt.get_cmap('plasma', n_bins)
        norm = BoundaryNorm(boundaries=bounds, ncolors=n_bins)

        # PNG raster
        plt.figure(figsize=(12, 5))
        plt.axis('off')
        plt.imshow(arr, cmap=cmap, norm=norm, origin='upper')
        plt.tight_layout(pad=0)
        png_name = f"{var}_hour{t:02d}_binned.png"
        plt.savefig(os.path.join(rasters_dir, png_name), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

        # GeoTIFF (raw data)
        tif_name = f"{var}_hour{t:02d}.tif"
        with rasterio.open(
            os.path.join(geotiffs_dir, tif_name), 'w',
            driver='GTiff', height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype, crs='EPSG:4326', transform=transform
        ) as dst:
            dst.write(arr, 1)
        print(f"Saved: {png_name}, {tif_name}")

print("Done!")
