import os
import eumdac
from datetime import datetime, timedelta
import zipfile
import xarray as xr
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import pyproj
import json

PRODUCTS = {
    "fire": "EO:EUM:DAT:0682",
    "wind": "EO:EUM:DAT:0676",
    "cloud": "EO:EUM:DAT:0678",
}

# ---------- Fetch latest product file for a type ------------
def fetch_latest(product_key, out_dir):
    consumer_key = 'Dc8fYqAKeujfAGbV0jvixrvvKcUa'
    consumer_secret = 'o_vBjfIqHAqr41RopbbWwUz56Xka'
    token = eumdac.AccessToken((consumer_key, consumer_secret))
    ds = eumdac.DataStore(token)
    collection = ds.get_collection(PRODUCTS[product_key])

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    products = list(collection.search(dtstart=start_time, dtend=end_time))
    if not products:
        print(f"No recent {product_key} data found.")
        return None
    latest = products[-1]
    product_id = str(latest)
    fname = os.path.join(out_dir, f"{product_id}.zip")
    with latest.open() as src, open(fname, "wb") as dst:
        dst.write(src.read())
    print(f"Downloaded: {fname}")

    # Unzip if necessary
    with zipfile.ZipFile(fname, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    # Find first .nc file in out_dir
    nc_file = next((f for f in os.listdir(out_dir) if f.endswith('.nc')), None)
    if not nc_file:
        print("No NetCDF file found in extracted data.")
        return None
    nc_path = os.path.join(out_dir, nc_file)
    return nc_path

# --------------- Fire (Raster) Handling --------------
def extract_africa_raster_from_netcdf(nc_path, variable="fire_result", out_tif_path=None, out_png_path=None):
    ds = xr.open_dataset(nc_path)
    arr = ds[variable].values  # (ny, nx)

    # Projection info
    proj_attrs = ds['mtg_geos_projection'].attrs
    h = proj_attrs['perspective_point_height']
    a = proj_attrs['semi_major_axis']
    b = proj_attrs['semi_minor_axis']
    lon_0 = proj_attrs['longitude_of_projection_origin']
    sweep = proj_attrs['sweep_angle_axis']

    x = ds['x'].values
    y = ds['y'].values
    xv, yv = np.meshgrid(x, y)
    xv_m = xv * h
    yv_m = yv * h

    proj = pyproj.Proj(proj='geos', h=h, lon_0=lon_0, sweep=sweep, a=a, b=b, units='m')
    lon, lat = proj(xv_m, yv_m, inverse=True)

    # Africa mask
    africa_mask = (
        (lat >= -35) & (lat <= 37) &
        (lon >= -20) & (lon <= 55) &
        np.isfinite(lat) & np.isfinite(lon)
    )
    arr_africa = np.where(africa_mask, arr, np.nan)

    # Save GeoTIFF
    if out_tif_path:
        ny, nx = arr_africa.shape
        left, right = -20, 55
        bottom, top = -35, 37
        transform = from_bounds(left, bottom, right, top, nx, ny)
        with rasterio.open(
            out_tif_path, 'w', driver='GTiff', height=ny, width=nx, count=1,
            dtype=arr_africa.dtype, crs='EPSG:4326', transform=transform
        ) as dst:
            dst.write(arr_africa, 1)
        print(f"Saved GeoTIFF: {out_tif_path}")

    # Save PNG
    if out_png_path:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=np.nanmin(arr_africa), vmax=np.nanmax(arr_africa))
        img = Image.fromarray((norm(arr_africa)*255).astype(np.uint8))
        img.save(out_png_path)
        print(f"Saved PNG: {out_png_path}")

    return arr_africa

# ------------- Wind (GeoJSON) Handling ------------------
def wind_nc_to_geojson(nc_path, out_geojson_path, region=(-20, 55, -35, 37), min_speed=0.5):
    ds = xr.open_dataset(nc_path)
    # These are 1D arrays, each wind observation
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    u = ds['speed_u_component'].values
    v = ds['speed_v_component'].values
    speed = ds['speed'].values
    direction = ds['direction'].values

    features = []
    for i in range(len(lats)):
        lat, lon = float(lats[i]), float(lons[i])
        if (lat < region[2]) or (lat > region[3]) or (lon < region[0]) or (lon > region[1]):
            continue
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        if speed[i] < min_speed:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "u": float(u[i]),
                "v": float(v[i]),
                "speed": float(speed[i]),
                "direction": float(direction[i])
            }
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_geojson_path, 'w') as f:
        json.dump(geojson, f)
    print(f"Saved GeoJSON: {out_geojson_path}")
    return geojson


def extract_africa_raster_from_cloud_netcdf(nc_path, variable="cloud_state", out_tif_path=None, out_png_path=None):
    ds = xr.open_dataset(nc_path)
    arr = ds[variable].values  # (ny, nx)

    # Projection info
    proj_attrs = ds['mtg_geos_projection'].attrs
    h = proj_attrs['perspective_point_height']
    a = proj_attrs['semi_major_axis']
    b = proj_attrs['semi_minor_axis']
    lon_0 = proj_attrs['longitude_of_projection_origin']
    sweep = proj_attrs['sweep_angle_axis']

    x = ds['x'].values
    y = ds['y'].values
    xv, yv = np.meshgrid(x, y)
    xv_m = xv * h
    yv_m = yv * h

    proj = pyproj.Proj(proj='geos', h=h, lon_0=lon_0, sweep=sweep, a=a, b=b, units='m')
    lon, lat = proj(xv_m, yv_m, inverse=True)

    # Africa mask
    africa_mask = (
        (lat >= -35) & (lat <= 37) &
        (lon >= -20) & (lon <= 55) &
        np.isfinite(lat) & np.isfinite(lon)
    )
    arr_africa = np.where(africa_mask, arr, np.nan)

    # Save GeoTIFF
    if out_tif_path:
        ny, nx = arr_africa.shape
        left, right = -20, 55
        bottom, top = -35, 37
        transform = from_bounds(left, bottom, right, top, nx, ny)
        with rasterio.open(
            out_tif_path, 'w', driver='GTiff', height=ny, width=nx, count=1,
            dtype=arr_africa.dtype, crs='EPSG:4326', transform=transform
        ) as dst:
            dst.write(arr_africa, 1)
        print(f"Saved GeoTIFF: {out_tif_path}")

    # Save PNG (categorical: use discrete color palette or greyscale)
    if out_png_path:
        # Cloud mask values: typically 0=clear, 1=cloud, 2=unknown, etc.
        # We'll use a simple palette for visual testing.
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        cmap = ListedColormap(['black', 'white', 'gray', 'blue', 'red'])  # edit for your classes
        img = Image.fromarray(np.uint8(np.nan_to_num(arr_africa, nan=0)))
        # Optionally use matplotlib for color:
        plt.imsave(out_png_path, img, cmap=cmap, vmin=0, vmax=4)
        print(f"Saved PNG: {out_png_path}")

    return arr_africa


if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    for prod in PRODUCTS:
        out_dir = f"/data/{prod}"
        os.makedirs(out_dir, exist_ok=True)
        nc_path = fetch_latest(prod, out_dir=out_dir)
        if not nc_path:
            continue
        if prod == "fire":
            out_tif = os.path.join(out_dir, f"{prod}_africa.tif")
            out_png = os.path.join(out_dir, f"{prod}_africa.png")
            extract_africa_raster_from_netcdf(
                nc_path,
                variable="fire_result",
                out_tif_path=out_tif,
                out_png_path=out_png
            )
        elif prod == "wind":
            out_geojson = os.path.join(out_dir, f"{prod}_africa.geojson")
            wind_nc_to_geojson(
                nc_path,
                out_geojson_path=out_geojson,
                region=(-20, 55, -35, 37),
                min_speed=0.5
            )
        elif prod == "cloud":
            out_tif = os.path.join(out_dir, f"{prod}_africa.tif")
            out_png = os.path.join(out_dir, f"{prod}_africa.png")
            extract_africa_raster_from_cloud_netcdf(
                nc_path,
                variable="cloud_state",
                out_tif_path=out_tif,
                out_png_path=out_png
            )
