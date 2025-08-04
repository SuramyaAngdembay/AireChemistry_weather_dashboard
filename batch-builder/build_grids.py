#!/usr/bin/env python3
import os
import numpy as np
import requests
from multiprocessing import Pool, cpu_count

# # Grid settings
# LON_MIN, LON_MAX, DLON = -180, 180, 3
# LAT_MIN, LAT_MAX, DLAT = -90, 90, 3

# lon_vals = np.arange(LON_MIN, LON_MAX + DLON, DLON)
# lat_vals = np.arange(LAT_MIN, LAT_MAX + DLAT, DLAT)
# NX, NY = len(lon_vals), len(lat_vals)

# SA bounds (from your SA_BOUNDS)
LON_MIN, LON_MAX, DLON = 16.45, 32.89, 0.25
LAT_MIN, LAT_MAX, DLAT = -34.83, -22.13, 0.25
lon_vals = np.arange(LON_MIN, LON_MAX + DLON, DLON)
lat_vals = np.arange(LAT_MIN, LAT_MAX + DLAT, DLAT)
NX, NY = len(lon_vals), len(lat_vals)


API_URL = os.getenv("OPEN_METEO_API", "http://open-meteo-api:8080/v1/forecast")
MODEL_PARAMS = {
    "hourly": ",".join([
        "precipitation", "temperature_2m",
        "wind_speed_10m", "wind_direction_10m", "cloud_cover",
        "relative_humidity_2m", "visibility"
    ]),
    "timezone": "UTC"
}
OUTPUT_PATH = "/data"

def fetch_point(args):
    lat, lon = args
    params = {"latitude": lat, "longitude": lon, **MODEL_PARAMS}
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        h = r.json()["hourly"]
        # Each is a list of N time steps
        rain = np.array(h["precipitation"])
        temp = np.array(h["temperature_2m"])
        ws = np.array(h["wind_speed_10m"])
        wd = np.array(h["wind_direction_10m"])
        rh = np.array(h["relative_humidity_2m"])
        cc = np.array(h["cloud_cover"])
        vis = np.array(h["visibility"])
        # Wind vector components for each step
        rad = np.deg2rad(wd)
        u = -ws * np.sin(rad)
        v = -ws * np.cos(rad)
        # Each output is (num_hours,)
        return rain, temp, u, v, ws, wd, rh, cc, vis
    except Exception as e:
        print(f"Error @ {lat},{lon}: {e}")
        # Use nan arrays of default length 24 (adjust if Open-Meteo changes)
        num_hours = 24
        return (np.full(num_hours, np.nan),) * 9

def build():
    points = [(lat, lon) for lat in lat_vals for lon in lon_vals]
    with Pool(min(8, cpu_count())) as pool:
        results = pool.map(fetch_point, points)

    # Determine the number of forecast hours from the first non-nan result
    for r in results:
        if not np.isnan(r[0][0]):
            num_hours = len(r[0])
            break
    else:
        num_hours = 24  # fallback

    varnames = ["rain", "temperature", "wind_u", "wind_v",
                "wind_speed", "wind_direction", "relative_humidity", "cloud_cover", "visibility"]

    # Prepare empty grids for each variable
    grids = {vn: np.full((num_hours, NY, NX), np.nan, dtype=np.float32) for vn in varnames}

    idx = 0
    for iy in range(NY):
        for ix in range(NX):
            data = results[idx]
            for i, vn in enumerate(varnames):
                grids[vn][:, iy, ix] = data[i]
            idx += 1

    for vn in varnames:
        np.save(os.path.join(OUTPUT_PATH, f"{vn}_forecast.npy"), grids[vn])
        np.save(os.path.join(OUTPUT_PATH, f"{vn}.npy"), grids[vn][0]) 
        print(f"Saved {vn}_forecast.npy shape: {grids[vn].shape}")

if __name__ == "__main__":
    build()
