#!/usr/bin/env python3
import os
import numpy as np
import requests
from multiprocessing import Pool, cpu_count

# 📍 Grid settings
LON_MIN, LON_MAX, DLON = -180, 180, 3  # adjust resolution
LAT_MIN, LAT_MAX, DLAT = -90, 90, 3

lon_vals = np.arange(LON_MIN, LON_MAX + DLON, DLON)
lat_vals = np.arange(LAT_MIN, LAT_MAX + DLAT, DLAT)
NX, NY = len(lon_vals), len(lat_vals)

API_URL = os.getenv("OPEN_METEO_API", "http://open-meteo-api:8080/v1/forecast")
MODEL_PARAMS = {
    "hourly": ",".join([
        "precipitation", "temperature_2m",
        "wind_speed_10m", "wind_direction_10m","cloud_cover","relative_humidity_2m","visibility"
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
        rain = h["precipitation"][0]
        temp = h["temperature_2m"][0]
        ws = h["wind_speed_10m"][0]
        wd = h["wind_direction_10m"][0]
        rh = h["relative_humidity_2m"][0]
        cc = h["cloud_cover"][0]
        vis = h["visibility"][0]
        rad = np.deg2rad(wd)
        u = -ws * np.sin(rad)
        v = -ws * np.cos(rad)
        return rain, temp, u, v, ws, wd, rh, cc,vis
    except Exception as e:
        print(f"Error @ {lat},{lon}: {e}")
        return (np.nan,) * 9


def build():
    grid = np.full((NY, NX, 9), np.nan, dtype=np.float32)
    points = [(lat, lon) for lat in lat_vals for lon in lon_vals]
    with Pool(min(8, cpu_count())) as pool:
        results = pool.map(fetch_point, points)

    idx = 0
    for iy in range(NY):
        for ix in range(NX):
            grid[iy, ix] = results[idx]
            idx += 1

    names = ["rain","temperature","wind_u","wind_v","wind_speed","wind_direction","relative_humidity","cloud_cover","visibility"]
    for i, nm in enumerate(names):
        np.save(os.path.join(OUTPUT_PATH, f"{nm}.npy"), grid[..., i])
    print("Grids saved:", ", ".join(names))

if __name__ == "__main__":
    build()
