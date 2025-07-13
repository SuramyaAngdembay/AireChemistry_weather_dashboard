#!/usr/bin/env python3
import os
import numpy as np
import requests
from multiprocessing import Pool, cpu_count
import tqdm

LON_MIN, LON_MAX, DLON = -180, 180, 3
LAT_MIN, LAT_MAX, DLAT = -90, 90, 3

lon_vals = np.arange(LON_MIN, LON_MAX + DLON, DLON)
lat_vals = np.arange(LAT_MIN, LAT_MAX + DLAT, DLAT)
NX, NY = len(lon_vals), len(lat_vals)

API_URL = os.getenv("OPEN_METEO_API", "http://open-meteo-api:8080/v1/forecast")

def fetch_point(args):
    lat, lon = args
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "rain,temperature_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "UTC"
    }
    for attempt in range(3):  # Added: Retries
        try:
            r = requests.get(API_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            return (
                data["hourly"]["rain"][0],  # Consider [0:24] for full day and np.mean() if averaging
                data["hourly"]["temperature_2m"][0],
                data["hourly"]["wind_speed_10m"][0],
                data["hourly"]["wind_direction_10m"][0]
            )
        except Exception as e:
            print(f"Error @ {lat},{lon} (attempt {attempt+1}): {e}")
            time.sleep(2)  # Backoff
    return (np.nan, np.nan, np.nan, np.nan)

def build():
    grid_rain = np.full((NY, NX), np.nan, dtype=np.float32)
    grid_temp = np.full((NY, NX), np.nan, dtype=np.float32)
    grid_ws = np.full((NY, NX), np.nan, dtype=np.float32)
    grid_wd = np.full((NY, NX), np.nan, dtype=np.float32)
    pairs = [(lat, lon) for lat in lat_vals for lon in lon_vals]
    with Pool(cpu_count()) as p:
        results = p.map(fetch_point, pairs)
    idx = 0
    for iy in range(NY):
        for ix in range(NX):
            rain, temp, ws, wd = results[idx]
            grid_rain[iy, ix] = rain
            grid_temp[iy, ix] = temp
            grid_ws[iy, ix] = ws
            grid_wd[iy, ix] = wd
            idx += 1
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(fetch_point, pairs), total=len(pairs)))
        
    np.save("/data/rain_grid.npy", grid_rain)
    np.save("/data/temperature_grid.npy", grid_temp)
    np.save("/data/wind_speed_grid.npy", grid_ws)
    np.save("/data/wind_dir_grid.npy", grid_wd)
    print("Grids saved to /data/")

if __name__ == "__main__":
    while True:  # Added: Infinite loop for periodic updates
        build()
        time.sleep(300)  # 5 minutes; align with OPEN_METEO_REPEAT_INTERVAL