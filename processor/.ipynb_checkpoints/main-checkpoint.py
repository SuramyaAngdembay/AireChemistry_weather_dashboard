from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

# Assume grid metadata is fixed and known
NX, NY = 360, 181   # Example: global 1-degree grid
LON_MIN, LON_MAX = -180.5, 179.5
LAT_MIN, LAT_MAX = -90.5, 90.5
DLON, DLAT = 1.0, 1.0

def grid_indices(lon_min, lon_max, lat_min, lat_max):
    ix0 = max(0, int(round((lon_min - LON_MIN) / DLON)))
    ix1 = min(NX, int(round((lon_max - LON_MIN) / DLON)) + 1)
    iy0 = max(0, int(round((LAT_MAX - lat_max) / DLAT)))
    iy1 = min(NY, int(round((LAT_MAX - lat_min) / DLAT)) + 1)
    return ix0, ix1, iy0, iy1

def coords(ix, iy):
    lon = LON_MIN + ix * DLON
    lat = LAT_MAX - iy * DLAT
    return lon, lat

@app.get("/rain")
def get_rain(
    lon_min: float = Query(...), lon_max: float = Query(...),
    lat_min: float = Query(...), lat_max: float = Query(...)
):
    grid = np.load("/data/rain_grid.npy")
    ix0, ix1, iy0, iy1 = grid_indices(lon_min, lon_max, lat_min, lat_max)
    result = []
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            lon, lat = coords(ix, iy)
            precip = float(grid[iy, ix])
            result.append({"COORDINATES": [lon, lat], "PRECIPITATION": precip})
    return JSONResponse(result)

@app.get("/temperature")
def get_temp(
    lon_min: float = Query(...), lon_max: float = Query(...),
    lat_min: float = Query(...), lat_max: float = Query(...)
):
    grid = np.load("/data/temp_grid.npy")
    ix0, ix1, iy0, iy1 = grid_indices(lon_min, lon_max, lat_min, lat_max)
    result = []
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            lon, lat = coords(ix, iy)
            intensity = float(grid[iy, ix])
            result.append([lon, lat, intensity])
    return JSONResponse(result)

@app.get("/wind")
def get_wind(
    lon_min: float = Query(...), lon_max: float = Query(...),
    lat_min: float = Query(...), lat_max: float = Query(...)
):
    grid_u = np.load("/data/wind_u.npy")
    grid_v = np.load("/data/wind_v.npy")
    ix0, ix1, iy0, iy1 = grid_indices(lon_min, lon_max, lat_min, lat_max)
    u_data, v_data = [], []
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            u_data.append(float(grid_u[iy, ix]))
            v_data.append(float(grid_v[iy, ix]))
    # Format as expected by Mapbox-Wind etc.
    wind_json = {
        "header": {
            "nx": ix1-ix0,
            "ny": iy1-iy0,
            "lo1": LON_MIN + ix0*DLON,
            "la1": LAT_MAX - iy0*DLAT,
            "lo2": LON_MIN + (ix1-1)*DLON,
            "la2": LAT_MAX - (iy1-1)*DLAT,
            "dx": DLON,
            "dy": DLAT,
        },
        "data_u": u_data,
        "data_v": v_data
    }
    return JSONResponse(wind_json)

