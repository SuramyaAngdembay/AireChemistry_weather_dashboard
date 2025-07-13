from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

# Grid settings
LON_MIN, LON_MAX, DLON = -180, 180, 3
LAT_MIN, LAT_MAX, DLAT = -90, 90, 3
lon_vals = np.arange(LON_MIN, LON_MAX + DLON, DLON)
lat_vals = np.arange(LAT_MIN, LAT_MAX + DLAT, DLAT)
NX, NY = len(lon_vals), len(lat_vals)

def grid_indices(lon_min, lon_max, lat_min, lat_max):
    ix0 = max(0, int(round((lon_min - LON_MIN) / DLON)))
    ix1 = min(NX, int(round((lon_max - LON_MIN) / DLON)) + 1)
    iy0 = max(0, int(round((lat_min - LAT_MIN) / DLAT)))
    iy1 = min(NY, int(round((lat_max - LAT_MIN) / DLAT)) + 1)
    return ix0, ix1, iy0, iy1

def wind_uv(speed, direction_deg):
    """Convert wind speed and direction (meteorological) to u/v components."""
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u, v

@app.get("/rain")
def get_rain(
    lon_min: float = Query(None),
    lon_max: float = Query(None),
    lat_min: float = Query(None),
    lat_max: float = Query(None)
):
    try:
        grid = np.load("/data/rain_grid.npy")
    except FileNotFoundError:
        return JSONResponse({"error": "Grid file not found. Run batch-builder first."}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
    # If bbox None, use full grid (added logic like wind endpoint)
    if None in (lon_min, lon_max, lat_min, lat_max):
        ix0, ix1, iy0, iy1 = 0, NX, 0, NY
    else:
        ix0, ix1, iy0, iy1 = grid_indices(lon_min, lon_max, lat_min, lat_max)
    
    result = []
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            lon = float(lon_vals[ix])
            lat = float(lat_vals[iy])
            precip = float(grid[iy, ix])
            result.append({"lon": lon, "lat": lat, "precipitation": precip})  # Standardized to dict
    return JSONResponse(result)

@app.get("/temperature")
def get_temperature(
    lon_min: float = Query(...),
    lon_max: float = Query(...),
    lat_min: float = Query(...),
    lat_max: float = Query(...)
):
    grid = np.load("/data/temperature_grid.npy")
    ix0, ix1, iy0, iy1 = grid_indices(lon_min, lon_max, lat_min, lat_max)
    result = []
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            lon = float(lon_vals[ix])
            lat = float(lat_vals[iy])
            temp = float(grid[iy, ix])
            result.append([lon, lat, temp])
    return JSONResponse(result)

@app.get("/wind")
def get_wind(
    component: str = Query("u", enum=["u", "v"]),
    lon_min: float = Query(None),
    lon_max: float = Query(None),
    lat_min: float = Query(None),
    lat_max: float = Query(None)
):
    ws_grid = np.load("/data/wind_speed_grid.npy")
    wd_grid = np.load("/data/wind_dir_grid.npy")
    u_grid, v_grid = wind_uv(ws_grid, wd_grid)
    grid = u_grid if component == "u" else v_grid

    # Determine subgrid indices (if bounding box provided)
    if None not in (lon_min, lon_max, lat_min, lat_max):
        ix0 = max(0, int(round((lon_min - LON_MIN) / DLON)))
        ix1 = min(NX, int(round((lon_max - LON_MIN) / DLON)) + 1)
        iy0 = max(0, int(round((lat_min - LAT_MIN) / DLAT)))
        iy1 = min(NY, int(round((lat_max - LAT_MIN) / DLAT)) + 1)
        subgrid = grid[iy0:iy1, ix0:ix1]
        sub_lon_vals = lon_vals[ix0:ix1]
        sub_lat_vals = lat_vals[iy0:iy1]
    else:
        subgrid = grid
        sub_lon_vals = lon_vals
        sub_lat_vals = lat_vals

    nx, ny = subgrid.shape[1], subgrid.shape[0]

    # Update header for subgrid
    header = {
        "parameterCategory": 2,
        "parameterNumber": 2 if component == "u" else 3,
        "nx": nx,
        "ny": ny,
        "lo1": float(sub_lon_vals[0] - DLON/2),
        "la1": float(sub_lat_vals[0] + DLAT/2),
        "lo2": float(sub_lon_vals[-1] + DLON/2),
        "la2": float(sub_lat_vals[-1] - DLAT/2),
        "dx": float(DLON),
        "dy": float(DLAT),
    }
    data = subgrid.flatten().astype(float).tolist()
    return JSONResponse({"header": header, "data": data})
