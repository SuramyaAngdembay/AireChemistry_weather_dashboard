from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

# Grid params
nx, ny = 360, 181
header_template = {
    "la1": 90.5, "la2": -90.5, "lo1": -180.5, "lo2": 179.5,
    "extent": [-180.5, -90.5, 179.5, 90.5],
    "nx": 360, "ny": 181, "dx": 1.0, "dy": 1.0,
}
# For wind endpoints, use these fields
wind_header_fields = {
    "parameterCategory": 1,  # Meteorological
    "parameterNumber": 2,    # U for /wind-u, 3 for /wind-v
    "GRIB_COMMENT": "u-component of wind [m/s]",
    "GRIB_DISCIPLINE": "0(Meteorological)",
    "GRIB_ELEMENT": "UGRD",
    "GRIB_FORECAST_SECONDS": "0 sec",
    "GRIB_IDS": "...",
    "GRIB_PDS_PDTN": "0",
    "GRIB_PDS_TEMPLATE_ASSEMBLED_VALUES": "...",
    "GRIB_PDS_TEMPLATE_NUMBERS": "...",
    "GRIB_REF_TIME": "1592611200 sec UTC",
    "GRIB_SHORT_NAME": "10-HTGL",
    "GRIB_UNIT": "[m/s]",
    "GRIB_VALID_TIME": "1592611200 sec UTC"
}

def serve_grid(fname, paramcat, paramnum, comment, units):
    grid = np.load(fname)
    header = dict(header_template)
    header.update(wind_header_fields)
    header["parameterCategory"] = paramcat
    header["parameterNumber"] = paramnum
    header["GRIB_COMMENT"] = comment
    header["GRIB_UNIT"] = units
    header["min"] = float(np.min(grid))
    header["max"] = float(np.max(grid))
    return JSONResponse([{
        "header": header,
        "data": grid.flatten().tolist()
    }])

@app.get("/wind-u")
def wind_u():
    return serve_grid(
        "/data/wind_u_dummy.npy", 1, 2, "u-component of wind [m/s]", "[m/s]"
    )

@app.get("/wind-v")
def wind_v():
    return serve_grid(
        "/data/wind_v_dummy.npy", 1, 3, "v-component of wind [m/s]", "[m/s]"
    )

@app.get("/temperature")
def temperature():
    grid = np.load("/data/temperature_dummy.npy")
    header = dict(header_template)
    header.update({
        "parameterCategory": 0,   # Temperature
        "parameterNumber": 0,
        "GRIB_COMMENT": "Air temperature [C]",
        "GRIB_UNIT": "[C]",
        "min": float(np.min(grid)),
        "max": float(np.max(grid)),
    })
    return JSONResponse([{
        "header": header,
        "data": grid.flatten().tolist()
    }])

@app.get("/rain")
def rain():
    grid = np.load("/data/rain_dummy.npy")
    header = dict(header_template)
    header.update({
        "parameterCategory": 1,   # Precipitation
        "parameterNumber": 7,
        "GRIB_COMMENT": "Total precipitation [mm]",
        "GRIB_UNIT": "[mm]",
        "min": float(np.min(grid)),
        "max": float(np.max(grid)),
    })
    return JSONResponse([{
        "header": header,
        "data": grid.flatten().tolist()
    }])
