from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
from datetime import datetime
import os
import glob
import json
import rasterio

app = FastAPI(title="South Africa Weather/Pollution API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = "/data"  # Docker volume mount


# ----------- Standard Weather Grid Endpoints -----------

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


header_template = {
    "la1": LAT_MAX + DLAT/2, "la2": LAT_MIN - DLAT/2,
    "lo1": LON_MIN - DLON/2, "lo2": LON_MAX + DLON/2,
    "extent": [LON_MIN - DLON/2, LAT_MIN - DLAT/2, LON_MAX + DLON/2, LAT_MAX + DLAT/2],
    "nx": NX, "ny": NY, "dx": DLON, "dy": DLAT
}

def serve(fname, paramCategory, paramNumber, comment, unit):
    grid = np.load(fname)
    header = dict(header_template)
    header.update({
        "parameterCategory": paramCategory,
        "parameterNumber": paramNumber,
        "GRIB_COMMENT": comment,
        "GRIB_UNIT": unit,
        "min": float(np.nanmin(grid)),
        "max": float(np.nanmax(grid)),
    })
    return JSONResponse([{"header": header, "data": grid.flatten().tolist()}])

def grid_to_point_list(grid):
    pts = []
    for iy in range(NY):
        for ix in range(NX):
            lon = float(lon_vals[ix])
            lat = float(lat_vals[iy])
            val = grid[iy, ix].item()
            if not np.isnan(val):
                pts.append([lon, lat, val])
    return pts

def grid_to_point_objs(grid):
    pts = []
    for iy in range(NY):
        for ix in range(NX):
            lon = float(lon_vals[ix])
            lat = float(lat_vals[iy])
            val = grid[iy, ix].item()
            if not np.isnan(val):
                pts.append({"COORDINATES": [lon, lat], "PRECIPITATION": val})
    return pts

@app.get("/temperature")
def temperature():
    grid = np.load("/data/temperature.npy")
    pts = grid_to_point_list(grid)
    return JSONResponse(pts)

@app.get("/rain")
def rain():
    grid = np.load("/data/rain.npy")
    pts = grid_to_point_objs(grid)
    return JSONResponse(pts)
    
@app.get("/relative_humidity")
def humidity():
    grid = np.load("/data/relative_humidity.npy")
    return JSONResponse(grid_to_point_list(grid))

@app.get("/cloud_cover")
def cloud_cover():
    grid = np.load("/data/cloud_cover.npy")
    return JSONResponse(grid_to_point_list(grid))
    
@app.get("/wind-u")
def wind_u():
    return serve("/data/wind_u.npy", 1, 2, "u-component of wind [m/s]", "[m/s]")

@app.get("/wind-v")
def wind_v():
    return serve("/data/wind_v.npy", 1, 3, "v-component of wind [m/s]", "[m/s]")

@app.get("/wind-speed")
def wind_speed():
    return serve("/data/wind_speed.npy", 1, 3, "wind speed [m/s]", "[m/s]")

@app.get("/wind-dirn")
def wind_direction():
    return serve("/data/wind_direction.npy", 1, 3, "wind dirn ", "degrees")

# ----------- South Africa MOCK "TEXTURE" ENDPOINTS -----------

SA_BOUNDS = {
    "min_lon": 16.45,
    "max_lon": 32.89,
    "min_lat": -34.83,
    "max_lat": -22.13
}

INDUSTRIAL_SOURCES = [
    {"name": "Johannesburg", "coords": [28.0473, -26.2041], "intensity": 0.9},
    {"name": "Sasolburg", "coords": [27.8167, -26.8167], "intensity": 0.85},
    {"name": "Secunda", "coords": [29.1667, -26.5167], "intensity": 0.8},
    {"name": "Durban", "coords": [31.0292, -29.8587], "intensity": 0.75},
    {"name": "Cape Town", "coords": [18.4241, -33.9249], "intensity": 0.7}
]

def generate_sa_pm25_texture():
    width, height = 512, 512
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    pm25 = np.zeros((height, width))
    for source in INDUSTRIAL_SOURCES:
        norm_x = (source["coords"][0] - SA_BOUNDS["min_lon"]) / (SA_BOUNDS["max_lon"] - SA_BOUNDS["min_lon"])
        norm_y = 1 - (source["coords"][1] - SA_BOUNDS["min_lat"]) / (SA_BOUNDS["max_lat"] - SA_BOUNDS["min_lat"])
        tex_x = norm_x * width
        tex_y = norm_y * height
        dist = np.sqrt((x - tex_x)**2 + (y - tex_y)**2)
        pm25 += source["intensity"] * np.exp(-dist / (50 + np.random.random() * 30))
    month = datetime.now().month
    season_factor = 0.7 + 0.3 * np.cos(2*np.pi*(month-6)/12)
    pm25 *= season_factor
    noise = np.random.normal(0, 0.05, (height, width))
    pm25 = np.clip(pm25 + noise, 0, 1)
    pm25 = (pm25 * 255).astype(np.uint8)
    img = Image.fromarray(pm25)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()

def generate_sa_wind_texture():
    width, height = 512, 512
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    month = datetime.now().month
    is_summer = month in [11, 0, 1, 2]
    if is_summer:
        u = np.ones((height, width)) * 0.7
        v = np.ones((height, width)) * -0.5
    else:
        u = np.ones((height, width)) * -0.6
        v = np.ones((height, width)) * 0.4
    turbulence = np.sin(2*np.pi*x/width*4) * np.cos(2*np.pi*y/height*3) * 0.3
    u += turbulence
    v += turbulence * 0.7
    r = ((u + 1) / 2 * 255).astype(np.uint8)
    g = ((v + 1) / 2 * 255).astype(np.uint8)
    b = np.zeros_like(r)
    wind_img = np.stack([r, g, b], axis=2)
    img = Image.fromarray(wind_img, 'RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()

@app.get("/api/sa/pm25-texture")
async def get_sa_pm25_texture():
    """Returns PM2.5 data for South Africa as PNG"""
    return Response(content=generate_sa_pm25_texture(), media_type="image/png")

@app.get("/api/sa/wind-texture")
async def get_sa_wind_texture():
    """Returns wind data for South Africa as PNG"""
    return Response(content=generate_sa_wind_texture(), media_type="image/png")

@app.get("/api/sa/hotspots")
async def get_sa_hotspots():
    """Returns industrial hotspots in South Africa"""
    features = [
        {
            "type": "Feature",
            "properties": {
                "name": source["name"],
                "intensity": source["intensity"],
                "description": f"Industrial area in {source['name']}"
            },
            "geometry": {
                "type": "Point",
                "coordinates": source["coords"]
            }
        } for source in INDUSTRIAL_SOURCES
    ]
    return {
        "type": "FeatureCollection",
        "features": features
    }

# ---- Mock weather stations (optional, can be removed if you don't want station time series) ----
class WeatherStation(BaseModel):
    id: str
    name: str
    coordinates: List[float]
    elevation: float

class StationDataPoint(BaseModel):
    time: int  # hour 0-23
    temperature: float  # °C
    humidity: float  # %
    wind_speed: float  # km/h
    wind_direction: float  # degrees
    pressure: float  # hPa
    precipitation: float  # mm

class StationDataResponse(BaseModel):
    station: WeatherStation
    data: List[StationDataPoint]
    forecast_hours: int

SA_WEATHER_STATIONS = [
    WeatherStation(id="JNB", name="Johannesburg", coordinates=[28.0473, -26.2041], elevation=1753),
    WeatherStation(id="CPT", name="Cape Town", coordinates=[18.4241, -33.9249], elevation=5),
    WeatherStation(id="DUR", name="Durban", coordinates=[31.0292, -29.8587], elevation=5),
    WeatherStation(id="PTA", name="Pretoria", coordinates=[28.2293, -25.7479], elevation=1339),
    WeatherStation(id="BLO", name="Bloemfontein", coordinates=[26.1596, -29.0852], elevation=1395),
]

def generate_elevation_map(width=512, height=512):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    elevation = 1000 * np.exp(-((x - width*0.6)**2 + (y - height*0.4)**2) / (150**2))
    elevation += 2000 * np.exp(-((x - width*0.65)**2 + (y - height*0.3)**2) / (50**2))
    elevation += 1500 * np.exp(-((x - width*0.3)**2 + (y - height*0.15)**2) / (40**2))
    elevation -= 500 * np.exp(-((x - width*0.1)**2 + (y - height*0.1)**2) / (200**2))
    elevation -= 500 * np.exp(-((x - width*0.9)**2 + (y - height*0.9)**2) / (200**2))
    return np.clip(elevation, 0, 3000) / 3000

def generate_temperature_texture(month, width=512, height=512):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    is_summer = month in [11, 0, 1]  # Dec-Feb
    base_temp = 25 if is_summer else 15
    temp = np.zeros((height, width))
    elevation_map = generate_elevation_map(width, height)
    lat_effect = (1 - y/height) * 10
    elevation_effect = elevation_map * 0.1 * 3000 / 100
    temp += base_temp
    temp -= lat_effect
    temp -= elevation_effect
    noise = np.random.normal(0, 0.5, (height, width))
    temp = np.clip(temp + noise, 0, 40)
    return ((temp - 0) / (40 - 0) * 255).astype(np.uint8)

def generate_precipitation_texture(month, width=512, height=512):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    precip = np.zeros((height, width))
    is_summer = month in [11, 0, 1]
    is_winter = month in [5, 6, 7]
    if is_summer:
        precip += 50 * np.exp(-((x - width*0.7)**2 + (y - height*0.3)**2) / (100**2))
    elif is_winter:
        precip += 60 * np.exp(-((x - width*0.3)**2 + (y - height*0.15)**2) / (80**2))
    else:
        precip += 20 * np.exp(-((x - width*0.5)**2 + (y - height*0.3)**2) / (150**2))
    elevation_map = generate_elevation_map(width, height)
    precip += elevation_map * 0.5 * 100
    noise = np.random.normal(0, 5, (height, width))
    precip = np.clip(precip + noise, 0, 100)
    return (precip / 100 * 255).astype(np.uint8)

def generate_wind_texture(month, width=512, height=512):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    is_summer = month in [11, 0, 1]
    if is_summer:
        u = np.ones((height, width)) * 0.8
        v = np.ones((height, width)) * -0.6
    else:
        u = np.ones((height, width)) * -0.7
        v = np.ones((height, width)) * 0.5
    elevation_map = generate_elevation_map(width, height)
    terrain_effect = np.gradient(elevation_map)
    u += terrain_effect[1] * 0.2
    v += terrain_effect[0] * 0.2
    r = ((u + 1) / 2 * 255).astype(np.uint8)
    g = ((v + 1) / 2 * 255).astype(np.uint8)
    b = np.zeros_like(r)
    wind_img = np.stack([r, g, b], axis=2)
    return wind_img

def generate_cloud_texture(month, width=512, height=512):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    elevation_map = generate_elevation_map(width, height)
    clouds = 0.3 + 0.5 * elevation_map
    clouds += 0.2 * np.exp(-((x - width*0.7)**2 + (y - height*0.3)**2) / (80**2))
    clouds = np.clip(clouds + np.random.normal(0, 0.1, (height, width)), 0, 1)
    arr = (clouds * 255).astype(np.uint8)
    return arr


def generate_station_data(station_id: str, hours: int = 24) -> List[StationDataPoint]:
    station = next((s for s in SA_WEATHER_STATIONS if s.id == station_id), None)
    if not station:
        return []
    base_temp = 15 + (station.elevation / 2000 * 10)
    is_coastal = station.id in ["CPT", "DUR"]
    data = []
    for hour in range(hours):
        temp = base_temp + 10 * np.sin(hour * np.pi / 12)
        temp += np.random.normal(0, 1.5)
        if is_coastal:
            temp = base_temp + 7 * np.sin(hour * np.pi / 12)
        data.append(StationDataPoint(
            time=hour,
            temperature=round(temp, 1),
            humidity=max(30, min(100, 50 + 30 * np.sin(hour * np.pi / 12) + np.random.normal(0, 5))),
            wind_speed=round(5 + 10 * np.random.random(), 1),
            wind_direction=np.random.randint(0, 360),
            pressure=round(1010 + (np.random.random() * 10 - 5), 1),
            precipitation=round(np.random.random() * 5 if np.random.random() > 0.7 else 0, 1)
        ))
    return data

FIXED_MONTH = datetime.now().month

# --- Static Weather Texture Endpoints ---

@app.get("/api/sa/weather-texture/temperature")
async def get_temperature_texture():
    arr = generate_temperature_texture(FIXED_MONTH)
    img = Image.fromarray(arr)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return Response(content=img_bytes.getvalue(), media_type="image/png")

@app.get("/api/sa/weather-texture/precipitation")
async def get_precipitation_texture():
    arr = generate_precipitation_texture(FIXED_MONTH)
    img = Image.fromarray(arr)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return Response(content=img_bytes.getvalue(), media_type="image/png")

@app.get("/api/sa/weather-texture/wind")
async def get_wind_texture():
    arr = generate_wind_texture(FIXED_MONTH)
    img = Image.fromarray(arr, "RGB")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return Response(content=img_bytes.getvalue(), media_type="image/png")

@app.get("/api/sa/weather-texture/cloud")
async def get_cloud_texture():
    arr = generate_cloud_texture(FIXED_MONTH)
    img = Image.fromarray(arr)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return Response(content=img_bytes.getvalue(), media_type="image/png")

# --- Static Station Endpoints ---

@app.get("/api/sa/station/JNB", response_model=StationDataResponse)
async def get_station_jnb():
    station_id = "JNB"
    station = next((s for s in SA_WEATHER_STATIONS if s.id == station_id), None)
    data = generate_station_data(station_id, hours=24)
    return {"station": station, "data": data, "forecast_hours": 24}

@app.get("/api/sa/station/CPT", response_model=StationDataResponse)
async def get_station_cpt():
    station_id = "CPT"
    station = next((s for s in SA_WEATHER_STATIONS if s.id == station_id), None)
    data = generate_station_data(station_id, hours=24)
    return {"station": station, "data": data, "forecast_hours": 24}

@app.get("/api/sa/station/DUR", response_model=StationDataResponse)
async def get_station_dur():
    station_id = "DUR"
    station = next((s for s in SA_WEATHER_STATIONS if s.id == station_id), None)
    data = generate_station_data(station_id, hours=24)
    return {"station": station, "data": data, "forecast_hours": 24}

@app.get("/api/sa/station/PTA", response_model=StationDataResponse)
async def get_station_pta():
    station_id = "PTA"
    station = next((s for s in SA_WEATHER_STATIONS if s.id == station_id), None)
    data = generate_station_data(station_id, hours=24)
    return {"station": station, "data": data, "forecast_hours": 24}

@app.get("/api/sa/station/BLO", response_model=StationDataResponse)
async def get_station_blo():
    station_id = "BLO"
    station = next((s for s in SA_WEATHER_STATIONS if s.id == station_id), None)
    data = generate_station_data(station_id, hours=24)
    return {"station": station, "data": data, "forecast_hours": 24}


@app.get("/api/sa/stations", response_model=List[WeatherStation])
async def get_weather_stations():
    return SA_WEATHER_STATIONS


#code for mock data AQI features:

def generate_sa_pollutant_texture(sources, decay=50, base_intensity=1.0, season_func=None,
                                  background=0.1, width=512, height=512, clip_max=1.0, scale=255):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    texture = np.zeros((height, width))
    # Add background (ambient) everywhere
    texture += background
    for source in sources:
        norm_x = (source["coords"][0] - SA_BOUNDS["min_lon"]) / (SA_BOUNDS["max_lon"] - SA_BOUNDS["min_lon"])
        norm_y = 1 - (source["coords"][1] - SA_BOUNDS["min_lat"]) / (SA_BOUNDS["max_lat"] - SA_BOUNDS["min_lat"])
        tex_x = norm_x * width
        tex_y = norm_y * height
        # More realistic: Use squared exponential for pollutants that disperse widely
        dist = np.sqrt((x - tex_x)**2 + (y - tex_y)**2)
        # Intensity may drop with squared distance for gases (O3, CO), or exponentially for particulates
        intensity = source.get("intensity", 1.0) * base_intensity
        if source.get("squared_falloff", False):
            texture += intensity * np.exp(-((dist / decay)**2))
        else:
            texture += intensity * np.exp(-dist / decay)
    if season_func:
        texture *= season_func(datetime.now().month)
    # Add some random variation (simulates meteorological mixing etc)
    noise = np.random.normal(0, 0.03, (height, width))
    texture = np.clip(texture + noise, 0, clip_max)
    arr = (texture * scale / clip_max).astype(np.uint8)
    img = Image.fromarray(arr)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()

# Add/adjust as needed
TRAFFIC_SOURCES = [
    {"name": "Johannesburg CBD", "coords": [28.0473, -26.2041], "intensity": 1.0},
    {"name": "Pretoria CBD", "coords": [28.2293, -25.7479], "intensity": 0.7},
    {"name": "Durban CBD", "coords": [31.0292, -29.8587], "intensity": 0.6},
    # ...major highways could be simulated as extra sources
]
COAL_POWER_PLANTS = [
    {"name": "Secunda Power", "coords": [29.1667, -26.5167], "intensity": 1.2},
    {"name": "Sasolburg Power", "coords": [27.8167, -26.8167], "intensity": 1.0},
]
INDUSTRIAL_SOURCES_PM = INDUSTRIAL_SOURCES  # reuse your existing list

OZONE_URBAN = [
    {"name": "Pretoria", "coords": [28.2293, -25.7479], "intensity": 0.8, "squared_falloff": True},
    {"name": "Johannesburg", "coords": [28.0473, -26.2041], "intensity": 1.0, "squared_falloff": True},
    {"name": "Durban", "coords": [31.0292, -29.8587], "intensity": 0.7, "squared_falloff": True},
    {"name": "Cape Town", "coords": [18.4241, -33.9249], "intensity": 0.5, "squared_falloff": True},
]

@app.get("/api/sa/pm1-texture")
async def get_sa_pm1_texture():
    # PM1.0: more localized, high in winter
    def pm1_season(month):
        return 0.9 + 0.2 * np.cos(2*np.pi*(month-6)/12)  # peak winter (Jun/Jul)
    return Response(content=generate_sa_pollutant_texture(
        sources=INDUSTRIAL_SOURCES_PM,
        decay=35,  # narrower plumes, more localized
        base_intensity=1.0,
        background=0.05,
        season_func=pm1_season
    ), media_type="image/png")

@app.get("/api/sa/pm10-texture")
async def get_sa_pm10_texture():
    # PM10: can spread wider, also from dust
    def pm10_season(month):
        return 0.95 + 0.15 * np.cos(2*np.pi*(month-6)/12)
    # Add an extra "dust source" in drier regions for realism if you want!
    pm10_sources = INDUSTRIAL_SOURCES_PM + [
        {"name": "Dust Belt", "coords": [24.0, -28.0], "intensity": 0.6}
    ]
    return Response(content=generate_sa_pollutant_texture(
        sources=pm10_sources,
        decay=80,   # wider plumes
        base_intensity=1.1,
        background=0.08,
        season_func=pm10_season
    ), media_type="image/png")

@app.get("/api/sa/o3-texture")
async def get_sa_o3_texture():
    # Ozone: forms from precursors, peaks in summer sunlight, widespread, quadratic decay
    def o3_season(month):
        # Higher in summer: DJF = Dec-Jan-Feb
        return 0.85 + 0.4 * np.sin(2*np.pi*(month-1)/12)
    return Response(content=generate_sa_pollutant_texture(
        sources=OZONE_URBAN,
        decay=120,       # much wider spread
        base_intensity=1.0,
        background=0.3,
        season_func=o3_season,
        clip_max=1.0,
        scale=255
    ), media_type="image/png")

@app.get("/api/sa/no2-texture")
async def get_sa_no2_texture():
    # NO2: mainly from traffic/power plants, winter peaks
    def no2_season(month):
        return 1.0 + 0.3 * np.cos(2*np.pi*(month-6)/12)
    # Add both traffic and power plants for realism
    no2_sources = TRAFFIC_SOURCES + COAL_POWER_PLANTS
    return Response(content=generate_sa_pollutant_texture(
        sources=no2_sources,
        decay=45,
        base_intensity=1.0,
        background=0.06,
        season_func=no2_season
    ), media_type="image/png")

@app.get("/api/sa/so2-texture")
async def get_sa_so2_texture():
    # SO2: almost entirely from coal/petrochemical industry, highest in winter
    def so2_season(month):
        return 1.0 + 0.35 * np.cos(2*np.pi*(month-6)/12)
    return Response(content=generate_sa_pollutant_texture(
        sources=COAL_POWER_PLANTS,
        decay=38,   # very localized
        base_intensity=1.2,
        background=0.02,
        season_func=so2_season
    ), media_type="image/png")


@app.get("/api/sa/co-texture")
async def get_sa_co_texture():
    # CO: from traffic, biomass burning, peaks in winter
    def co_season(month):
        return 0.9 + 0.25 * np.cos(2*np.pi*(month-6)/12)
    co_sources = TRAFFIC_SOURCES + [
        {"name": "Bushfire Zone", "coords": [27.0, -25.0], "intensity": 0.6}
    ]
    return Response(content=generate_sa_pollutant_texture(
        sources=co_sources,
        decay=65,
        base_intensity=1.0,
        background=0.08,
        season_func=co_season
    ), media_type="image/png")



def static_file_or_404(path, media_type=None):
    if os.path.exists(path):
        return FileResponse(path, media_type=media_type)
    return {"error": "File not found", "path": path}

# FIRE
@app.get("/api/sat/fire.png")
def fire_png():
    return static_file_or_404("/data/fire/fire_africa.png", media_type="image/png")

@app.get("/api/sat/fire.tif")
def fire_tif():
    return static_file_or_404("/data/fire/fire_africa.tif", media_type="image/tiff")

# CLOUD
@app.get("/api/sat/cloud.png")
def cloud_png():
    return static_file_or_404("/data/cloud/cloud_africa.png", media_type="image/png")

@app.get("/api/sat/cloud.tif")
def cloud_tif():
    return static_file_or_404("/data/cloud/cloud_africa.tif", media_type="image/tiff")


# WIND (geojson only for wind, or you can add raster/PNG if you generate them)
@app.get("/api/sat/wind.geojson")
def wind_geojson():
    return static_file_or_404("/data/wind/wind_africa.geojson", media_type="application/geo+json")


@app.get("/api/sat/wind.png")
def wind_png():
    # Find the latest quicklook PNG in /data/wind/quicklooks
    pngs = glob.glob("/data/wind/quicklooks/*.png")
    if not pngs:
        return {"error": "No wind quicklook PNG found"}
    latest = max(pngs, key=os.path.getctime)
    return FileResponse(latest, media_type="image/png")


# This should be set via Docker env, not hardcoded!
FIRMS_MAP_KEY = ("9f4a8dc4d22a7341ed4584aebb2adf00")
FIRMS_BBOX = [-20.0, -35.0, 55.0, 38.0]  # Full Africa, [W, S, E, N]
FIRMS_DAYS = 3  # How many past days of fires

def fetch_firms_fires(map_key=FIRMS_MAP_KEY, bbox=FIRMS_BBOX, days=FIRMS_DAYS, product="MODIS_NRT"):
    west, south, east, north = bbox
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{product}/"
        f"{west},{south},{east},{north}/{days}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    # Make sure we actually got data
    if 'acq_date' not in df.columns:
        return None
    df['acq_datetime'] = pd.to_datetime(
        df.acq_date.astype(str) + ' ' + df.acq_time.astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M"
    )
    return df

@app.get("/api/fire/firms-hotspots.geojson")
def get_firms_hotspots_geojson():
    """
    Live fire hotspots (NASA FIRMS, last 3 days) over Africa.
    Set the FIRMS_MAP_KEY as an environment variable in Docker.
    """
    df = fetch_firms_fires()
    if df is None or df.empty:
        return JSONResponse({"error": "No fire data or invalid MAP_KEY"}, status_code=404)
    # Convert to GeoJSON-like dict for simple frontend mapping
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row.longitude), float(row.latitude)]
            },
            "properties": {
                "datetime": row.acq_datetime.isoformat(),
                "brightness": float(row.brightness),
                "satellite": row.satellite,
                "instrument": row.instrument,
                "frp": float(row.frp),
                "confidence": row.confidence,
                "daynight": row.daynight,
            }
        })
    return {"type": "FeatureCollection", "features": features}





@app.get("/api/fire/firms-hotspots.html")
def get_firms_hotspots_html():
    df = fetch_firms_fires()
    if df is None or df.empty:
        return HTMLResponse("<h2>No fire data or invalid MAP_KEY</h2>")
    m = folium.Map(location=[0, 22], zoom_start=3, tiles="CartoDB positron")
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=3 + float(row.frp)**0.5 / 5.0,
            color="orange" if row.daynight == 'D' else "red",
            fill=True, fill_opacity=0.7,
            popup=f"{row.acq_datetime} UTC<br>FRP: {row.frp}"
        ).add_to(m)
    html = m.get_root().render()
    return HTMLResponse(html)



@app.get("/frames/temperature/temperature{timestep}.gif")
def get_temp_frame(timestep: int):
    path = f"/data/frames/temperature/temperature{timestep}.gif"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/frames/temperature/metadata.json")
def get_temp_metadata():
    path = "/data/frames/temperature/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/precipitation/precipitation{timestep}.gif")
def get_precip_frame(timestep: int):
    path = f"/data/frames/precipitation/precipitation{timestep}.gif"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/precipitation/metadata.json")
def get_precip_metadata():
    path = "/data/frames/precipitation/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/relative_humidity/relative_humidity{timestep}.gif")
def get_humidity_frame(timestep: int):
    path = f"/data/frames/relative_humidity/relative_humidity{timestep}.gif"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/relative_humidity/metadata.json")
def get_humidity_metadata():
    path = "/data/frames/relative_humidity/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/cloud_cover/cloud_cover{timestep}.gif")
def get_cloud_frame(timestep: int):
    path = f"/data/frames/cloud_cover/cloud_cover{timestep}.gif"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/cloud_cover/metadata.json")
def get_cloud_metadata():
    path = "/data/frames/cloud_cover/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/visibility/visibility{timestep}.gif")
def get_visibility_frame(timestep: int):
    path = f"/data/frames/visibility/visibility{timestep}.gif"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/visibility/metadata.json")
def get_visibility_metadata():
    path = "/data/frames/visibility/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/wind_speed/wind_speed{timestep}.gif")
def get_wind_speed_frame(timestep: int):
    path = f"/data/frames/wind_speed/wind_speed{timestep}.gif"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/frames/wind_speed/metadata.json")
def get_wind_speed_metadata():
    path = "/data/frames/wind_speed/metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Not found"}, status_code=404)



## Era5

from fastapi import HTTPException
from fastapi.responses import FileResponse

# Map short name to (long name, units)
ERA5_VARS = {
    "t2m":   ("2 metre temperature", "K"),
    "skt":   ("Skin temperature", "K"),
    "stl1":  ("Soil temperature level 1", "K"),
    "stl2":  ("Soil temperature level 2", "K"),
    "stl3":  ("Soil temperature level 3", "K"),
    "stl4":  ("Soil temperature level 4", "K"),
    "asn":   ("Snow albedo", "(0 - 1)"),
    "snowc": ("Snow cover", "%"),
    "rsn":   ("Snow density", "kg m**-3"),
    "sde":   ("Snow depth", "m"),
    "slhf":  ("Time-integrated surface latent heat flux", "J m**-2"),
    "fal":   ("Forecast albedo", "(0 - 1)"),
    "tp":    ("Total precipitation", "m"),
}

# --- Raster PNG Endpoints ---
for short, (long, units) in ERA5_VARS.items():
    exec(f"""
@app.get("/api/era5/raster/{short}" + "/{{hour}}", response_class=FileResponse)
def serve_era5_raster_{short}(hour: int):
    '''
    Raster PNG for {long} [{units}]
    '''
    png_path = f"/data/era5_rasters_bins/{short}_hour{{hour:02d}}_binned.png"
    if not os.path.exists(png_path):
        raise HTTPException(status_code=404, detail="Raster image not found")
    return FileResponse(png_path, media_type="image/png")
""")

# # --- GeoTIFF Endpoints ---
# for short, (long, units) in ERA5_VARS.items():
#     exec(f"""
# @app.get("/api/era5/geotiff/{short}" + "/{{hour}}", response_class=FileResponse)
# def serve_era5_geotiff_{short}(hour: int):
#     '''
#     GeoTIFF for {long} [{units}]
#     '''
#     tif_path = f"/data/era5_geotiffs/{short}_hour{{hour:02d}}.tif"
#     if not os.path.exists(tif_path):
#         raise HTTPException(status_code=404, detail="GeoTIFF not found")
#     return FileResponse(tif_path, media_type="image/tiff")
# """)



INTERP_DIR = "/data/interpolated_rasters"

AVAILABLE_VARS = ["d2m", "t2m", "stl1", "fal", "str", "sp", "tp"]

# ---- GIF endpoints for each variable ----

@app.get("/api/interpolated/gif/d2m")
def get_gif_d2m():
    path = f"{INTERP_DIR}/d2m_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")

@app.get("/api/interpolated/gif/t2m")
def get_gif_t2m():
    path = f"{INTERP_DIR}/t2m_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")

@app.get("/api/interpolated/gif/stl1")
def get_gif_stl1():
    path = f"{INTERP_DIR}/stl1_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")

@app.get("/api/interpolated/gif/fal")
def get_gif_fal():
    path = f"{INTERP_DIR}/fal_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")

@app.get("/api/interpolated/gif/str")
def get_gif_str():
    path = f"{INTERP_DIR}/str_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")

@app.get("/api/interpolated/gif/sp")
def get_gif_sp():
    path = f"{INTERP_DIR}/sp_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")

@app.get("/api/interpolated/gif/tp")
def get_gif_tp():
    path = f"{INTERP_DIR}/tp_animation.gif"
    if not os.path.exists(path):
        raise HTTPException(404, f"GIF not found: {path}")
    return FileResponse(path, media_type="image/gif")


# def index_to_timestamp(var, idx):
#     # Find all timestamps for this variable, sort in order
#     pattern = f"{INTERP_DIR}/{var}_*.tif"
#     files = sorted(glob(pattern))
#     if not files:
#         raise HTTPException(404, f"No files found for variable {var}")
#     if idx < 0 or idx >= len(files):
#         raise HTTPException(404, f"Invalid index {idx} for {var} (found {len(files)} files)")
#     fname = os.path.basename(files[idx])
#     ts = fname[len(var)+1:-4]
#     return ts

# # ---- GeoTIFF endpoints for each variable by timestep number ----

# for var in AVAILABLE_VARS:
#     exec(f"""
# @app.get("/api/interpolated/geotiff/{var}/{{timestep}}")
# def get_geotiff_{var}(timestep: int):
#     path = f"{{INTERP_DIR}}/{var}_{{timestep:02d}}.tif"
#     if not os.path.exists(path):
#         raise HTTPException(404, f"GeoTIFF not found: {{path}}")
#     return FileResponse(path, media_type="image/tiff")
# """)

for var in AVAILABLE_VARS:
    exec(f"""
@app.get("/api/interpolated/cog/{var}")
def get_cog_{var}():
    path = f"{{INTERP_DIR}}/{var}_multiband_cog_clipped.tif"
    if not os.path.exists(path):
        raise HTTPException(404, f"COG not found: {{path}}")
    return FileResponse(path, media_type="image/tiff")
""")


# Example units mapping, update with your actual units if needed
VAR_UNITS = {
    "t2m": "°C",
    "d2m": "°C",
    "stl1": "°C",
    "fal": "unitless",
    "str": "J/m²",
    "sp": "hPa",
    "tp": "mm"
}

from datetime import datetime, timedelta

# --- Hardcoded hourly timestamps for all 48 COG bands ---
COG_BAND_TIMESTAMPS = [
    (datetime(2025, 7, 26, 0, 0, 0) + timedelta(hours=i)).isoformat()
    for i in range(48)
]


for var in AVAILABLE_VARS:
    exec(f"""
@app.get("/api/interpolated/cog/{var}/metadata")
def get_cog_{var}_metadata():
    import os
    import rasterio
    path = f"{{INTERP_DIR}}/{var}_multiband_cog_clipped.tif"
    if not os.path.exists(path):
        raise HTTPException(404, f"COG not found: {{path}}")
    with rasterio.open(path) as src:
        meta = {{
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "nodata": src.nodatavals[0],
            "units": VAR_UNITS.get("{var}", ""),
            "timestamp_range": [COG_BAND_TIMESTAMPS[0], COG_BAND_TIMESTAMPS[-1]]
            }}
    return JSONResponse(meta)
""")


# # ---- GIF endpoints for each variable ----

# for var in AVAILABLE_VARS:
#     exec(f"""
# @app.get("/api/interpolated/gif/{var}")
# def get_gif_{var}():
#     path = f"{{INTERP_DIR}}/{var}_animation.gif"
#     if not os.path.exists(path):
#         raise HTTPException(404, f"GIF not found: {{path}}")
#     return FileResponse(path, media_type="image/gif")
# """)



@app.get("/api/sa/bounds")
async def get_sa_bounds():
    return SA_BOUNDS

@app.get("/")
async def root():
    return {
        "message": "South Africa Weather API",
        "info": {
            "variable_code_legend": {
                "t2m":   "2 metre air temperature [K]",
                "d2m":   "2 metre dewpoint temperature [K]",
                "stl1":  "Soil temperature level 1 [K] (0–7 cm)",
                "fal":   "Forecast albedo [0–1]",
                "str":   "Surface net upward longwave radiation [J/m²]",
                "sp":    "Surface pressure [Pa]",
                "tp":    "Total precipitation [m]",
                # Add more as needed
            },
            "example_timestamp_format": "2025-07-26T00-00"
        },
        "endpoints": {
            # --- Pointwise Grid Data (SA-only, Real Data) ---
            "/temperature": "Pointwise temperature grid (real Open-Meteo data, South Africa)",
            "/rain": "Pointwise precipitation grid (real Open-Meteo data, South Africa)",
            "/relative_humidity": "Pointwise relative humidity grid (real Open-Meteo data, South Africa)",
            "/cloud_cover": "Pointwise cloud cover grid (real Open-Meteo data, South Africa)",
            "/wind-u": "Wind U-component grid (real Open-Meteo data, South Africa)",
            "/wind-v": "Wind V-component grid (real Open-Meteo data, South Africa)",
            "/wind-speed": "Wind speed grid (real Open-Meteo data, South Africa)",
            "/wind-dirn": "Wind direction grid (real Open-Meteo data, South Africa)",

            # --- Animated Weather Raster Layers (Real Data, Mapbox-ready overlays) ---
            "/frames/temperature/temp{timestep}.gif": "Animated temperature overlay frame for timestep (real forecast, for map animation)",
            "/frames/temperature/metadata.json": "Temperature overlay metadata (bounding box, colormap, etc.)",
            "/frames/precipitation/precipitation{timestep}.gif": "Animated precipitation overlay frame for timestep (real forecast, for map animation)",
            "/frames/precipitation/metadata.json": "Precipitation overlay metadata",
            "/frames/relative_humidity/relative_humidity{timestep}.gif": "Animated relative humidity overlay frame (real forecast)",
            "/frames/relative_humidity/metadata.json": "Relative humidity overlay metadata",
            "/frames/cloud_cover/cloud_cover{timestep}.gif": "Animated cloud cover overlay frame (real forecast)",
            "/frames/cloud_cover/metadata.json": "Cloud cover overlay metadata",
            "/frames/visibility/visibility{timestep}.gif": "Animated visibility overlay frame (real forecast)",
            "/frames/visibility/metadata.json": "Visibility overlay metadata",
            "/frames/wind_speed/wind_speed{timestep}.gif": "Animated wind speed overlay frame (real forecast)",
            "/frames/wind_speed/metadata.json": "Wind speed overlay metadata",

            # --- MOCK Texture Endpoints (legacy/test/demo only) ---
            "/api/sa/pm1-texture": "PM1.0 texture PNG (MOCK, synthetic for demo)",
            "/api/sa/pm2.5-texture": "PM2.5 texture PNG (MOCK, synthetic for demo)",
            "/api/sa/pm10-texture": "PM10 texture PNG (MOCK, synthetic for demo)",
            "/api/sa/o3-texture": "Ozone (O3) texture PNG (MOCK, synthetic for demo)",
            "/api/sa/no2-texture": "Nitrogen Dioxide (NO2) texture PNG (MOCK, synthetic for demo)",
            "/api/sa/so2-texture": "Sulfur Dioxide (SO2) texture PNG (MOCK, synthetic for demo)",
            "/api/sa/co-texture": "Carbon Monoxide (CO) texture PNG (MOCK, synthetic for demo)",
            "/api/sa/wind-texture": "Wind vector texture PNG (MOCK, synthetic for demo)",

            # --- MOCK Weather Station & Static Textures (synthetic demo) ---
            "/api/sa/hotspots": "Industrial hotspots GeoJSON (MOCK, demo)",
            "/api/sa/stations": "Mock weather station metadata (MOCK, demo)",
            "/api/sa/station/JNB": "Mock station JNB timeseries (MOCK, demo)",
            "/api/sa/station/CPT": "Mock station CPT timeseries (MOCK, demo)",
            "/api/sa/station/DUR": "Mock station DUR timeseries (MOCK, demo)",
            "/api/sa/station/PTA": "Mock station PTA timeseries (MOCK, demo)",
            "/api/sa/station/BLO": "Mock station BLO timeseries (MOCK, demo)",

            "/api/sa/weather-texture/temperature": "Temperature texture PNG (MOCK, synthetic for demo)",
            "/api/sa/weather-texture/precipitation": "Precipitation texture PNG (MOCK, synthetic for demo)",
            "/api/sa/weather-texture/wind": "Wind texture PNG (RGB, MOCK, synthetic for demo)",
            "/api/sa/weather-texture/cloud": "Cloud cover texture PNG (MOCK, synthetic for demo)",

            # --- Satellite and Fire Data (external/real) ---
            "/api/sa/cloud-latest.png": "Latest satellite cloud PNG (from EUMETSAT)",
            "/api/sa/fire-latest.png": "Latest satellite fire PNG (from EUMETSAT)",
            "/api/sa/wind-latest.png": "Latest satellite wind PNG (from EUMETSAT)",
            "/api/sa/cloud-latest.geojson": "Latest cloud GeoJSON (from EUMETSAT)",
            "/api/sa/fire-latest.geojson": "Latest fire GeoJSON (from EUMETSAT)",
            "/api/sa/wind-latest.geojson": "Latest wind GeoJSON (from EUMETSAT)",

            "/api/sa/bounds": "SA bounding box (lon/lat)",

            # --- ERA5 Raster PNGs and GeoTIFFs ---
            "/api/era5/raster/t2m/{hour}": "2 metre temperature [K] (PNG color-binned)",
            "/api/era5/raster/skt/{hour}": "Skin temperature [K] (PNG color-binned)",
            "/api/era5/raster/stl1/{hour}": "Soil temperature level 1 [K] (PNG color-binned)",
            "/api/era5/raster/stl2/{hour}": "Soil temperature level 2 [K] (PNG color-binned)",
            "/api/era5/raster/stl3/{hour}": "Soil temperature level 3 [K] (PNG color-binned)",
            "/api/era5/raster/stl4/{hour}": "Soil temperature level 4 [K] (PNG color-binned)",
            "/api/era5/raster/asn/{hour}": "Snow albedo [(0-1)] (PNG color-binned)",
            "/api/era5/raster/snowc/{hour}": "Snow cover [%] (PNG color-binned)",
            "/api/era5/raster/rsn/{hour}": "Snow density [kg m**-3] (PNG color-binned)",
            "/api/era5/raster/sde/{hour}": "Snow depth [m] (PNG color-binned)",
            "/api/era5/raster/slhf/{hour}": "Time-integrated surface latent heat flux [J m**-2] (PNG color-binned)",
            "/api/era5/raster/fal/{hour}": "Forecast albedo [(0-1)] (PNG color-binned)",
            "/api/era5/raster/tp/{hour}": "Total precipitation [m] (PNG color-binned)",

            # --- Interpolated (new) GeoTIFFs and GIFs ---
            "/api/interpolated/geotiff/{var}/{timestamp}":
                "GeoTIFF for variable `{var}` at time `{timestamp}`. See variable_code_legend for meaning.",
            "/api/interpolated/geotiff/{var}/index/{idx}":
                "GeoTIFF for variable `{var}` by time index `{idx}` (0-based, sorted by timestamp).",
            "/api/interpolated/gif/{var}":
                "GIF animation for variable `{var}` (all available timesteps).",

            

        }
    }
