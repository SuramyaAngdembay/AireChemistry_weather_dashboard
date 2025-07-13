from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
from datetime import datetime

app = FastAPI(title="South Africa Weather/Pollution API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------- Standard Weather Grid Endpoints -----------

LON_MIN, LON_MAX, DLON = -180, 180, 3
LAT_MIN, LAT_MAX, DLAT = -90, 90, 3
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




@app.get("/api/sa/bounds")
async def get_sa_bounds():
    return SA_BOUNDS

@app.get("/")
async def root():
    return {
        "message": "South Africa Weather API",
        "endpoints": {
            "/temperature": "Pointwise temp grid",
            "/rain": "Pointwise rain grid",
            "/api/sa/pm25-texture": "PM2.5 PNG",
            "/api/sa/wind-texture": "Wind PNG",
            "/api/sa/hotspots": "Hotspots GeoJSON",
            "/api/sa/stations": "Mock station meta",
            "/api/sa/station/JNB": "Mock station JNB timeseries",
            "/api/sa/weather-texture/temperature": "Temp PNG",
            "/api/sa/weather-texture/precipitation": "Precip PNG",
            "/api/sa/weather-texture/wind": "Wind PNG",
            "/api/sa/weather-texture/cloud": "Cloud PNG",
        }
    }