from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from datetime import datetime
from typing import List


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

LON_MIN, LON_MAX, DLON = -180, 180, 3
LAT_MIN, LAT_MAX, DLAT = -90, 90, 3
lon_vals = np.arange(LON_MIN, LON_MAX + DLON, DLON)
lat_vals = np.arange(LAT_MIN, LAT_MAX + DLAT, DLON)
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

@app.get("/visibility")
def visibility():
    grid = np.load("/data/visibility.npy")
    return JSONResponse(grid_to_point_list(grid))

@app.get("/cloud_cover")
def cloud_cover():
    grid = np.load("/data/cloud_cover.npy")
    return JSONResponse(grid_to_point_list(grid))
    
@app.get("/wind-u")
def wind_u():
    return serve(
        "/data/wind_u.npy", 1, 2,
        "u-component of wind [m/s]", "[m/s]"
    )

@app.get("/wind-v")
def wind_v():
    return serve(
        "/data/wind_v.npy", 1, 3,
        "v-component of wind [m/s]", "[m/s]"
    )

@app.get("/wind-speed")
def wind_speed():
    return serve(
        "/data/wind_speed.npy", 1, 3,
        "wind speed [m/s]", "[m/s]"
    )

@app.get("/wind-dirn")
def wind_direction():
    return serve(
        "/data/wind_direction.npy", 1, 3,
        "wind dirn ", "degrees"
    )
    
# South Africa bounding box
SA_BOUNDS = {
    "min_lon": 16.45,
    "max_lon": 32.89,
    "min_lat": -34.83,
    "max_lat": -22.13
}

# Major industrial areas in South Africa
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
    
    # Initialize empty pollution field
    pm25 = np.zeros((height, width))
    
    # Add pollution sources for each industrial area
    for source in INDUSTRIAL_SOURCES:
        # Normalize coordinates to texture space
        norm_x = (source["coords"][0] - SA_BOUNDS["min_lon"]) / (SA_BOUNDS["max_lon"] - SA_BOUNDS["min_lon"])
        norm_y = 1 - (source["coords"][1] - SA_BOUNDS["min_lat"]) / (SA_BOUNDS["max_lat"] - SA_BOUNDS["min_lat"])
        
        tex_x = norm_x * width
        tex_y = norm_y * height
        
        # Add gaussian plume for this source
        dist = np.sqrt((x - tex_x)**2 + (y - tex_y)**2)
        pm25 += source["intensity"] * np.exp(-dist / (50 + np.random.random() * 30))
    
    # Add seasonal variation (worse in winter)
    month = datetime.now().month
    season_factor = 0.7 + 0.3 * np.cos(2*np.pi*(month-6)/12)
    pm25 *= season_factor
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.05, (height, width))
    pm25 = np.clip(pm25 + noise, 0, 1)
    
    # Convert to 8-bit
    pm25 = (pm25 * 255).astype(np.uint8)
    img = Image.fromarray(pm25)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def generate_sa_wind_texture():
    width, height = 512, 512
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    
    # Prevailing winds in South Africa:
    # - Summer: SE winds
    # - Winter: NW winds
    month = datetime.now().month
    is_summer = month in [11, 0, 1, 2]  # Nov-Feb
    
    if is_summer:
        # Southeast winds
        u = np.ones((height, width)) * 0.7  # East component
        v = np.ones((height, width)) * -0.5  # South component
    else:
        # Northwest winds
        u = np.ones((height, width)) * -0.6  # West component
        v = np.ones((height, width)) * 0.4    # North component
    
    # Add some turbulence
    turbulence = np.sin(2*np.pi*x/width*4) * np.cos(2*np.pi*y/height*3) * 0.3
    u += turbulence
    v += turbulence * 0.7
    
    # Normalize to [0,255]
    r = ((u + 1) / 2 * 255).astype(np.uint8)  # East-West
    g = ((v + 1) / 2 * 255).astype(np.uint8)  # North-South
    b = np.zeros_like(r)                      # Unused
    
    wind_img = np.stack([r, g, b], axis=2)
    img = Image.fromarray(wind_img, 'RGB')
    
    # Save to bytes
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







# # Models
# class WeatherStation(BaseModel):
#     id: str
#     name: str
#     coordinates: List[float]
#     elevation: float

# class StationDataPoint(BaseModel):
#     time: int  # hour 0-23
#     temperature: float  # °C
#     humidity: float  # %
#     wind_speed: float  # km/h
#     wind_direction: float  # degrees
#     pressure: float  # hPa
#     precipitation: float  # mm

# class StationDataResponse(BaseModel):
#     station: WeatherStation
#     data: List[StationDataPoint]
#     forecast_hours: int

# Cache for generated textures
TEXTURE_CACHE = {}

def generate_elevation_map(width=512, height=512):
    """Generate elevation map for South Africa"""
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    
    # Highveld region (eastern plateau)
    elevation = 1000 * np.exp(-((x - width*0.6)**2 + (y - height*0.4)**2) / (150**2))
    
    # Drakensberg mountains
    elevation += 2000 * np.exp(-((x - width*0.65)**2 + (y - height*0.3)**2) / (50**2))
    
    # Cape Fold Mountains
    elevation += 1500 * np.exp(-((x - width*0.3)**2 + (y - height*0.15)**2) / (40**2))
    
    # Coastal areas are low
    elevation -= 500 * np.exp(-((x - width*0.1)**2 + (y - height*0.1)**2) / (200**2))
    elevation -= 500 * np.exp(-((x - width*0.9)**2 + (y - height*0.9)**2) / (200**2))
    
    return np.clip(elevation, 0, 3000) / 3000  # Normalize to 0-1

def generate_temperature_texture(month, width=512, height=512):
    """Generate temperature texture for South Africa"""
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    
    # Base temperature pattern
    is_summer = month in [11, 0, 1]  # Dec-Feb
    base_temp = 25 if is_summer else 15
    
    temp = np.zeros((height, width))
    elevation_map = generate_elevation_map(width, height)
    
    # Latitude effect (colder south)
    lat_effect = (1 - y/height) * 10
    
    # Elevation effect (-1°C per 100m)
    elevation_effect = elevation_map * 0.1 * 3000 / 100  # Convert normalized elevation to meters
    
    # Generate temperature field
    temp += base_temp
    temp -= lat_effect
    temp -= elevation_effect
    
    # Add noise for realism
    noise = np.random.normal(0, 0.5, (height, width))
    temp = np.clip(temp + noise, 0, 40)
    
    # Normalize to 0-255
    return ((temp - 0) / (40 - 0) * 255).astype(np.uint8)

def generate_precipitation_texture(month, width=512, height=512):
    """Generate precipitation texture for South Africa"""
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    
    precip = np.zeros((height, width))
    is_summer = month in [11, 0, 1]
    is_winter = month in [5, 6, 7]
    
    if is_summer:
        # Summer rainfall in northeast
        precip += 50 * np.exp(-((x - width*0.7)**2 + (y - height*0.3)**2) / (100**2))
    elif is_winter:
        # Winter rainfall in southwest (Cape)
        precip += 60 * np.exp(-((x - width*0.3)**2 + (y - height*0.15)**2) / (80**2))
    else:
        # Transition seasons
        precip += 20 * np.exp(-((x - width*0.5)**2 + (y - height*0.3)**2) / (150**2))
    
    # Orographic precipitation
    elevation_map = generate_elevation_map(width, height)
    precip += elevation_map * 0.5 * 100  # Scale to mm
    
    # Add noise
    noise = np.random.normal(0, 5, (height, width))
    precip = np.clip(precip + noise, 0, 100)
    
    return (precip / 100 * 255).astype(np.uint8)


def generate_weather_texture(parameter: str, month: int = None, width=512, height=512):
    """Generate weather texture for the given parameter"""
    month = month if month is not None else datetime.now().month - 1  # 0-based
    
    if parameter == "temperature":
        arr = generate_temperature_texture(month, width, height)
        return Image.fromarray(arr)
    elif parameter == "precipitation":
        arr = generate_precipitation_texture(month, width, height)
        return Image.fromarray(arr)
    elif parameter == "wind":
        arr = generate_wind_texture(month, width, height)
        return Image.fromarray(arr)
    elif parameter == "pressure":
        # Generate pressure texture (simplified)
        x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
        is_summer = month in [11, 0, 1]
        pressure = 1010 + 10 * np.exp(-((x - width*0.6)**2 + (y - height*0.4)**2) / (100**2))
        if not is_summer:
            pressure = 1020 - pressure + 1010  # Invert for winter
        pressure = np.clip(pressure + np.random.normal(0, 0.5, (height, width)), 980, 1040)
        arr = ((pressure - 980) / (1040 - 980) * 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif parameter == "cloud":
        # Generate cloud cover texture
        x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
        elevation_map = generate_elevation_map(width, height)
        clouds = 0.3 + 0.5 * elevation_map  # More clouds over mountains
        clouds += 0.2 * np.exp(-((x - width*0.7)**2 + (y - height*0.3)**2) / (80**2))  # Coastal
        clouds = np.clip(clouds + np.random.normal(0, 0.1, (height, width)), 0, 1)
        arr = (clouds * 255).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        raise ValueError(f"Unknown parameter: {parameter}")

# def generate_station_data(station_id: str, hours: int = 24) -> List[StationDataPoint]:
#     """Generate mock station data for the given hours"""
#     station = next((s for s in SA_WEATHER_STATIONS if s["id"] == station_id), None)
#     if not station:
#         return []
    
#     base_temp = 15 + (station["elevation"] / 2000 * 10)  # Colder at higher elevations
#     is_coastal = station["id"] in ["CPT", "DUR"]
    
#     data = []
#     for hour in range(hours):
#         # Daily temperature cycle
#         temp = base_temp + 10 * np.sin(hour * np.pi / 12)
        
#         # Add some randomness
#         temp += np.random.normal(0, 1.5)
        
#         # Coastal areas are more moderate
#         if is_coastal:
#             temp = base_temp + 7 * np.sin(hour * np.pi / 12)
        
#         data.append(StationDataPoint(
#             time=hour,
#             temperature=round(temp, 1),
#             humidity=max(30, min(100, 50 + 30 * np.sin(hour * np.pi / 12) + np.random.normal(0, 5))),
#             wind_speed=round(5 + 10 * np.random.random(), 1),
#             wind_direction=np.random.randint(0, 360),
#             pressure=round(1010 + (np.random.random() * 10 - 5), 1),
#             precipitation=round(np.random.random() * 5 if np.random.random() > 0.7 else 0, 1)
#         ))
    
#     return data

# @app.get("/api/sa/stations", response_model=List[WeatherStation])
# async def get_weather_stations():
#     """Get list of all weather stations in South Africa"""
#     return SA_WEATHER_STATIONS

# @app.get("/api/sa/station/{station_id}", response_model=StationDataResponse)
# async def get_station_data(station_id: str, hours: int = 24):
#     """Get weather data for a specific station"""
#     station = next((s for s in SA_WEATHER_STATIONS if s["id"] == station_id), None)
#     if not station:
#         raise HTTPException(status_code=404, detail="Station not found")
    
#     data = generate_station_data(station_id, hours)
#     return {
#         "station": station,
#         "data": data,
#         "forecast_hours": hours
#     }

@app.get("/api/sa/weather-texture/{parameter}")
async def get_weather_texture(parameter: str, month: int = None):
    """Get weather texture image for visualization"""
    valid_params = ["temperature", "precipitation", "wind", "pressure", "cloud"]
    if parameter not in valid_params:
        raise HTTPException(status_code=400, detail="Invalid parameter")
    
    # Check cache first
    cache_key = f"{parameter}_{month if month else 'current'}"
    if cache_key in TEXTURE_CACHE:
        return TEXTURE_CACHE[cache_key]
    
    # Generate new texture
    img = generate_weather_texture(parameter, month)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    # Cache for 15 minutes
    TEXTURE_CACHE[cache_key] = FileResponse(img_bytes, media_type="image/png")
    
    return TEXTURE_CACHE[cache_key]

@app.get("/api/sa/bounds")
async def get_sa_bounds():
    """Get the geographic bounds of South Africa"""
    return SA_BOUNDS

@app.get("/")
async def root():
    return {
        "message": "South Africa Weather API",
        "endpoints": {
            "/api/sa/stations": "List of weather stations",
            "/api/sa/station/{id}": "Get station data",
            "/api/sa/weather-texture/{param}": "Get weather texture image",
            "/api/sa/bounds": "Get SA geographic bounds"
        }
    }