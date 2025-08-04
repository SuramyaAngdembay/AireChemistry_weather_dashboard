FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgdal-dev gdal-bin libexpat1 && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir eumdac xarray netCDF4 pillow numpy rasterio pyproj matplotlib

CMD ["python", "fetch_satellite_data.py"]
