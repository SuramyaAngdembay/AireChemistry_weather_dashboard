FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (libexpat1 is the minimum for rasterio)
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
