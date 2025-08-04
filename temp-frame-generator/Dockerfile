FROM python:3.10-slim

WORKDIR /app

COPY generate_temp_frames.py .

RUN pip install numpy pillow matplotlib

CMD ["python", "generate_temp_frames.py"]
