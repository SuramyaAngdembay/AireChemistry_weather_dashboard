FROM python:3.10-slim
WORKDIR /app
COPY build_grids.py .
RUN pip install numpy requests tqdm
CMD ["python", "build_grids.py"]

