FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OpenCV/Tesseract/pyzbar deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr tesseract-ocr-eng libzbar0 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Fetch nutrition detection model
RUN mkdir -p nutrition/nutrition_extractor/data && \
    curl -L -o nutrition/nutrition_extractor/data/frozen_inference_graph.pb \
    https://github.com/openfoodfacts/off-nutrition-table-extractor/raw/master/nutrition_extractor/data/frozen_inference_graph.pb
EXPOSE 6969
CMD ["python", "server.py"]
