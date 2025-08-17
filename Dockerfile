FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OpenCV/Tesseract/pyzbar deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libzbar0 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch nutrition detection model
RUN curl -L -o dungeon_web/nutrition/frozen_inference_graph.pb \
    https://raw.githubusercontent.com/openfoodfacts/open-nutrition-table-extractor/master/data/frozen_inference_graph.pb


COPY . .
EXPOSE 6969
CMD ["python", "server.py"]
