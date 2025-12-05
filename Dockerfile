FROM python:3.10-slim

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 libv4l-dev v4l-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY app.py .
COPY best.onnx .

EXPOSE 8000 8501

# Start FastAPI in the background and Streamlit UI in the foreground
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
