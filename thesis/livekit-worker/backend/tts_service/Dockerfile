FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any (e.g., for onnxruntime-gpu if using CUDA)
# RUN apt-get update && apt-get install -y --no-install-recommends some-lib
# For CPU, typically no extra system deps are needed for piper-tts's onnxruntime

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY piperttsapi.py .

# Set up a cache directory for models and make it a volume
ENV XDG_CACHE_HOME=/app/.cache
RUN mkdir -p $XDG_CACHE_HOME/cached_path
VOLUME $XDG_CACHE_HOME/cached_path

# Pre-download models (optional, but good for image build time if models are stable)
# RUN python -c "from cached_path import cached_path; cached_path('hf://Darejkal/kokoko/cahaibenhaycogangobennhau.onnx'); cached_path('hf://Darejkal/kokoko/model.onnx.json')"

ENV PORT=8000
EXPOSE $PORT

CMD ["python", "piperttsapi.py"]
