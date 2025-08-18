FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (e.g., ffmpeg if whisper needs it for other formats, though faster-whisper might not directly)
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hfwhisperapi.py .

ENV XDG_CACHE_HOME=/app/.cache
RUN mkdir -p $XDG_CACHE_HOME/hf_cache # For Hugging Face model cache by faster-whisper
RUN mkdir -p $XDG_CACHE_HOME/torch_cache # For PyTorch hub models if any
VOLUME $XDG_CACHE_HOME/hf_cache
VOLUME $XDG_CACHE_HOME/torch_cache


# Pre-download models (optional)
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('Darejkal/vv-pp-ww-mm-cc-22')"

ENV PORT=8000
EXPOSE $PORT

CMD ["python", "hfwhisperapi.py"]
