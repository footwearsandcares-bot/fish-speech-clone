FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg git build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install Fish-Speech (Standard install, no extras to save space)
RUN git clone https://github.com/fishaudio/fish-speech.git . && \
    pip3 install --no-cache-dir . 

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Weights baking
RUN pip3 install --no-cache-dir modelscope && \
    python3 -c "from modelscope import snapshot_download; \
    snapshot_download('fishaudio/fish-speech-1.4', local_dir='checkpoints/s1-mini')"

COPY handler.py .
CMD ["python3", "-u", "handler.py"]