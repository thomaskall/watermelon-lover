FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcamera-dev \
    libcamera-apps \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libopencv-dev \
    python3-opencv \
    python3-pyaudio \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils \
    pulseaudio \
    libgl1-mesa-glx \
    libgtk-3-0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Set entrypoint
ENTRYPOINT ["python", "data_collection.py"]

# Default command (can be overridden)
CMD ["--help"]