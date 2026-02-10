# Base Image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install System Dependencies
# android-tools-adb: For communicating with emulator/device
# ffmpeg: For decoding Scrcpy stream
# libgl1-mesa-glx: Required by OpenCV
# build-essential: For compiling some python packages
RUN apt-get update && apt-get install -y \
    adb \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    git \
    pkg-config \
    meson \
    ninja-build \
    libsdl2-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavformat-dev \
    libavutil-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Build scrcpy from source
WORKDIR /tmp
RUN git clone https://github.com/Genymobile/scrcpy
WORKDIR /tmp/scrcpy
RUN ./install_release.sh
WORKDIR /app

# Set Working Directory
WORKDIR /app

# Install Python Dependencies from pyproject.toml
# We copy only the config first to leverage Docker cache
COPY pyproject.toml .
# We need to install 'pip-tools' or just use pip directly if pyproject.toml is supported
# Since it's a simple project, we can convert to requirements or just install "."
# Let's install current directory as editable or standard
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy Source Code
COPY . .

# Create directories for artifacts
RUN mkdir -p logs models

# Entrypoint to handle ADB connection
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]

# Default Command
# Play with the model by default
CMD ["python", "play.py"]
