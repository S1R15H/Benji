# Benji Bananas RL Agent

This project implements a Deep Reinforcement Learning agent to play *Benji Bananas*.

## Prerequisites

Before running the code, you must install the following system dependencies.

### 1. Install Android Tools and Scrcpy
The project relies on `scrcpy` for low-latency screen capture.

**On macOS (Homebrew):**
```bash
brew install android-platform-tools scrcpy ffmpeg
```

## Quick Start (Docker)
The easiest way to run the training is via Docker.

1. **Prerequisites**: Android Emulator running and connected to ADB (`adb devices`).
2. **Build & Run**:
   ```bash
   docker compose up --build
   ```
   This will build the container and start training. Logs will appear in `./logs`.

## Manual Installation
1. Install dependencies:
   ```bash
   pip install .
   ```
2. Start Training:
   ```bash
   python train.py
   ```

### 2. Connect Android Device
1. Enable **Developer Options** and **USB Debugging** on your Android device (or Emulator).
2. Connect via USB.
3. Verify connection:
   ```bash
   adb devices
   ```
   *You should see a device ID listed.*

### 3. Install Python Dependencies
```bash
pip install -e .[dev]
```

## Usage
(To be added)
