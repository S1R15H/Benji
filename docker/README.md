# Docker Usage Guide

## Prerequisites
- Docker & Docker Compose installed.
- An Android Emulator (BlueStacks or AVD) running on your host machine.
- ADB connected to the emulator.

## Build the Image
```bash
docker build -t benji-rl .
```

## Run Training
```bash
docker compose up
```

## Connecting to Host ADB (Mac/Windows Workaround)
Since Docker on Mac runs in a VM, `--network host` doesn't pass through USB/ADB effortlessly. 

**Recommended Workflow for Mac:**
1. Ensure your host `adb` can see the emulator (`adb devices`).
2. Kill the *host* adb server so the container can claim the port? **NO**, that's hard.
3. **Easier (TCP method):**
   - Enable ADB over TCP on your emulator.
   - Initializing code inside the container runs `adb connect host.docker.internal:5555`.

*Note: For this project, Scrcpy (Python client) relies on invoking the `adb` binary. If running in Docker, the internal `adb` needs to see the device.*
