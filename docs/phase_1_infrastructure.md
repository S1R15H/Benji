# Phase 1: Infrastructure & Environment Setup

## 1. Overview
This document outlines the foundational work completed in **Phase 1** of the `benji-rl` project. The primary objective was to establish a high-performance, low-latency visual connection between the Android environment (running *Benji Bananas*) and our Python-based Reinforcement Learning agent.

The system uses a pipeline approach to stream video data efficiently:
`Android Device` -> `ADB` -> `Scrcpy` -> `Named Pipe (FIFO)` -> `FFmpeg` -> `Python (NumPy)`

## 2. Importance
Establishing this reliable infrastructure is critical because:
*   **Low Latency**: RL agents require immediate feedback. Standard screen capture methods (like ADB screencap) are too slow (hundreds of ms). Our pipeline achieves real-time streaming.
*   **Raw Data Access**: We need programmatic access to pixel data (NumPy arrays) for the Neural Network, not just a window on the screen.
*   **Stability**: The connection must be robust enough to handle thousands of frames without desyncing or crashing.

## 3. Tools Used
*   **ADB (Android Debug Bridge)**: The core communication link between the PC and the Android device over USB or TCP/IP.
*   **Scrcpy**: A high-performance screen mirroring tool. We use it to capture the H.264 video stream from the device.
*   **FFmpeg**: A powerful multimedia framework used to decode the raw H.264 stream from Scrcpy into raw RGB/BGR pixel data that Python can consume.
*   **OS Named Pipes (FIFO)**: A mechanism to pass data between processes without writing to disk or risking buffer deadlocks.

## 4. Phase 1 Implementation & "Broken Pipe" Fix

### The Challenge
Initially, we attempted to pipe `scrcpy` output directly to `ffmpeg` via standard output (`stdout`). However, we encountered severe stability issues:
1.  **Log Pollution**: `scrcpy` prints informational logs to `stdout` alongside the video data, corrupting the stream for `ffmpeg`.
2.  **Buffering Deadlocks**: Standard pipes can fill up if the consumer (`ffmpeg`) is slower or waiting for a header, causing the producer (`scrcpy`) to hang.

### The Solution: Named Pipes (FIFO)
We refactored the `ScrcpyClient` to use a **OS Named Pipe (FIFO)**.
1.  We create a temporary FIFO functionality in `/tmp/`.
2.  We instruct `scrcpy` to write the video stream to this file path.
3.  We instruct `ffmpeg` to read from this same file path.
4.  We silenced `scrcpy` logs using `-V error`.

This ensures a clean separation of data and control messages, resulting in a stable stream.

### Key Code Snippet
Here is the robust initialization logic from `src/env/scrcpy_client.py`:

```python
import os
import uuid
import subprocess

def start(self):
    # 1. Create a unique FIFO path to avoid collisions
    self._fifo_path = f"/tmp/scrcpy_fifo_{uuid.uuid4().hex}"
    
    if os.path.exists(self._fifo_path):
        os.remove(self._fifo_path)
    os.mkfifo(self._fifo_path)
    
    # 2. Start Scrcpy writing to the FIFO
    # -V error: Suppress logs
    # --record self._fifo_path: Write video to the named pipe
    scrcpy_cmd = [
        "scrcpy", 
        "--max-size", str(self.max_width),
        "--no-audio", "--no-window",
        "-V", "error",
        "--record", self._fifo_path,
        "--record-format", "mkv"
    ]
    self.scrcpy_process = subprocess.Popen(scrcpy_cmd, ...)

    # 3. Start FFmpeg reading from the FIFO
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", self._fifo_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        ...
    ]
    self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, ...)
```

## 5. Input Injection & Latency Optimization

### The Challenge
Standard `adb shell input tap x y` commands are too slow for real-time reinforcement learning, introducing 100ms+ latency per action. This delay makes it impossible for an agent to react to fast-moving game physics.

### The Solution: Persistent ADB Shell
We implemented a **Persistent ADB Shell** mechanism in `ScrcpyClient`.
1.  Instead of spawning a new `subprocess` for every tap (expensive OS operation), we keep a single `subprocess.Popen(["adb", "shell"], stdin=PIPE)` running.
2.  We write commands (e.g., `input tap 100 200\n`) directly to the shell's standard input.
3.  We handle broken pipes with automatic reconnection logic.

### Results
-   **Baseline Latency**: ~134 ms/tap
-   **Optimized Latency**: ~0.14 ms/tap

This improvement guarantees that the bottleneck for reaction time will be the Neural Network inference and game rendering, not the I/O system.

## 6. Next Steps
With the infrastructure (Vision + Control) fully verified, we are ready for Phase II:
1.  **Gym Wrapper**: Wrap this client in a standard Gymnasium interface (`step()`, `reset()`).
2.  **Data Collection**: Use this fast interface to collect expert gameplay data.
