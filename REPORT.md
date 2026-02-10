# Benji Bananas Autonomous Agent - Final Report

## Executive Summary
This project successfully implements a Deep Reinforcement Learning (DRL) agent capable of playing *Benji Bananas*, a physics-based infinite runner. By leveraging **Proximal Policy Optimization (PPO)** and custom **Computer Vision** pipelines, the agent learns to control momentum and navigate procedurally generated terrain entirely from raw pixel input.

## System Architecture

### 1. Perception (Computer Vision)
- **Input**: Raw H.264 video stream via `scrcpy`.
- **Preprocessing**:
    - **Grayscale Conversion**: Reduces dimensionality.
    - **Resizing**: Downscaled to 128x128 for CNN efficiency.
    - **Frame Stacking (k=4)**: Enables the network to perceive velocity and acceleration.
- **OCR Reward**: custom template matching and digit recognition extracts the in-game score to guide the learning process.

### 2. Decision (Reinforcement Learning)
- **Algorithm**: PPO (Proximal Policy Optimization).
- **Network**: Custom 4-layer CNN (NatureCNN variant) + MLP Policy Head.
- **Action Space**: Discrete (Hold/Release) with timer-based frequency control (30 Hz).
- **Hyperparameters**: Tuned for long-horizon planning (`gamma=0.999`, `ent_coef=0.05`).

### 3. Execution (Control)
- **Interface**: Direct TCP socket connection to Scrcpy server on the Android device.
- **Latency**: Sub-50ms round-trip time suitable for real-time control.

### 4. Training Strategy
- **Imitation Learning (Warm Start)**:
    - **Data Collection**: Collected expert gameplay frames and actions.
    - **Behavioral Cloning (BC)**: Pre-trained the CNN policy using Supervised Learning (`bc.py`) to mimic human play. This provided a stable starting point for RL, avoiding the initial "random flailing" phase.
- **Reinforcement Learning (Fine-tuning)**:
    - **PPO**: The pre-trained weights were loaded into the PPO agent, which then continued training to optimize the reward function and surpass human performance.

## MLOps & Reproducibility
- **Dockerized**: The entire training stack is containerized for portability.
- **CI/CD**: GitHub Actions pipeline ensures code quality and environment integrity.
- **Monitoring**: Tensorboard integration for real-time metric tracking.

## Future Work
- **Domain Randomization**: Augmenting colors/textures to make the agent robust to different game themes.


---
**Created by**: Sirish Gurung
**Date**: February 2026
