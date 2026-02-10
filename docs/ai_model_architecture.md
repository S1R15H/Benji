# AI Model & Architecture Documentation

This document details the neural network architecture, algorithm, and tooling choices for the Benji Bananas autonomous agent.

## 1. Core Algorithm: Proximal Policy Optimization (PPO)

We use **PPO (Proximal Policy Optimization)**, a policy gradient method from OpenAI.

### Why PPO?
*   **Stability**: Unlike DQN (which can diverge) or A2C (which can be noisy), PPO constrains the policy update step to prevent drastic changes that collapse training.
*   **Sample Efficiency**: It strikes a balance between sample complexity and wall-clock training time.
*   **Robustness**: It works well "out of the box" with default hyperparameters for a wide variety of environments.

---

## 2. Network Architecture: NatureCNN

We use the standard **"NatureCNN"** architecture (Mnih et al., 2015), made famous by the original DeepMind Atari achievements. This is the default feature extractor in Stable Baselines 3.

### Input Layer
*   **Shape**: `(4, 128, 128)`
    *   **4 Processed Frames** stacked together (Frame Stacking).
    *   **Resolution**: 128x128 pixels.
    *   **Color**: Grayscale (1 channel).
*   **Purpose**: Stacking 4 frames allows the network to perceive **velocity** and **acceleration** (swing momentum) from static images.

### Feature Extractor (Encoder)
The visual input passes through three Convolutional Neural Network (CNN) layers to extract spatial features (edges, curves, vine shapes).

1.  **Conv Layer 1**:
    *   **Filters**: 32
    *   **Kernel Size**: 8x8
    *   **Stride**: 4
    *   **Activation**: ReLU
2.  **Conv Layer 2**:
    *   **Filters**: 64
    *   **Kernel Size**: 4x4
    *   **Stride**: 2
    *   **Activation**: ReLU
3.  **Conv Layer 3**:
    *   **Filters**: 64
    *   **Kernel Size**: 3x3
    *   **Stride**: 1
    *   **Activation**: ReLU
4.  **Flatten Layer**: Converts the 3D feature maps into a 1D vector (size 3136).
5.  **Dense Layer**: Fully Connected layer (512 units) with ReLU activation.

### Output Heads (The "Brain")
After the feature extractor, the network splits into two separate heads:

1.  **Actor (Policy) Head**:
    *   **Structure**: Linear Layer -> Softmax
    *   **Output**: 2 values (Probability of `Release` vs `Hold`).
    *   **Purpose**: Decides *what to do*.
2.  **Critic (Value) Head**:
    *   **Structure**: Linear Layer
    *   **Output**: 1 scalar value.
    *   **Purpose**: Estimates *how good* the current state is (expected future reward). This helps the Actor learn.

---

## 3. Technology Stack

### Frameworks
*   **[Stable Baselines 3 (SB3)](https://stable-baselines3.readthedocs.io/)**: 
    *   The industry-standard PyTorch library for reliable reinforcement learning implementations. We use `PPO` and `VecFrameStack` from here.
*   **[Gymnasium](https://gymnasium.farama.org/)**:
    *   The standard API for defining RL environments (`step()`, `reset()`, `observation_space`).
*   **[PyTorch](https://pytorch.org/)**:
    *   The underlying Deep Learning framework handling tensors and gradients.

### Preprocessing Tools
*   **OpenCV (cv2)**:
    *   Used for resizing images to 128x128.
    *   Used for converting RGB to Grayscale.
*   **Scrcpy + FFmpeg**:
    *   Used for low-latency (<50ms) video capture from the Android device.

---

## 4. Hyperparameters (Current Configuration)

*   **Learning Rate**: `2.5e-4` (Standard Adam optimizer default).
*   **n_steps**: `2048` (Increased from 128 to allow for longer-horizon planning).
*   **Batch Size**: `64`.
*   **n_epochs**: `10` (Number of times to optimize surrogate loss per update).
*   **Gamma (Discount Factor)**: `0.99` (Values long-term rewards highly).
*   **Entropy Coefficient**: `0.01` (Encourages exploration/randomness early on).
