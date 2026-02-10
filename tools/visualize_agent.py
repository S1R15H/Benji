import sys
import os
import cv2
import numpy as np
import time
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent.model import BenjiAgent
from env.benji_env import BenjiBananasEnv

def main():
    parser = argparse.ArgumentParser(description="Visualize Benji Agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model zip")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to play")
    parser.add_argument("--output", type=str, default="agent_play.mp4", help="Output video file")
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found.")
        return

    print("Initializing Agent and Environment...")
    # Note: We need a REAL environment to record useful video
    # If device is not connected, this will fail or hang.
    try:
        agent = BenjiAgent(model_path=args.model, offline=False)
    except Exception as e:
        print(f"Failed to connect to device: {e}")
        return

    env = agent.env # Get the Monitor wrapped env
    # Monitor -> BenjiBananasEnv
    # We need access to the raw frame from BenjiBananasEnv to save video
    # stable_baselines3 Monitor wraps it.
    # But BenjiAgent.env is a Monitor.
    # To get raw env: agent.env.env
    real_env = env.env 
    
    # Reset
    obs, info = env.reset()
    
    # Video Writer
    # Frame size: 800x448 (Raw from Scrcpy)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (800, 448))
    
    print(f"Recording {args.steps} steps to {args.output}...")
    
    total_reward = 0
    current_action = "RELEASE"
    
    for i in range(args.steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Visualize
        frame = real_env.current_frame.copy() if real_env.current_frame is not None else np.zeros((448, 800, 3), dtype=np.uint8)
        
        # Overlay UI
        # Action
        color = (0, 255, 0) if action == 1 else (0, 0, 255)
        text = "HOLD" if action == 1 else "RELEASE"
        cv2.putText(frame, f"Action: {text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Reward
        cv2.putText(frame, f"Reward: {reward:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Step
        cv2.putText(frame, f"Step: {i}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        out.write(frame)
        
        if terminated or truncated:
            print("Episode Finished. Resetting...")
            obs, info = env.reset()
            
    out.release()
    agent.close()
    print("Done.")

if __name__ == "__main__":
    main()
