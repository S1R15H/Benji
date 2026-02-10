import sys
import os
import argparse
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from env.benji_env import BenjiBananasEnv
from agent.model import BenjiAgent

def main():
    parser = argparse.ArgumentParser(description="Run Benji Bananas Agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--render", action="store_true", help="Render RGB array (slower)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        print(f"Error: Model not found at {args.model}")
        return

    print(f"Loading Agent from {args.model}...")
    try:
        # Use BenjiAgent to handle environment wrapping (Stacking, Transpose)
        agent = BenjiAgent(model_path=args.model, offline=False)
        env = agent.venv # Use the wrapped vector environment
        
        print("Starting Play Loop...")
        
        for ep in range(args.episodes):
            print(f"Episode {ep+1}/{args.episodes}")
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # VecEnv automatically resets when done, so we need to rely on the 'dones' flag
            # But for a manual loop, we iterate until done is True
            # Since VecEnv auto-resets, we can't easily break the loop exactly on termination without checking info
            
            while not done:
                # Predict action
                action, _states = agent.model.predict(obs, deterministic=True)
                
                # VecEnv step returns: obs, rewards, dones, infos
                obs, rewards, dones, infos = env.step(action)
                
                # Extract scalar values from vector
                reward = rewards[0]
                done = dones[0]
                info = infos[0]
                
                total_reward += reward
                steps += 1
                
                # Optional: print rewards if non-zero
                if reward != 0:
                     # Access reward components from info if available
                     # Because of auto-reset, info might contain terminal info
                     dist = 0
                     if 'reward_components' in info:
                         dist = info['reward_components'].get('raw_dist', 0)
                     elif 'terminal_observation' in info and 'reward_components' in info.get('terminal_info', {}):
                          # Check terminal info
                          pass
                          
                     print(f"Step {steps} | Reward: {reward:.4f}")
            
            print(f"Episode Finished. Total Reward: {total_reward:.4f} | Steps: {steps}")
            time.sleep(1) # Pause between games
            
    except KeyboardInterrupt:
        print("\nStopping play...")
    finally:
        if 'agent' in locals():
            agent.close()

if __name__ == "__main__":
    main()
