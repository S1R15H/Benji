import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from gymnasium.utils.env_checker import check_env
from env.benji_env import BenjiBananasEnv

def main():
    print("Initializing Environment...")
    # Check if we are in CI/Test mode
    is_testing = os.getenv("TESTING_MODE", "false").lower() == "true"
    env = BenjiBananasEnv(offline=is_testing)
    
    print("Running Gymnasium API Check...")
    # This checks for observation space conformity, action space types, reset returns, etc.
    try:
        check_env(env, skip_render_check=True)
        print("PASS: Environment complies with Gymnasium API.")
    except Exception as e:
        print(f"FAIL: Environment check failed: {e}")
        env.close()
        sys.exit(1)
        
    print("\nTesting Reset...")
    obs, info = env.reset()
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Type: {obs.dtype}")
    
    if obs.shape != (1, 128, 128):
        print(f"FAIL: Expected (1, 128, 128), got {obs.shape}")
        
    print("\nTesting Step Loop (10 steps)...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Terminated={terminated}")
        time.sleep(0.1)
        
    print("\nClosing Environment...")
    env.close()
    print("SUCCESS: Environment verification complete.")

if __name__ == "__main__":
    main()
