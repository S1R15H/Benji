import sys
import os
import time
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from env.benji_env import BenjiBananasEnv

def test_benji_env_compliance():
    print("Initializing Environment...")
    # Check if we are in CI/Test mode
    is_testing = os.getenv("TESTING_MODE", "false").lower() == "true"
    
    # Ideally, we should mock the scrcpy client if offline=True is not enough
    # But BenjiBananasEnv(offline=True) should be safe.
    env = BenjiBananasEnv(offline=is_testing)
    
    print("Running Gymnasium API Check...")
    # this might fail on render modes if not careful, but skip_render_check helps
    check_env(env, skip_render_check=True)
    
    print("\nTesting Reset...")
    obs, info = env.reset()
    assert obs.shape == (1, 128, 128), f"Expected (1, 128, 128), got {obs.shape}"
    assert obs.dtype == "uint8"
    
    print("\nTesting Step Loop (5 steps)...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (1, 128, 128)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        
    env.close()

if __name__ == "__main__":
    # Allow running as script too
    os.environ["TESTING_MODE"] = "true"
    test_benji_env_compliance()
