import sys
import os
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from env.benji_env import BenjiBananasEnv

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_restart():
    print("Initializing Environment...")
    env = BenjiBananasEnv(render_mode="rgb_array", offline=False)
    
    try:
        print("Please manually trigger a Game Over screen on the device.")
        print("Expected behavior: The script should detect the screen and restart the game.")
        
        for i in range(60): # Run for 60 seconds
            print(f"Checking for Game Over... ({60-i})")
            env._check_and_restart()
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()

if __name__ == "__main__":
    test_restart()
