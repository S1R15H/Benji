import sys
import os
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from env.benji_env import BenjiBananasEnv

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Environment...")
    env = BenjiBananasEnv(offline=False)
    
    print("Starting Reset Loop (5 times)...")
    try:
        for i in range(5):
            print(f"\n--- Reset #{i+1} ---")
            print("Calling env.reset()...")
            env.reset()
            print("Reset complete. The game SHOULD be playing now.")
            print("Sleeping 5 seconds to observe...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
