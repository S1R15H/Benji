import sys
import os
import cv2
import time
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from env.benji_env import BenjiBananasEnv

def capture_snapshot(filename="game_start.jpg"):
    print("Initializing Environment...")
    # Use the full environment to ensure correct startup sequence (adb, scrcpy, control)
    try:
        # Initialize env (will start scrcpy, adb, etc.)
        env = BenjiBananasEnv(offline=False)
        
        print("Waiting for frames...")
        # Give it a moment to settle
        time.sleep(2)
        
        # Grab a frame
        frame = env.client.get_frame()
        
        # If None, try resetting to force a fresh frame pull
        if frame is None:
             print("Frame is None, trying reset step...")
             env.reset()
             time.sleep(1)
             frame = env.client.get_frame()
        
        if frame is not None:
            cv2.imwrite(filename, frame)
            abs_path = os.path.abspath(filename)
            print(f"Snapshot saved to: {abs_path}")
        else:
            print("Error: Could not grab frame from environment.")
            
        env.close()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "game_start.jpg"
    capture_snapshot(fn)
