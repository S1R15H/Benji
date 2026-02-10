import sys
import os
import cv2
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from env.scrcpy_client import ScrcpyClient

def capture_snapshot(filename="game_over_1.jpg"):
    print("Initializing Client...")
    client = ScrcpyClient(max_width=800)
    
    try:
        client.start()
        # Buffer
        time.sleep(3)
        
        print("Capturing frame...")
        # Retry for up to 5 seconds
        start_time = time.time()
        frame = None
        while time.time() - start_time < 5:
            frame = client.get_frame()
            if frame is not None:
                break
            time.sleep(0.1)
        
        if frame is not None:
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved to: {os.path.abspath(filename)}")
        else:
            print("Error: Could not grab frame after 5 seconds.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.stop()

if __name__ == "__main__":
    fn = sys.argv[1] if len(sys.argv) > 1 else "game_over_screen.jpg"
    capture_snapshot(fn)
