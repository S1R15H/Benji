import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from env.scrcpy_client import ScrcpyClient

def test_adb():
    print("Initializing ScrcpyClient...")
    client = ScrcpyClient(max_width=800)
    try:
        client.start()
        time.sleep(2) # Buffer
        
        print("\n--- TEST 1: TAP ---")
        print("Tapping (750, 400)... check device.")
        x, y = 750, 400
        client.tap(x, y)
        time.sleep(2)
        
        print("\n--- TEST 2: SYNC SWIPE (1 second) ---")
        print("Swiping down... check device.")
        client.swipe(x, y, x, y, 1000)
        time.sleep(2)
        
        print("\n--- TEST 3: ASYNC SWIPE (3 seconds) ---")
        print("Starting Async Swipe...")
        client.start_async_hold(x, y)
        
        print("Holding for 2 seconds...")
        time.sleep(2)
        
        print("Releasing (pkill)...")
        client.stop_async_hold(x, y)
        
        print("\n--- TEST COMPLETE ---")
        
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.stop()

if __name__ == "__main__":
    test_adb()
