import sys
import os
import time
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from env.reward import BenjiReward

def generate_mock_assets(digits_dir):
    """
    If assets are missing, generate dummy 0-9 pngs to allow benchmark to run 
    (strictly for timing, not accuracy unless we mock the input too).
    """
    if not os.path.exists(digits_dir):
        os.makedirs(digits_dir)
        
    for i in range(10):
        path = os.path.join(digits_dir, f"{i}.png")
        if not os.path.exists(path):
            # Create a 10x15 white image with the number text
            img = np.zeros((20, 15), dtype=np.uint8)
            cv2.putText(img, str(i), (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)
            cv2.imwrite(path, img)
    print("Generated mock digit assets for benchmarking.")

def main():
    assets_dir = os.path.join(os.path.dirname(__file__), '../src/env/assets/digits')
    
    # Check if assets exist
    if not os.listdir(assets_dir) if os.path.exists(assets_dir) else True:
         print("Assets not found. Generating mocks...")
         generate_mock_assets(assets_dir)
    
    print("Initializing BenjiReward (Template Matching)...")
    reward_engine = BenjiReward()
    
    # Create a dummy frame (800x448)
    frame = np.zeros((448, 800, 3), dtype=np.uint8)
    
    # Draw some fake numbers in the ROI
    # Dist ROI: (715, 39, 55, 19)
    # Banana ROI: (715, 59, 55, 19)
    
    # 1. Dist "123"
    cv2.putText(frame, "123", (720, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # 2. Bananas "45"
    cv2.putText(frame, "45", (720, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print("Benchmarking calculate() over 1000 iterations...")
    
    times = []
    
    start_global = time.time()
    for _ in range(1000):
        t0 = time.time()
        r, comps = reward_engine.calculate(frame, False)
        dt = time.time() - t0
        times.append(dt)
        
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"Total Time: {time.time() - start_global:.4f}s")
    print(f"Average: {avg_time*1000:.4f} ms per frame")
    print(f"FPS Capacity: {fps:.2f}")
    
    if avg_time < 0.005: # < 5ms
        print("\nSUCCESS: Reward calculation is extremely fast!")
    else:
        print("\nWARNING: Still relatively slow. Check Template Matching size.")

if __name__ == "__main__":
    main()
