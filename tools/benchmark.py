import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from env.benji_env import BenjiBananasEnv

def main():
    print("Initializing Environment...")
    env = BenjiBananasEnv(render_mode=None, offline=False)
    
    print("\n" + "="*40)
    print("      LATENCY & THROUGHPUT TEST      ")
    print("="*40)
    print(f"Target FPS: 15.0")
    print(f"Target Step Duration: 66.6 ms")
    print("-" * 40)
    
    # Warmup
    print("Warming up (10 steps)...")
    env.reset()
    for _ in range(10):
        env.step(0)
        
    print("Starting Benchmark loop...")
    
    steps = 100
    durations = []
    
    start_time = time.time()
    
    for i in range(steps):
        step_start = time.time()
        
        # Alternate Holding to trigger input logic
        action = 1 if i % 20 < 10 else 0
        
        obs, reward, done, _, info = env.step(action)
        
        if done:
            env.reset()
            
        dur = (time.time() - step_start) * 1000 # ms
        durations.append(dur)
        
    total_time = time.time() - start_time
    env.close()
    
    # Analysis
    avg_dur = np.mean(durations)
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    fps = steps / total_time
    
    print("\n" + "="*40)
    print("           RESULTS           ")
    print("="*40)
    print(f"Total Steps:    {steps}")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"Actual FPS:     {fps:.2f} (Target: 15.0)")
    print("-" * 40)
    print(f"Avg Step Time:  {avg_dur:.2f} ms (Target: ~66ms)")
    print(f"Min Step Time:  {min_dur:.2f} ms")
    print(f"Max Step Time:  {max_dur:.2f} ms")
    print("="*40)

    if fps < 10.0:
        print("\n[FAIL] Pipeline is too slow! (<10 FPS)")
        print("- Possible Cause: OCR CPU usage, Socket Latency, or Decoding lag.")
    elif fps > 20.0:
        print("\n[WARN] Pipeline is too fast! (>20 FPS)")
        print("- Is frame synchronization working?")
    else:
        print("\n[PASS] Pipeline Latency is acceptable based on FPS.")

if __name__ == "__main__":
    main()
