import os
import sys
import time
import cv2
import csv
from datetime import datetime
from pynput import mouse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from env.scrcpy_client import ScrcpyClient

class DataCollector:
    def __init__(self, fps_limit=30):
        self.fps_limit = fps_limit
        self.is_recording = False
        self.is_holding = False
        self.was_holding = False # Track previous state for edge detection
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("data", "raw", f"session_{timestamp}")
        self.frames_dir = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.session_dir, "actions.csv")
        
        # Setup Scrcpy
        self.client = ScrcpyClient(max_width=800)
        
        # Action Queue for synchronization
        self.frame_count = 0
        
        # Mouse Listener
        self.mouse_listener = mouse.Listener(
            on_click=self.on_click
        )

        # Coordinate config (Bottom Right)
        self.TOUCH_X = 750
        self.TOUCH_Y = 400

    def on_click(self, x, y, button, pressed):
        """Callback for mouse clicks."""
        if button == mouse.Button.left:
            self.is_holding = pressed
            # Async command handling is in main loop

    def start(self):
        print(f"Starting Data Collector...")
        print(f"Saving to: {self.session_dir}")
        print("Controls: LEFT CLICK to Swing (Hold). Close window or Ctrl+C to stop.")
        
        self.client.start()
        # Wait for video
        time.sleep(2)
        
        self.mouse_listener.start()
        
        self.is_recording = True
        
        # Open CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "action", "timestamp", "reward"])
            
            last_time = time.time()
            frame_interval = 1.0 / self.fps_limit
            
            try:
                while self.is_recording:
                    current_time = time.time()
                    
                    if current_time - last_time >= frame_interval:
                        # 1. Capture Frame
                        frame = self.client.get_frame()
                        
                        if frame is not None:
                            # 2. Get Action State
                            action = 1 if self.is_holding else 0
                            
                            # 3. Send Action to Device (Feedback)
                            if action == 1:
                                    # Start Asynchronous Swipe if not already holding
                                    if not self.was_holding:
                                        # Use client method (handles scaling)
                                        self.client.start_async_hold(self.TOUCH_X, self.TOUCH_Y)
                            
                            elif self.was_holding and action == 0:
                                # Release Logic: Kill the swipe
                                self.client.stop_async_hold(self.TOUCH_X, self.TOUCH_Y)
                            
                            self.was_holding = (action == 1) 

                            # 4. Save Data
                            frame_filename = f"frame_{self.frame_count:06d}.jpg"
                            frame_path = os.path.join(self.frames_dir, frame_filename)
                            
                            # Save frame (Raw)
                            cv2.imwrite(frame_path, frame)
                            
                            # Log action
                            # Reward is 0 for now (calculated offline or in Phase 3)
                            writer.writerow([self.frame_count, action, current_time, 0])
                            
                            # 5. Visualization
                            # Show "Recording" indicator
                            display_frame = frame.copy()
                            color = (0, 0, 255) if action == 1 else (0, 255, 0)
                            text = "HOLD" if action == 1 else "RELEASE"
                            cv2.putText(display_frame, f"REC: {self.frame_count} | {text}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.circle(display_frame, (self.TOUCH_X, self.TOUCH_Y), 10, color, -1)
                            
                            cv2.imshow("Data Collector preview", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            
                            self.frame_count += 1
                            last_time = current_time
                        else:
                            print("Warning: No frame received")
                            time.sleep(0.1)
                            
            except KeyboardInterrupt:
                print("\nStopping recording...")
            finally:
                self.stop()

    def stop(self):
        self.is_recording = False
        self.mouse_listener.stop()
        self.client.stop()
        cv2.destroyAllWindows()
        print(f"Session saved. Total frames: {self.frame_count}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.start()
