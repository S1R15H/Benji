import os
import cv2
import csv
import sys
import glob

def verify_session(session_name):
    base_path = os.path.join("data", "raw", session_name)
    csv_path = os.path.join(base_path, "actions.csv")
    frames_dir = os.path.join(base_path, "frames")
    
    print(f"Verifying Session: {session_name}")
    
    # 1. Check CSV
    if not os.path.exists(csv_path):
        print("FAIL: actions.csv not found!")
        return
    
    actions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append(row)
            
    print(f"CSV Rows: {len(actions)}")
    
    # 2. Check Frames
    if not os.path.exists(frames_dir):
        print("FAIL: frames/ directory not found!")
        return
        
    frame_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
    print(f"Frame Files: {len(frame_files)}")
    
    if len(actions) != len(frame_files):
        print("WARNING: Mismatch between CSV rows and Frame files!")
    else:
        print("PASS: Frame count matches CSV count.")
        
    # 3. Generate Video
    print("Generating Verification Video (first 300 frames)...")
    output_video = "verification.avi"
    
    # Sort actions by frame_id
    actions.sort(key=lambda x: int(x['frame_id']))
    
    # Read first frame to get size
    first_frame_path = os.path.join(frames_dir, f"frame_{int(actions[0]['frame_id']):06d}.jpg")
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print("FAIL: Could not read first frame")
        return
        
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))
    
    limit = 300
    for i, row in enumerate(actions[:limit]):
        frame_id = int(row['frame_id'])
        action = int(row['action'])
        
        img_path = os.path.join(frames_dir, f"frame_{frame_id:06d}.jpg")
        img = cv2.imread(img_path)
        
        if img is not None:
            # Overlay Info
            color = (0, 0, 255) if action == 1 else (0, 255, 0)
            text = "HOLD" if action == 1 else "RELEASE"
            
            cv2.putText(img, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Action: {text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw touch point
            cv2.circle(img, (750, 400), 10, color, -1)
            
            video_writer.write(img)
        else:
            print(f"Missing frame: {frame_id}")
            
    video_writer.release()
    print(f"Video saved to {output_video}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        session = sys.argv[1]
    else:
        # Find latest
        raw_dir = os.path.join("data", "raw")
        sessions = sorted([d for d in os.listdir(raw_dir) if d.startswith("session_")])
        if not sessions:
            print("No sessions found.")
            sys.exit(1)
        session = sessions[-1]
        
    verify_session(session)
