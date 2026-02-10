
import cv2
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from env.benji_env import BenjiBananasEnv

def main():
    print("Initializing Debug Viewer...")
    # Use offline=False to connect to real device (or emulator)
    # Ensure Scrcpy is working
    try:
        env = BenjiBananasEnv(render_mode="rgb_array", offline=False)
    except Exception as e:
        print(f"Failed to init environment: {e}")
        return

    print("Environment Initialized. Press 'q' to quit.")
    
    obs, _ = env.reset()
    
    while True:
        # Step with "Release" action (0) just to observe
        # Or "Hold" (1) if you want to test swinging. 
        # For Observation, let's just do No-Op equivalent (Release usually safer)
        action = 0 
        
        start_t = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        proc_t = time.time() - start_t
        
        # 1. Get Components
        raw_frame = env.current_frame
        if raw_frame is None: 
            print("No frame?")
            continue
            
        components = info.get("reward_components", {})
        dist_val = components.get("raw_dist", 0)
        banana_val = components.get("raw_bananas", 0)
        
        # 2. Visualization Setup
        # 2. Visualization Setup
        # Canvas: 1100x600 (Compact)
        canvas = np.zeros((600, 1100, 3), dtype=np.uint8)
        
        # A. Raw Frame (Left) -> Resize to fit
        # Raw is 800x448 usually
        h, w = raw_frame.shape[:2]
        
        # Target main video width: 600
        target_video_w = 600
        scale = target_video_w / w
        new_w = target_video_w
        new_h = int(h * scale)
        
        # Check if new_h exceeds canvas height
        if new_h > 600:
             scale = 600 / h
             new_h = 600
             new_w = int(w * scale)

        # resize raw
        if new_w > 0 and new_h > 0:
            resized_raw = cv2.resize(raw_frame, (new_w, new_h))
            canvas[:new_h, :new_w] = resized_raw
        
        # Draw ROI on Raw
        rc = env.reward_calculator
        if hasattr(rc, 'dist_roi'):
            # Draw Distance ROI (Green)
            rx, ry, rw, rh = rc.dist_roi
            cv2.rectangle(canvas, (int(rx*scale), int(ry*scale)), (int((rx+rw)*scale), int((ry+rh)*scale)), (0, 255, 0), 2)
            # Draw Banana ROI (Yellow)
            bx, by, bw, bh = rc.banana_roi
            cv2.rectangle(canvas, (int(bx*scale), int(by*scale)), (int((bx+bw)*scale), int((by+bh)*scale)), (0, 255, 255), 2)
        
        # B. Preprocessed (Right Top)
        if len(obs.shape) == 2:
            ai_view = obs
        elif obs.shape[-1] == 1:
            ai_view = obs[:, :, 0]
        else:
            ai_view = obs[-1]
            
        # Resize to 200x200 (Compact)
        ai_view_norm = cv2.normalize(ai_view, None, 0, 255, cv2.NORM_MINMAX)
        # Handle potential zero-size if preprocessor failed (unlikely)
        ai_view_vis = cv2.resize(ai_view_norm, (200, 200), interpolation=cv2.INTER_NEAREST)
        ai_view_vis = cv2.cvtColor(ai_view_vis, cv2.COLOR_GRAY2BGR)
        
        x_offset = new_w + 20 # 620
        y_offset = 20
        canvas[y_offset:y_offset+200, x_offset:x_offset+200] = ai_view_vis
        cv2.putText(canvas, "AI View (128x128)", (x_offset, y_offset-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # D. HUD / Metadata (Right Side, below AI View)
        x_hud = x_offset + 200 + 20 # 840
        y_hud = 40
        line_h = 30
        
        color = (0, 255, 255)
        cv2.putText(canvas, f"Dist: {dist_val}m", (x_hud, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(canvas, f"Bananas: {banana_val}", (x_hud, y_hud + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Sticky Rewards Logic
        # Rewards are instantaneous (1 frame), so we hold them for 30 frames (1s) for visibility
        if not hasattr(env, '_last_rewards'): env._last_rewards = {}
        if not hasattr(env, '_reward_timers'): env._reward_timers = {}
        
        for key in ['dist_reward', 'banana_reward', 'momentum_reward', 'survival_reward']:
            val = components.get(key, 0.0)
            if val != 0:
                env._last_rewards[key] = val
                env._reward_timers[key] = 30 # Hold for 30 frames
            elif env._reward_timers.get(key, 0) > 0:
                env._reward_timers[key] -= 1
            else:
                env._last_rewards[key] = 0.0
                
        disp_dist = env._last_rewards.get('dist_reward', 0.0)
        disp_banana = env._last_rewards.get('banana_reward', 0.0)
        disp_mom = env._last_rewards.get('momentum_reward', 0.0)
        disp_surv = env._last_rewards.get('survival_reward', 0.0)

        cv2.putText(canvas, "Rewards (Hold 1s):", (x_hud, y_hud + 3*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f" D:{disp_dist:.3f} B:{disp_banana:.3f}", (x_hud, y_hud + 4*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f" M:{disp_mom:.3f} S:{disp_surv:.3f}", (x_hud, y_hud + 5*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(canvas, f"Time: {proc_t*1000:.1f}ms", (x_hud, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # E. OCR DEBUG VIEW (Under AI View)
        # Show raw crops used for Template Matching
        if hasattr(rc, 'dist_roi'):
            # Show Dist Crop
            dx, dy, dw, dh = rc.dist_roi
            if dy+dh <= raw_frame.shape[0] and dx+dw <= raw_frame.shape[1]:
                roi_d = raw_frame[dy:dy+dh, dx:dx+dw]
                # Scale up for visibility
                vis_d = cv2.resize(roi_d, (dw*3, dh*3), interpolation=cv2.INTER_NEAREST)
                
                y_ocr = y_offset + 200 + 40
                canvas[y_ocr:y_ocr+dh*3, x_offset:x_offset+dw*3] = vis_d
                cv2.putText(canvas, "Dist ROI", (x_offset, y_ocr-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show Banana Crop
            bx, by, bw, bh = rc.banana_roi
            if by+bh <= raw_frame.shape[0] and bx+bw <= raw_frame.shape[1]:
                roi_b = raw_frame[by:by+bh, bx:bx+bw]
                # Scale up
                vis_b = cv2.resize(roi_b, (bw*3, bh*3), interpolation=cv2.INTER_NEAREST)
                
                y_ocr_b = y_ocr + dh*3 + 20
                canvas[y_ocr_b:y_ocr_b+bh*3, x_offset:x_offset+bw*3] = vis_b
                cv2.putText(canvas, "Banana ROI", (x_offset, y_ocr_b-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Show
        cv2.imshow("Benji Brain Debugger", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Snapshot
            timestamp = int(time.time())
            fname = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(fname, raw_frame)
            print(f"Saved snapshot to {fname}")
            
        if terminated:
            print("Game Over. Resetting...")
            env.reset()
            
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
