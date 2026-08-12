import cv2
import os

def crop_and_save():
    # Load the debug snapshot
    input_path = "game_start_debug.jpg"
    output_path = "src/env/game_start.jpg"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    h, w = img.shape[:2]
    print(f"Input Resolution: {w}x{h}")
    
    # Define crop region for the stone on the left
    # x: 0 to 120 (15%)
    # y: 250 to 450 (bottom-left)
    crop_x1 = 0
    crop_x2 = 120
    crop_y1 = 250
    crop_y2 = 440
    
    # Ensure bounds
    crop_x2 = min(crop_x2, w)
    crop_y2 = min(crop_y2, h)
    
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    cv2.imwrite(output_path, cropped)
    print(f"Template saved to {output_path}")
    print(f"Template Size: {cropped.shape[1]}x{cropped.shape[0]}")

if __name__ == "__main__":
    crop_and_save()
