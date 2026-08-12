import cv2
import numpy as np
import argparse
import os

def process_canny(img_bgr):
    """Standard Canny Edge Detection"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Wide translation of thresholds
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def process_color_filter(img_bgr):
    """
    Filter for Brown (Vines) and Orange/Yellow (Benji).
    Top 1/3 is Sky (Blue/White) -> Mask Out
    Bottom is Green (Trees/Grass) -> Mask Out
    """
    # User Defaults:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Benji: RGB(253,217,164) -> HSV(~18, 90, 253) [Low Saturation]
    # Vine: RGB(255,181,21) -> HSV(~20, 235, 255) [High Saturation, Orange]
    # Banana: RGB(253,240,8) -> HSV(~28, 247, 253) [High Saturation, Yellow]

    # 1. Benji (Pale Orange/Skin tone)
    # Key differentiator: Low Saturation (< 150)
    lower_benji = np.array([10, 50, 200])
    upper_benji = np.array([25, 160, 255])
    mask_benji = cv2.inRange(hsv, lower_benji, upper_benji)

    # 2. Vine (Vivid Orange)
    # Key differentiator: High Saturation (> 180), Orange Hue (15-24)
    lower_vine = np.array([15, 180, 180])
    upper_vine = np.array([24, 255, 255])
    mask_vine = cv2.inRange(hsv, lower_vine, upper_vine)

    # 3. Banana (Bright Yellow)
    # Key differentiator: Yellow Hue (25-35)
    lower_banana = np.array([25, 150, 180])
    upper_banana = np.array([35, 255, 255])
    mask_banana = cv2.inRange(hsv, lower_banana, upper_banana)
    
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    
    # 1. Vines: Thicken (Critical at 128x128)
    mask_vine = cv2.dilate(mask_vine, kernel, iterations=1)
    
    # 2. Benji: Clean up noise (Open is okay here as Benji is a large blob)
    mask_benji = cv2.morphologyEx(mask_benji, cv2.MORPH_OPEN, kernel)
    
    # 3. Banana: Do NOT Open. They are small and will disappear.
    # Just simple dilation to make them more visible to CNN?
    # Or raw mask. Let's Dilate slightly to ensure they are at least 3x3 pixels.
    mask_banana = cv2.dilate(mask_banana, kernel, iterations=1)
    
    # Combine (Priority logic)
    # Background = 0
    combined = np.zeros_like(mask_vine)
    
    # Assign distinct linear intensities
    # Benji = 85  (Dark Gray)
    # Banana = 170 (Light Gray)
    # Vine = 255 (White)
    
    combined[mask_benji > 0] = 85
    combined[mask_banana > 0] = 170
    combined[mask_vine > 0] = 255
    
    return combined

def process_laplacian(img_bgr):
    """Laplacian gradient"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--out", type=str, default="preprocessing_test.jpg", help="Output comparison image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"File not found: {args.image}")
        return
        
    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image")
        return
        
    # Resize for consistent processing visualization (Environment uses 800x448 input, then 128x128 output)
    # We should simulate the full pipeline: 
    # 1. Resize to target shape (128x128) -> Process
    # OR 
    # 2. Process Full Res -> Resize
    # Env currently does: Resize BGR -> 128x128 -> Convert Gray (or vice versa in Preprocessor)
    # Let's try processing at 128x128 to match what the CNN sees.
    
    target_shape = (128, 128)
    img_resized = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
    
    # 1. Original (Gray)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. Canny
    canny = process_canny(img_resized)
    
    # 3. Color Filter
    color_mask = process_color_filter(img_resized)
    
    # 4. Laplacian
    lap = process_laplacian(img_resized)
    
    # Concatenate for display
    # Top Row: Original Gray | Canny
    # Bottom Row: Color Mask | Laplacian
    
    top = np.hstack((gray, canny))
    bottom = np.hstack((color_mask, lap))
    final = np.vstack((top, bottom))
    
    # Annotate
    final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 255, 0)
    thickness = 1
    
    cv2.putText(final_bgr, "Original (Gray)", (10, 15), font, scale, color, thickness)
    cv2.putText(final_bgr, "Canny Edges", (10 + 128, 15), font, scale, color, thickness)
    cv2.putText(final_bgr, "Color Filter (HSV)", (10, 15 + 128), font, scale, color, thickness)
    cv2.putText(final_bgr, "Laplacian", (10 + 128, 15 + 128), font, scale, color, thickness)
    
    cv2.imwrite(args.out, final_bgr)
    print(f"Saved comparison to {args.out}")

if __name__ == "__main__":
    main()
