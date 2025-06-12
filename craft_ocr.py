import cv2
import numpy as np
import os
from craft_text_detector import Craft
import craft_text_detector.craft_utils as craft_utils
import pytesseract

# Configure paths
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def safe_adjust(polys, ratio_w, ratio_h, ratio_net=2):
    adjusted = []
    for poly in polys:
        if poly is not None:
            adjusted.append(np.array(poly) * (ratio_w * ratio_net, ratio_h * ratio_net))
    return adjusted

craft_utils.adjustResultCoordinates = safe_adjust

def ensure_horizontal_rotation(cropped):
    """Ensure text is horizontal by checking aspect ratio"""
    h, w = cropped.shape[:2]
    if h > w:  # If vertical orientation
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped

def rotate_and_crop_horizontal(image, poly):
    """Rotate and crop text region ensuring horizontal output"""
    poly_pts = poly.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(poly_pts)
    
    # Get the angle and size
    angle = rect[-1]
    width, height = rect[1]
    
    # Adjust angle to make text horizontal
    if angle < -45:
        angle = -(90 + angle)
        width, height = height, width  # Swap dimensions
    
    # Get rotation matrix
    center = rect[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the entire image
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Crop the rotated rectangle
    cropped = cv2.getRectSubPix(rotated, (int(width), int(height)), center)
    
    # Ensure horizontal orientation
    cropped = ensure_horizontal_rotation(cropped)
    
    return cropped

def process_image_to_horizontal_segments(image_path, output_dir):
    """Process image and save all text segments in horizontal orientation"""
    os.makedirs(output_dir, exist_ok=True)
    craft = Craft(output_dir=None, crop_type="poly", cuda=False)
    
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Detect text regions
    prediction = craft.detect_text(orig)
    
    for i, poly in enumerate(prediction["boxes"]):
        try:
            # Get horizontal cropped segment
            cropped = rotate_and_crop_horizontal(orig, poly)
            
            # Apply additional preprocessing
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Save the segment
            segment_path = os.path.join(output_dir, f"segment_{i:03d}.png")
            cv2.imwrite(segment_path, processed)
            print(f"Saved horizontal segment {i}")
            
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
    
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

if __name__ == "__main__":
    # Configuration
    input_image = "IMG_0491.JPG"
    output_dir = "horizontal_segments"
    
    # Process image
    process_image_to_horizontal_segments(input_image, output_dir)