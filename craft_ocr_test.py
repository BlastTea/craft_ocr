import cv2
import numpy as np
import os
import json
import pytesseract
import matplotlib.pyplot as plt
from craft_text_detector import Craft
import craft_text_detector.craft_utils as craft_utils
from calculate import run_detection_evaluation

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

def process_and_save_segments(image_path, output_dir):
    """Process image and save all text segments with extracted text"""
    os.makedirs(output_dir, exist_ok=True)
    craft = Craft(output_dir=None, crop_type="poly", cuda=False)
    
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"Error loading image: {image_path}")
        return []
    
    # Detect text regions
    prediction = craft.detect_text(orig)
    results = []
    
    for i, poly in enumerate(prediction["boxes"]):
        try:
            # Get horizontal cropped segment
            cropped = rotate_and_crop_horizontal(orig, poly)
            
            # Apply additional preprocessing
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
            
            # Save the segment
            segment_path = os.path.join(output_dir, f"segment_{i:03d}.png")
            cv2.imwrite(segment_path, processed)
            
            # Save text to file
            txt_path = os.path.join(output_dir, f"segment_{i:03d}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text.strip())
            
            results.append({
                'segment_id': i,
                'image_path': segment_path,
                'text_path': txt_path,
                'text': text.strip()
            })
            
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            results.append({
                'segment_id': i,
                'error': str(e)
            })
    
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    
    return results

def run_ocr_on_all_categories(base_dir='data', output_base='output_results'):
    """Process all images in category folders"""
    os.makedirs(output_base, exist_ok=True)
    results_summary = {}

    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n📁 Processing category: {category}")
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"⚠️  No image found in {category}")
            continue

        image_path = os.path.join(category_path, images[0])
        output_dir = os.path.join(output_base, category)
        result = process_and_save_segments(image_path, output_dir)

        results_summary[category] = {
            "total_segments": len([r for r in result if 'text' in r and r['text']]),
            "raw": result
        }

    # Save raw JSON results
    json_path = os.path.join(output_base, 'craft_ocr_testing_result.json')
    with open(json_path, 'w') as f:
        def convert_numpy(obj):
            if isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)  # fallback for others

        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=convert_numpy)

    return results_summary, json_path

def plot_summary(results_summary, output_base='output_results'):
    """Create visualization of results"""
    categories = list(results_summary.keys())
    values = [results_summary[cat]['total_segments'] for cat in categories]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Image Categories')
    plt.ylabel('Detected Text Segments')
    plt.title('CRAFT OCR - Text Detection by Image Category')
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    chart_path = os.path.join(output_base, 'craft_ocr_testing_result.png')
    plt.savefig(chart_path)
    plt.close()

    return chart_path

if __name__ == "__main__":
    # Process single image (optional)
    # process_and_save_segments("IMG_0491.JPG", "horizontal_segments")
    
    # Process all categories
    summary, json_file = run_ocr_on_all_categories()
    chart_file = plot_summary(summary)
    evaluation_results = run_detection_evaluation()


    print("\n✅ Processing complete!")
    print(f"📊 Chart saved at: {chart_file}")
    print(f"📄 JSON results saved at: {json_file}")
    
    eval_json_path = os.path.join('output_results', 'evaluation_results.json')
    with open(eval_json_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("\n📊 Hasil Evaluasi:")
    for category, metrics in evaluation_results.items():
        print(f"\n🔹 {category}:")
        print(f"  - Precision: {metrics['precision']:.2f}")
        print(f"  - Recall: {metrics['recall']:.2f}")
        print(f"  - F1-Score: {metrics['f1_score']:.2f}")
        print(f"  - IoU: {metrics['mean_iou']:.2f}")
        print(f"  - FPS: {metrics['fps']:.2f}")

