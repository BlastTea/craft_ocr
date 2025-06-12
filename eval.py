import os
import cv2
import json
import time
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict
import matplotlib.pyplot as plt
from craft_text_detector import Craft
import pytesseract
from calculate import calculate_iou
from craft_ocr_test import rotate_and_crop_horizontal

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

def load_icdar_gt(gt_path):
    """
    Load ground truth dari format ICDAR:
    x1,y1,x2,y2,x3,y3,x4,y4,transkrip
    """
    gt_data = []
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(',')
            coords = list(map(float, parts[:8]))
            text = parts[8] if len(parts) > 8 else ''
            
            # Format polygon ICDAR ke [[x1,y1], [x2,y2], ...]
            polygon = [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]]
            ]
            gt_data.append({'polygon': polygon, 'text': text})
    return gt_data

def prepare_icdar_data(image_dir, gt_dir):
    """
    Siapkan pasangan gambar-GT ICDAR.
    Returns: List of {'image_path': ..., 'gt_path': ...}
    """
    data_pairs = []
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.jpg', '.png')):
            base_name = os.path.splitext(img_file)[0]
            gt_file = f"gt_{base_name}.txt"
            gt_path = os.path.join(gt_dir, gt_file)
            
            if os.path.exists(gt_path):
                data_pairs.append({
                    'image_path': os.path.join(image_dir, img_file),
                    'gt_path': gt_path
                })
    return data_pairs


def evaluate_icdar_sample(image_path, gt_path, output_dir=None):
    """
    Proses evaluasi untuk satu sampel ICDAR.
    """
    # Load data
    orig = cv2.imread(image_path)
    gt_boxes = load_icdar_gt(gt_path)
    
    # Deteksi dengan CRAFT
    craft = Craft(output_dir=None, crop_type="poly", cuda=False)
    prediction = craft.detect_text(orig)
    craft.unload_craftnet_model()
    
    # Inisialisasi metrics
    metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'ious': [],
        'text_accuracy': 0
    }
    
    # Match prediksi dengan GT
    matched_gt_indices = set()
    for pred_box in prediction['boxes']:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt_indices:
                continue
                
            iou = calculate_iou(gt_box['polygon'], pred_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Hitung TP/FP
        if best_iou >= 0.5:  # Threshold IoU
            metrics['true_positives'] += 1
            matched_gt_indices.add(best_gt_idx)
            metrics['ious'].append(best_iou)
            
            # OCR evaluation jika ada teks di GT
            if gt_boxes[best_gt_idx]['text'] != '###':
                cropped = rotate_and_crop_horizontal(orig, pred_box)
                text = pytesseract.image_to_string(cropped, config='--oem 3 --psm 7')
                if text.strip().lower() == gt_boxes[best_gt_idx]['text'].lower():
                    metrics['text_accuracy'] += 1
        else:
            metrics['false_positives'] += 1
    
    metrics['false_negatives'] = len(gt_boxes) - metrics['true_positives']
    
    return metrics

def evaluate_icdar2015(image_dir, gt_dir, output_base='icdar_results'):
    """
    Jalankan evaluasi pada seluruh dataset ICDAR 2015.
    """
    os.makedirs(output_base, exist_ok=True)
    data_pairs = prepare_icdar_data(image_dir, gt_dir)
    results = []
    
    for pair in data_pairs:
        start_time = time.time()
        metrics = evaluate_icdar_sample(
            pair['image_path'],
            pair['gt_path'],
            output_dir=output_base
        )
        
        # Hitung metrik tambahan
        metrics['image'] = os.path.basename(pair['image_path'])
        metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
        metrics['recall'] = metrics['true_positives'] / len(load_icdar_gt(pair['gt_path'])) if len(load_icdar_gt(pair['gt_path'])) > 0 else 0
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['fps'] = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        
        results.append(metrics)
    
    # Simpan hasil
    result_path = os.path.join(output_base, 'eval_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_icdar_results(results):
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(recalls, precisions, c=f1_scores, cmap='viridis')
    plt.colorbar(label='F1-Score')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (ICDAR 2015)')
    plt.grid()
    plt.show()
    
    
if __name__ == "__main__":
    # Evaluasi data test
    test_results = evaluate_icdar2015(
        image_dir='icdar2015/ch4_test_images',
        gt_dir='icdar2015/ch4_test_localization_transcription_gt'
    )
    
    # Evaluasi data training (opsional)
    train_results = evaluate_icdar2015(
        image_dir='icdar2015/ch4_training_images',
        gt_dir='icdar2015/ch4_training_localization_transcription_gt'
    )
    
    # Visualisasi
    plot_icdar_results(test_results)