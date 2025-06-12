import os
import json
import time
import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict

def calculate_iou(gt_poly, pred_poly):
    """Hitung IoU antara dua polygon."""
    gt = Polygon(gt_poly)
    pred = Polygon(pred_poly)
    intersection = gt.intersection(pred).area
    union = gt.union(pred).area
    return intersection / union if union > 0 else 0.0

def evaluate_detection(gt_json_path, pred_polygons, iou_threshold=0.5):
    """
    Evaluasi performa deteksi (CRAFT) berdasarkan IoU.
    
    Args:
        gt_json_path: Path ke file JSON ground truth (format LabelMe).
        pred_polygons: List polygon hasil prediksi CRAFT.
        iou_threshold: Threshold IoU untuk menentukan TP/FP.
    
    Returns:
        Dict berisi metrik evaluasi.
    """
    # Load ground truth
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    # Format ground truth
    gt_polygons = [shape['points'] for shape in gt_data['shapes']]
    
    # Inisialisasi metrics
    metrics = {
        'total_gt': len(gt_polygons),
        'total_pred': len(pred_polygons),
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'ious': []
    }
    
    # Hitung IoU dan cocokkan prediksi dengan GT
    matched_gt_indices = set()
    for pred_poly in pred_polygons:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_poly in enumerate(gt_polygons):
            if gt_idx in matched_gt_indices:
                continue
            
            iou = calculate_iou(gt_poly, pred_poly)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Update TP/FP/FN
        if best_gt_idx != -1:
            metrics['true_positives'] += 1
            matched_gt_indices.add(best_gt_idx)
            metrics['ious'].append(best_iou)
        else:
            metrics['false_positives'] += 1
    
    metrics['false_negatives'] = metrics['total_gt'] - metrics['true_positives']
    
    # Hitung metrik akhir
    metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
    metrics['recall'] = metrics['true_positives'] / metrics['total_gt'] if metrics['total_gt'] > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    metrics['mean_iou'] = np.mean(metrics['ious']) if metrics['ious'] else 0
    
    return metrics

def run_detection_evaluation(gt_dir='ground_truth', pred_base_dir='output_results'):
    """
    Jalankan evaluasi deteksi untuk semua kategori.
    
    Args:
        gt_dir: Direktori berisi file JSON ground truth.
        pred_base_dir: Direktori output hasil prediksi.
    
    Returns:
        Dict hasil evaluasi untuk semua kategori.
    """
    evaluation_results = {}
    
    for category in os.listdir(pred_base_dir):
        category_path = os.path.join(pred_base_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        # Cari file JSON ground truth
        gt_json_path = os.path.join(gt_dir, f"{category}.json")
        if not os.path.exists(gt_json_path):
            print(f"⚠️ Ground truth tidak ditemukan untuk {category}")
            continue
        
        # Load polygon prediksi (asumsi disimpan di file JSON)
        pred_polygons = []
        for segment_file in os.listdir(category_path):
            if segment_file.endswith('.json'):
                with open(os.path.join(category_path, segment_file), 'r') as f:
                    pred_data = json.load(f)
                pred_polygons.append(pred_data['polygon'])
        
        # Evaluasi
        start_time = time.time()
        metrics = evaluate_detection(gt_json_path, pred_polygons)
        metrics['fps'] = len(pred_polygons) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        
        evaluation_results[category] = metrics
    
    return evaluation_results