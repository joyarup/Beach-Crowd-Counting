# src/evaluation/metrics.py
import numpy as np
import pandas as pd
from collections import defaultdict

def load_ground_truth(csv_path):
    """
    Load and process ground truth data from CSV file.
    Returns a dictionary with image filenames as keys and lists of coordinates as values.
    """
    df = pd.read_csv(csv_path, header=None, 
                     names=['label', 'x', 'y', 'image', 'width', 'height'])
    
    # Group by image filename
    ground_truth = defaultdict(list)
    for _, row in df.iterrows():
        ground_truth[row['image']].append((row['x'], row['y']))
    
    return ground_truth

def create_detection_mask(detections, image_shape):
    """
    Convert detection mask to coordinate list.
    Returns list of (x,y) coordinates where detections were made.
    """
    coords = []
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            if detections[y, x] > 0:
                coords.append((x, y))
    return coords

def calculate_detection_metrics(ground_truth_coords, detection_coords, max_distance=50):
    """
    Calculate detection metrics including MSE and matching statistics.
    Uses nearest neighbor matching with maximum distance threshold.
    """
    if not ground_truth_coords or not detection_coords:
        return {
            'mse': float('inf'),
            'matched_detections': 0,
            'false_positives': len(detection_coords),
            'false_negatives': len(ground_truth_coords),
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Convert to numpy arrays for efficient computation
    gt_array = np.array(ground_truth_coords)
    det_array = np.array(detection_coords)
    
    # Calculate distance matrix between all ground truth and detection points
    distances = np.sqrt(((gt_array[:, np.newaxis] - det_array) ** 2).sum(axis=2))
    
    # Initialize matching arrays
    gt_matched = np.zeros(len(gt_array), dtype=bool)
    det_matched = np.zeros(len(det_array), dtype=bool)
    squared_errors = []
    
    # Match points using nearest neighbor with distance threshold
    while True:
        # Find minimum distance that hasn't been matched
        unmatched_distances = distances[~gt_matched][:, ~det_matched]
        if unmatched_distances.size == 0:
            break
            
        min_dist_idx = np.unravel_index(np.argmin(unmatched_distances), 
                                       unmatched_distances.shape)
        min_dist = unmatched_distances[min_dist_idx]
        
        if min_dist > max_distance:
            break
            
        # Get original indices
        gt_idx = np.where(~gt_matched)[0][min_dist_idx[0]]
        det_idx = np.where(~det_matched)[0][min_dist_idx[1]]
        
        # Mark as matched and add to squared errors
        gt_matched[gt_idx] = True
        det_matched[det_idx] = True
        squared_errors.append(min_dist ** 2)
    
    # Calculate metrics
    matched_detections = len(squared_errors)
    false_positives = np.sum(~det_matched)
    false_negatives = np.sum(~gt_matched)
    
    mse = np.mean(squared_errors) if squared_errors else float('inf')
    precision = matched_detections / len(detection_coords) if detection_coords else 0
    recall = matched_detections / len(ground_truth_coords) if ground_truth_coords else 0
    
    return {
        'mse': mse,
        'matched_detections': matched_detections,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall
    }

def evaluate_detections(results, ground_truth_file, image_filenames):
    """
    Evaluate detection results against ground truth data.
    """
    # Load ground truth data
    ground_truth = load_ground_truth(ground_truth_file)
    
    all_metrics = []
    overall_metrics = defaultdict(list)
    
    for result, filename in zip(results, image_filenames):
        # Get detection coordinates from mask
        detection_coords = create_detection_mask(result['detection_mask'], 
                                              result['detection_mask'].shape)
        
        # Get ground truth coordinates for this image
        gt_coords = ground_truth.get(filename, [])
        
        # Calculate metrics
        metrics = calculate_detection_metrics(gt_coords, detection_coords)
        metrics['filename'] = filename
        all_metrics.append(metrics)
        
        # Aggregate metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                overall_metrics[key].append(value)
    
    # Calculate average metrics
    average_metrics = {
        'avg_mse': np.mean(overall_metrics['mse']),
        'avg_precision': np.mean(overall_metrics['precision']),
        'avg_recall': np.mean(overall_metrics['recall']),
        'total_matched': sum(overall_metrics['matched_detections']),
        'total_false_positives': sum(overall_metrics['false_positives']),
        'total_false_negatives': sum(overall_metrics['false_negatives']),
        'per_image_metrics': all_metrics
    }
    
    return average_metrics