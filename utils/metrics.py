import numpy as np

def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x1_min = x1 - w1 / 2
    x1_max = x1 + w1 / 2
    y1_min = y1 - h1 / 2
    y1_max = y1 + h1 / 2
    
    x2_min = x2 - w2 / 2
    x2_max = x2 + w2 / 2
    y2_min = y2 - h2 / 2
    y2_max = y2 + h2 / 2
    
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
