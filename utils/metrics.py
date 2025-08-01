# --- START OF FILE utils/metrics.py ---
import numpy as np

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def calculate_best_iou(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return 0.0
    return max(calculate_iou(pred, gt) for pred in pred_boxes for gt in gt_boxes)

def calculate_best_recall(pred_boxes, gt_boxes):
    if not gt_boxes:
        return 0.0
    iou_threshold = 0.5
    detected_count = sum(1 for gt in gt_boxes if any(calculate_iou(pred, gt) >= iou_threshold for pred in pred_boxes))
    return detected_count / len(gt_boxes)

def calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    if not pred_boxes or not gt_boxes:
        return 0.0

    iou_threshold = 0.5
    unique_classes = set(gt_labels)
    ap_scores = []

    for cls in unique_classes:
        pred_cls_indices = [i for i, lbl in enumerate(pred_labels) if lbl == cls]
        gt_cls_boxes = [box for i, box in enumerate(gt_boxes) if gt_labels[i] == cls]

        if not pred_cls_indices or not gt_cls_boxes:
            continue

        pred_cls_boxes = [pred_boxes[i] for i in pred_cls_indices]
        pred_cls_scores = [pred_scores[i] for i in pred_cls_indices]
        
        sorted_indices = np.argsort(pred_cls_scores)[::-1]
        
        tp = np.zeros(len(pred_cls_boxes))
        fp = np.zeros(len(pred_cls_boxes))
        matched_gt = set()

        for i in sorted_indices:
            pred_box = pred_cls_boxes[i]
            max_iou = -1
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_cls_boxes):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = j

            if max_iou >= iou_threshold:
                if best_gt_idx not in matched_gt:
                    tp[i] = 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
        recalls = cum_tp / len(gt_cls_boxes)
        
        precisions = np.concatenate(([0.], precisions, [0.]))
        recalls = np.concatenate(([0.], recalls, [1.]))
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        ap = 0
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0
# --- END OF FILE utils/metrics.py ---