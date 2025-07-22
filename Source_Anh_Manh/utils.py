import os
import torch
import random
import numpy as np
import pandas as pd
import cv2
from collections import namedtuple, deque


# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_IL_TRAJECTORIES = 100 # Nếu train IL cho mỗi agent thì 100 trajectory hơi ít, nên tăng lên 500-1,000 hoặc lấy hết batch train IL để pretrain đủ tốt.
NUM_IL_EPOCHS = 100 # Hợp lý, nhưng nên log loss để biết khi nào early stopping.
REPLAY_BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 100 # Chuẩn cho DQN, nếu thấy DQN lắc lư mạnh có thể giảm xuống (50).
EXPLORATION_MODE = 'GUIDED_EXPLORE'
ALPHA = 0.001 # Phù hợp cho Adam, nếu model sâu hơn có thể giảm còn 0.0005.
EPS_START = 0.9
GUIDED_EXPLORE = 'GUIDED_EXPLORE'
GAMMA = 0.99
EPS_END = 0.05
EPS_DECAY = 0.999
SUCCESS_CRITERIA_EPS = 1000
NU = 0.5
THRESHOLD = 0.6 #  Phù hợp tiêu chuẩn object detection, nếu muốn agent dễ hơn có thể giảm về 0.5.
MAX_STEPS = 200 # Nếu episode quá dài, cân nhắc giảm còn 100-150.
TRIGGER_STEPS = 10 # Hợp lý (tránh trigger quá sớm), có thể tune thêm.
NUMBER_ACTIONS = 6
ACTION_HISTORY_SIZE = 7
OBJ_CONFIG = 'MULTI_OBJECT'
N_CLASSES = 20
TARGET_SIZE = (448, 448)
FEATURE_DIM = 512
USE_DATASET = True
ENV_MODE = 0
EPOCHS = 100
DATASET = None
CURRENT_CLASSES = None # Danh sách các lớp, cập nhật từ dataset
WINDOW_SIZE = (448, 448)
PHASE = 'il'

env_config = {
    "dataset": DATASET,
    "alpha": ALPHA,
    "current_class": CURRENT_CLASSES,
    "phase": PHASE,  # Mặc định là IL, sẽ thay đổi theo phase
    "nu": NU,
    "threshold": THRESHOLD,
    "max_steps": MAX_STEPS,
    "trigger_steps": TRIGGER_STEPS,
    "number_actions": NUMBER_ACTIONS,
    "action_history_size": ACTION_HISTORY_SIZE,
    "object_config": OBJ_CONFIG,
    "n_classes": N_CLASSES,
    "target_size": TARGET_SIZE,
    "feature_dim": FEATURE_DIM,
    "device": device,
    "use_dataset": USE_DATASET,
    "env_mode": ENV_MODE,
    "epochs": EPOCHS,
    "window_size": WINDOW_SIZE,
    "num_il_trajectories": NUM_IL_TRAJECTORIES, # [UPDATE]: add num_il_trajectories to env_config
    "target_update_freq": TARGET_UPDATE_FREQ,
    "exploration_mode": EXPLORATION_MODE  # [FIX]: Add missing exploration_mode key
}

# Transition named tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY, batch_size=BATCH_SIZE):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.rewards = deque(maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)
        self.rewards.append(transition.reward)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))
        states = torch.from_numpy(np.array(batch.state, dtype=np.float32))
        actions = torch.from_numpy(np.array(batch.action, dtype=np.int64)).unsqueeze(1)
        rewards = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).unsqueeze(1)
        dones = torch.from_numpy(np.array(batch.done, dtype=np.bool_)).unsqueeze(1).to(torch.bool)
        next_states = torch.from_numpy(np.array(batch.next_state, dtype=np.float32))
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory)

def transform_input(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def calculate_best_iou(pred_boxes, gt_boxes):
    return max(calculate_iou(pred, gt) for pred in pred_boxes for gt in gt_boxes) if pred_boxes and gt_boxes else 0.0

def calculate_best_recall(pred_boxes, gt_boxes):
    iou_threshold = 0.5
    ious = [calculate_iou(pred, gt) for pred in pred_boxes for gt in gt_boxes]
    return sum(1 for iou in ious if iou >= iou_threshold) / len(gt_boxes) if gt_boxes else 0.0

def calculate_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    """
    Calculate mean Average Precision (mAP) at IoU=0.5.
    """
    if not pred_boxes or not gt_boxes or not pred_labels or not gt_labels or not pred_scores:
        return 0.0
    if len(pred_boxes) != len(pred_labels) or len(pred_boxes) != len(pred_scores):
        print("Warning: Mismatch in pred_boxes, pred_labels, or pred_scores lengths")
        return 0.0

    iou_threshold = 0.5
    classes = list(set(gt_labels))
    ap_scores = []

    for cls in classes:
        pred_cls_mask = [lbl == cls for lbl in pred_labels]
        gt_cls_mask = [lbl == cls for lbl in gt_labels]

        if not any(pred_cls_mask) or not any(gt_cls_mask):
            continue

        pred_cls_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if pred_cls_mask[i]]
        pred_cls_scores = [pred_scores[i] for i in range(len(pred_scores)) if pred_cls_mask[i]]
        gt_cls_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if gt_cls_mask[i]]

        # Sort predictions by score
        sorted_indices = np.argsort(pred_cls_scores)[::-1]
        pred_cls_boxes = [pred_cls_boxes[i] for i in sorted_indices]
        pred_cls_scores = [pred_cls_scores[i] for i in sorted_indices]

        # Calculate TP and FP
        true_positives = np.zeros(len(pred_cls_boxes))
        false_positives = np.zeros(len(pred_cls_boxes))
        matched_gt = set()

        for i, pred_box in enumerate(pred_cls_boxes):
            max_iou = 0
            max_gt_idx = -1
            for j, gt_box in enumerate(gt_cls_boxes):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j

            if max_iou >= iou_threshold and max_gt_idx != -1:
                true_positives[i] = 1
                matched_gt.add(max_gt_idx)
            else:
                false_positives[i] = 1

        # Calculate precision and recall
        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(false_positives)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
        recalls = cum_tp / len(gt_cls_boxes) if gt_cls_boxes else np.zeros_like(cum_tp)

        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0
