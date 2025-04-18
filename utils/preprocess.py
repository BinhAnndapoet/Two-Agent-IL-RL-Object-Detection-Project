# utils/preprocess.py
import cv2
import torch
import numpy as np
import torch.nn.functional as F

def preprocess_image(img, img_size):
    """Preprocess image: resize and normalize."""
    img = cv2.resize(img, img_size)
    return img

def augment_data(img_tensor, bboxes, labels):
    """Apply data augmentation: random flip."""
    if np.random.rand() > 0.5:
        img_tensor = torch.flip(img_tensor, dims=[2])
        if bboxes:
            bboxes[:, 0] = 1 - bboxes[:, 0]  # Flip x-coordinate
    return img_tensor, bboxes, labels

def extract_local_patch(tensor, center, patch_size):
    """Extract patch around center."""
    x, y = center
    h, w = tensor.shape[2], tensor.shape[3]
    x_pixel = int(x * w)
    y_pixel = int(y * h)
    half_size = patch_size // 2
    x_start = max(0, x_pixel - half_size)
    x_end = min(w, x_pixel + half_size)
    y_start = max(0, y_pixel - half_size)
    y_end = min(h, y_pixel + half_size)
    
    patch = tensor[:, :, y_start:y_end, x_start:x_end]
    if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
        patch = F.pad(patch, (0, patch_size - patch.shape[3], 0, patch_size - patch.shape[2]))
    return patch