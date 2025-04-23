import cv2
import torch
import numpy as np
import torch.nn.functional as F

def preprocess_image(img, img_size):
    """Preprocess image: resize and normalize."""
    img = cv2.resize(img, img_size)
    return img

def extract_local_patch(tensor, center, patch_size, padding_mode='replicate'):
    """
    Extract a square patch around the center from a tensor, with specified padding mode.
    
    Args:
        tensor (torch.Tensor): Input tensor [B, C, H, W], e.g., image or feature map.
        center (tuple or list): Normalized coordinates (x, y) in [0, 1] for patch center.
        patch_size (int): Size of the square patch (height and width).
        padding_mode (str): Padding mode for boundary cases ('replicate', 'reflect', or 'zeros').
                           Default: 'replicate' for natural boundary handling.
    
    Returns:
        torch.Tensor: Patch tensor [B, C, patch_size, patch_size].

    """

    x, y = center
    h, w = tensor.shape[2], tensor.shape[3]

    # Convert normalized coordinates to pixel coordinates
    x_pixel = int(x * w)
    y_pixel = int(y * h)

    # Calculate patch boundaries
    half_size = patch_size // 2
    x_start = max(0, x_pixel - half_size)
    x_end = min(w, x_pixel + half_size)
    y_start = max(0, y_pixel - half_size)
    y_end = min(h, y_pixel + half_size)

    # Extract patch
    patch = tensor[:, :, y_start:y_end, x_start:x_end]

    # Pad if patch is smaller than patch_size
    if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
        pad_left = max(0, half_size - x_pixel)
        pad_right = max(0, (x_pixel + half_size) - w)
        pad_top = max(0, half_size - y_pixel)
        pad_bottom = max(0, (y_pixel + half_size) - h)
        patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode)

    return patch
