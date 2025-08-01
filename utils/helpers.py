# --- START OF FILE utils/helpers.py ---
import cv2
import torch
import numpy as np

def transform_input(image, target_size):
    """Transforms a numpy image to a normalized tensor."""
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)
# --- END OF FILE utils/helpers.py ---