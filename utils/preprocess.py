import cv2
import torch
import numpy as np
import torch.nn.functional as F

def preprocess_image(img, img_size):
    """Preprocess image: resize and normalize."""
    img = cv2.resize(img, img_size)
    return img

def augment_data():
    pass
