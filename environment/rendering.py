# --- START OF FILE environment/rendering.py ---
import cv2
import pygame
import numpy as np
import random
import colorsys
from utils.metrics import calculate_best_iou

def render(env, mode="rgb_array"):
    if env.original_image is None:
        return np.zeros((env.height, env.width, 3), dtype=np.uint8)

    img = env.original_image.copy()
    x1, y1, x2, y2 = map(int, env.bbox)

    for idx, bbox in enumerate(env.current_gt_bboxes):
        if idx not in env.detected_objects:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), env.color, 2)

    if mode == "human":
        # Pygame rendering logic...
        pass
    return img

def display(env, mode):
    img = env.original_image.copy()
    if mode == "detection":
        for bbox, color in zip(env.classification_dictionary["bbox"], env.classification_dictionary["color"]):
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        for gt_bbox in env.current_gt_bboxes:
            cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), 2)
    return img

def decode_render_action(env, action):
    # Action decoding logic...
    return str(action)

def draw_ior_cross(image, bbox):
    # Drawing logic...
    return image

def generate_random_color():
    h = random.random()
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))
# --- END OF FILE environment/rendering.py ---