import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from utils.preprocess import preprocess_image, augment_data

class VOC2012Dataset(Dataset):
    """PASCAL VOC 2012 Dataset for object detection."""
    def __init__(self, voc_root, split='train', img_size=(448, 448)):
        """
        Args:
            voc_root (str): Path to VOC2012 dataset.
            split (str): 'train' or 'val'.
            img_size (tuple): Target image size (height, width).
        """
        self.voc_root = voc_root
        self.split = split
        self.img_size = img_size
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')
        self.split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')
        
        # Load image IDs
        with open(self.split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f]
        
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # 20 classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        annotation_path = os.path.join(self.annotation_dir, f'{img_id}.xml')

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img, self.img_size)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Load annotations
        bboxes, labels = self._parse_annotation(annotation_path)

        # Apply augmentation (only for training)
        if self.split == 'train':
            img_tensor, bboxes, labels = augment_data(img_tensor, bboxes, labels)

        return {
            'image': img_tensor,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),  # [N, 4] (x, y, w, h)
            'labels': torch.tensor(labels, dtype=torch.long),     # [N]
        }

    def _parse_annotation(self, annotation_path):
        """Parse VOC XML annotation file."""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bboxes = []
        labels = []

        img_width = float(root.find('size/width').text)
        img_height = float(root.find('size/height').text)

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.classes:
                continue
            class_idx = self.classes.index(class_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to center format (x, y, w, h) and normalize
            x = (xmin + xmax) / 2 / img_width
            y = (ymin + ymax) / 2 / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height

            bboxes.append([x, y, w, h])
            labels.append(class_idx)

        return bboxes, labels
