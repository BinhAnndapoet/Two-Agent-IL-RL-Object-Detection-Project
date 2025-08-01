# --- START OF FILE data/dataset.py ---
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from typing import Tuple, List
import torch

from config import TARGET_SIZE

class PascalVOCDataset(Dataset):
    def __init__(self, root_dir: str, image_set: str, indices=None):
        self.dataset = datasets.VOCDetection(root=root_dir, year='2012', image_set=image_set)
        self.indices = indices if indices is not None else list(range(len(self.dataset)))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, List[List[int]], List[str]]:
        img, target = self.dataset[self.indices[index]]
        img_np = cv2.resize(np.array(img), TARGET_SIZE)
        
        bboxes, labels = [], []
        orig_w, orig_h = float(target['annotation']['size']['width']), float(target['annotation']['size']['height'])
        
        objects = target['annotation']['object']
        if not isinstance(objects, list): objects = [objects]
            
        for obj in objects:
            bndbox = obj['bndbox']
            xmin = int(float(bndbox['xmin']) * TARGET_SIZE[0] / orig_w)
            ymin = int(float(bndbox['ymin']) * TARGET_SIZE[1] / orig_h)
            xmax = int(float(bndbox['xmax']) * TARGET_SIZE[0] / orig_w)
            ymax = int(float(bndbox['ymax']) * TARGET_SIZE[1] / orig_h)
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['name'])
        
        img_tensor = self.transform(img_np)
        img_id = target['annotation']['filename']
        return img_id, img_tensor, bboxes, labels

def custom_collate_fn(batch):
    img_ids, imgs, bboxes_list, labels_list = zip(*batch)
    return list(img_ids), torch.stack(imgs), list(bboxes_list), list(labels_list)

def load_pascal_voc(root_dir: str, il_ratio: float = 0.2):
    trainval_dataset_info = datasets.VOCDetection(root=root_dir, year='2012', image_set='trainval', download=True)
    indices = list(range(len(trainval_dataset_info)))
    
    train_il_indices, train_dqn_indices = train_test_split(indices, train_size=il_ratio, random_state=42)
    
    train_il_dataset = PascalVOCDataset(root_dir, 'trainval', train_il_indices)
    train_dqn_dataset = PascalVOCDataset(root_dir, 'trainval', train_dqn_indices)
    test_dataset = PascalVOCDataset(root_dir, 'val')

    datasets_dict = {
        'train_il': (train_il_dataset, DataLoader(train_il_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)),
        'train_dqn': (train_dqn_dataset, DataLoader(train_dqn_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)),
        'test': (test_dataset, DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn))
    }
    
    print(f"Dataset splits: IL={len(train_il_dataset)}, DQN={len(train_dqn_dataset)}, Test={len(test_dataset)}")
    return datasets_dict, train_il_dataset.class_names
# --- END OF FILE data/dataset.py ---