# import os
# import torch
# import numpy as np
# from torchvision import datasets, transforms
# from sklearn.model_selection import train_test_split
# import cv2
# from utils import transform_input, TARGET_SIZE

# class PascalVOCDataset:
#     """Custom dataset loader for PASCAL VOC 2012 with lazy loading."""
    
#     def __init__(self, root_dir, split='train', transform=None):
#         """
#         Initialize PASCAL VOC 2012 dataset.
        
#         Args:
#             root_dir (str): Root directory of PASCAL VOC 2012 dataset.
#             split (str): 'train', 'val', or 'trainval'.
#             transform (callable, optional): Transform to apply to images.
#         """
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform or transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
        
#         # Load PASCAL VOC dataset
#         self.dataset = datasets.VOCDetection(
#             root=root_dir,
#             year='2012',
#             image_set=split,
#             download=True  # Assume dataset is already downloaded
#         )
        
#         self.class_names = [
#             'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#             'train', 'tvmonitor'
#         ]
        
#         # Store only image IDs for lazy loading
#         self.img_ids = [self.dataset[i][1]['annotation']['filename'].split('.')[0] 
#                         for i in range(len(self.dataset))]
    
#     def __len__(self):
#         """
#         Return the total number of samples in the dataset.
#         """
#         return len(self.img_ids)
    
#     def __getitem__(self, idx):
#         """
#         Get item by index with lazy loading.
        
#         Args:
#             idx (int): Index of the item.
            
#         Returns:
#             tuple: (image, bboxes, labels)
#                 - image: Tensor of shape [C, H, W] after transform (normalized).
#                 - bboxes: List of [xmin, ymin, xmax, ymax] (int).
#                 - labels: List of class names (str).
#         """
#         img_id = self.img_ids[idx]
#         try:
#             img, target = self.dataset[idx]
#         except Exception as e:
#             raise ValueError(f"Error loading image {img_id}: {e}")
        
#         # Convert image to numpy array (H, W, C, RGB)
#         img_np = np.array(img)
#         if img_np.ndim == 2:  # Handle grayscale images
#             img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
#         # Resize image to 448x448
#         img_np = cv2.resize(img_np, TARGET_SIZE).astype(np.float32)
#         img_np = img_np / 255.0  # Normalize to [0,1]
        
#         # Extract bounding boxes and labels
#         bboxes = []
#         labels = []
#         objects = target['annotation']['object']
#         if isinstance(objects, dict):  # Handle single object case
#             objects = [objects]
            
#         orig_width, orig_height = float(target['annotation']['size']['width']), float(target['annotation']['size']['height'])
#         for obj in objects:
#             try:
#                 bbox = obj['bndbox']
#                 # Convert to [xmin, ymin, xmax, ymax] and scale to 448x448
#                 xmin = int(float(bbox['xmin']) * TARGET_SIZE[0] / orig_width)
#                 ymin = int(float(bbox['ymin']) * TARGET_SIZE[1] / orig_height)
#                 xmax = int(float(bbox['xmax']) * TARGET_SIZE[0] / orig_width)
#                 ymax = int(float(bbox['ymax']) * TARGET_SIZE[1] / orig_height)
#                 # Ensure valid bounding box
#                 if xmin < 0 or ymin < 0 or xmax > TARGET_SIZE[0] or ymax > TARGET_SIZE[1]:
#                     continue
#                 bboxes.append([xmin, ymin, xmax, ymax])
#                 labels.append(obj['name'])
#             except (KeyError, ValueError) as e:
#                 print(f"Warning: Skipping invalid bounding box for image {img_id}: {e}")
#                 continue
        
#         # Apply transform
#         if self.transform:
#             img_np = self.transform(img_np)
        
#         return img_np, bboxes, labels
    
#     def get_class_names(self):
#         """
#         Return the list of class names.
#         """
#         return self.class_names

# def load_pascal_voc(root_dir, train_ratio=0.8, il_ratio=0.2):
#     """
#     Load and split PASCAL VOC 2012 dataset into train (IL + DQN) and validation sets.
    
#     Args:
#         root_dir (str): Root directory of PASCAL VOC 2012 dataset.
#         train_ratio (float): Ratio of data for training (default: 0.8).
#         il_ratio (float): Ratio of training data for Imitation Learning (default: 0.2).
        
#     Returns:
#         dict: Dictionary containing 'train_il', 'train_dqn', and 'val' datasets.
#               Each dataset is a dict with image IDs as keys and (image, bboxes, labels) as values.
#         list: List of class names.
#     """
#     # Load trainval dataset
#     trainval_dataset = PascalVOCDataset(root_dir, split='trainval')
    
#     # Split into train and validation
#     img_ids = trainval_dataset.img_ids
#     train_ids, val_ids = train_test_split(img_ids, train_size=train_ratio, random_state=42)
    
#     # Further split train into IL and DQN
#     train_il_ids, train_dqn_ids = train_test_split(train_ids, train_size=il_ratio, random_state=42)
    
#     # Create dataset splits as dictionaries
#     datasets = {
#         'train_il': {img_id: trainval_dataset.__getitem__(trainval_dataset.img_ids.index(img_id)) 
#                      for img_id in train_il_ids},
#         'train_dqn': {img_id: trainval_dataset.__getitem__(trainval_dataset.img_ids.index(img_id)) 
#                       for img_id in train_dqn_ids},
#         'val': {img_id: trainval_dataset.__getitem__(trainval_dataset.img_ids.index(img_id)) 
#                 for img_id in val_ids}
#     }
    
#     return datasets, trainval_dataset.class_names

# def get_env_config(root_dir):
#     """
#     Update env_config with PASCAL VOC 2012 dataset.
    
#     Args:
#         root_dir (str): Root directory of PASCAL VOC 2012 dataset.
        
#     Returns:
#         dict: Updated env_config with dataset and class names.
#     """
#     from utils import env_config
#     datasets, class_names = load_pascal_voc(root_dir)
#     env_config['dataset'] = datasets
#     env_config['current_class'] = class_names
#     env_config['n_classes'] = len(class_names)
#     return env_config


import os
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import cv2
from utils import transform_input, N_CLASSES, TARGET_SIZE
from torch.utils.data import DataLoader

class PascalVOCDataset:
    """Custom dataset loader for PASCAL VOC 2012 with lazy loading."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize PASCAL VOC 2012 dataset.
        
        Args:
            root_dir (str): Root directory of PASCAL VOC 2012 dataset.
            split (str): 'train', 'val', or 'trainval'.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 1])
        ])
        
        # Load dataset
        self.dataset = datasets.VOCDetection(
            root=root_dir,
            year='2012',
            image_set=split,
            download=False  # Assume dataset is already downloaded
        )
        
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # Store image IDs and indices for lazy loading
        self.img_ids = [self.dataset[i][1]['annotation']['filename'].split('.')[0] 
                         for i in range(len(self.dataset))]
        self.indices = list(range(len(self.dataset)))
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.indices)
    
    def __getitem__(self, index):
        """
        Get item by index with lazy loading.
        
        Args:
            index (int): Index of the item.
            
        Returns:
            tuple: (image_id, image, bboxes, labels)
                - image_id: String, filename without extension.
                - image: Tensor of shape [C, H, W] after transform (normalized                - bboxes: List of [xmin, ymin, xmax, ymax] (int).
                - labels: List of class names (str).
            
        """
        img_id = self.img_ids[index]
        try:
            img, target = self.dataset[index]
        except Exception as e:
            raise ValueError(f"Error loading image {img_id}: {e}")
        
        # Convert image to numpy array H, W, C
        img_np = np.array(img)
        if img_np.shape.nd == 2:  # Handle grayscale images
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # Resize image to 448x448
        img_np = cv2.resize(img_np, TARGET_SIZE).astype(np.float32)
        img_np = img_np / 255.0  # Normalize to [0,1]
        
        # Extract bounding boxes and labels
        bboxes = []
        labels = []
        objects = target['annotation']['object']
        if isinstance(objects, dict):  # Handle single object case
            objects = [objects]
            
        orig_width, orig_height = float(target['annotation']['size']['width']), float(target['annotation']['size']['height'])
        for obj in objects:
            try:
                bbox = obj['bndbox']
                # Convert to [xmin, ymin, ymax] and scale to 448x448
                xmin = int(float(bbox['xmin']) * TARGET_SIZE[0] / orig_width)
                
                ymin = int(float(bbox['ymin']) * TARGET_SIZE[1] / orig_height)
                
                xmax = int(float(bbox['xmax']) * TARGET_SIZE[0] / orig_width)
                
                ymax = int(float(bbox['ymax']) * TARGET_SIZE[1] / orig_height)
                
                # Ensure valid bounding box
                if xmin < 0 or ymin < 0 or xmax > TARGET_SIZE[0] or ymax > TARGET_SIZE[1]:
                    continue
                
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(obj['name'])
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping invalid bounding box for image {img_id}: {e}")
                
                continue
        
        # Apply transform
        if self.transform:
            img_np = self.transform(img_np)
        
        return img_id, img_np, bboxes, labels
    
    def get_class_names(self):
        """
        Return the list of class names.
        """
        return self.class_names

def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length bboxes and labels.
    
    Args:
        batch: List of tuples (img_id, img, bboxes, labels) from __getitem__.
        
    Returns:
        tuple: (img_ids, imgs, bboxes_list, labels_list)
            - img_ids: List of image IDs (str).
            - imgs: Tensor of shape [B, C, H, W].
            - bboxes_list: List of B lists, each containing [xmin, ymin, xmax, ymax].
            - labels_list: List of B lists, each containing class names (str).
    """
    img_ids, imgs, bboxes_list, labels_list = zip(*batch)
    imgs = torch.stack(imgs)  # [B, C, H, W]
    return list(img_ids), imgs, list(bboxes_list), list(labels_list)

def load_pascal_voc(root_dir, train_ratio=0.8, il_ratio=0.2):
    """
    Load and split PASCAL VOC 2012 dataset into train (IL + DQN) and validation sets.
    
    Args:
        root_dir (str): Root directory of PASCAL VOC 2012 dataset.
        train_ratio (float): Ratio of data for training (default: 0.8).
        il_ratio (float): Ratio of training data for Imitation Learning (default: 0.2).
        
    Returns:
        dict: Dictionary containing 'train_il', 'train_dqn', and 'val' datasets.
              Each dataset is a tuple (PascalVOCDataset, DataLoader).
        list: List of class names.
    """
    # Load trainval dataset
    trainval_dataset = PascalVOCDataset(root_dir, split='trainval')
    
    # Split into train and validation
    indices = trainval_dataset.indices
    img_ids = trainval_dataset.img_ids
    train_indices, val_indices = train_test_split(indices, train_size=train_ratio, random_state=42)
    
    # Further split train into IL and DQN
    train_il_indices, train_dqn_indices = train_test_split(train_indices, train_size=il_ratio, random_state=42)
    
    # Create dataset instances
    train_il_dataset = PascalVOCDataset(root_dir, split='trainval', transform=trainval_dataset.transform)
    train_dqn_dataset = PascalVOCDataset(root_dir, split='trainval', transform=trainval_dataset.transform)
    val_dataset = PascalVOCDataset(root_dir, split='trainval', transform=trainval_dataset.transform)
    
    # Assign indices and img_ids
    train_il_dataset.indices = train_il_indices
    train_il_dataset.img_ids = [img_ids[i] for i in train_il_indices]
    train_dqn_dataset.indices = train_dqn_indices
    train_dqn_dataset.img_ids = [img_ids[i] for i in train_dqn_indices]
    val_dataset.indices = val_indices
    val_dataset.img_ids = [img_ids[i] for i in val_indices]
    
    # Create DataLoaders
    train_il_loader = DataLoader(train_il_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    train_dqn_loader = DataLoader(train_dqn_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    # Create dataset splits
    datasets = {
        'train_il': (train_il_dataset, train_il_loader),
        'train_dqn': (train_dqn_dataset, train_dqn_loader),
        'val': (val_dataset, val_loader)
    }
    
    return datasets, trainval_dataset.class_names

def get_env_config(root_dir):
    """
    Update env_config with PASCAL VOC 2012 dataset.
    
    Args:
        root_dir (str): Root directory of PASCAL VOC 2012 dataset.
        
    Returns:
        dict: Updated env_config with dataset and class names.
    """
    from utils import env_config
    datasets, class_names = load_pascal_voc(root_dir)
    env_config['dataset'] = datasets
    env_config['current_class'] = class_names
    env_config['n_classes'] = len(class_names)
    return env_config