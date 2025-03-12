import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import glob
import scipy.io as io

class CrowdDataset(Dataset):
    """
    Enhanced dataset class for crowd counting datasets.
    Features:
    - Improved data loading efficiency
    - Better augmentation pipeline
    - Clearer organization
    - Support for .mat ground truth files
    """
    def __init__(self, data_root, dataset='SHHA', transform=None, train=False, 
                 patch_size=128, num_patches=4, flip=False, scale_range=(0.7, 1.3)):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing the dataset
            dataset: Dataset name (e.g., 'SHHA', 'SHHB')
            transform: Image transformations to apply
            train: Whether this is for training (enables augmentation)
            patch_size: Size of image patches for cropping (training only)
            num_patches: Number of patches to extract per image (training only)
            flip: Whether to apply random horizontal flipping
            scale_range: Range for random scaling during training
        """
        self.root_path = data_root
        self.dataset = dataset
        self.transform = transform
        self.train = train
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.flip = flip
        self.scale_range = scale_range
        
        # Set up paths based on the dataset structure
        if train:
            self.img_dir = os.path.join(data_root, dataset, 'train_data', 'images')
            self.gt_dir = os.path.join(data_root, dataset, 'train_data', 'ground_truth')
        else:
            self.img_dir = os.path.join(data_root, dataset, 'test_data', 'images')
            self.gt_dir = os.path.join(data_root, dataset, 'test_data', 'ground_truth')
        
        # Get all image files
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        self.num_samples = len(self.img_list)
        
        print(f"Loaded {self.num_samples} images for {dataset} {'training' if train else 'testing'}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """Get a single sample from the dataset with optional augmentation"""
        # Load image and annotations
        img_path = self.img_list[index]
        img_basename = os.path.basename(img_path)
        img_id = img_basename.split('.')[0]
        
        # Get corresponding ground truth mat file 
        gt_path = os.path.join(self.gt_dir, f'GT_{img_id}.mat')
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Load ground truth points from mat file
        points = self._load_points_from_mat(gt_path)
        
        # Apply transformations to image
        if self.transform is not None:
            img = self.transform(img)
        
        # Training augmentations
        if self.train:
            # Random scaling
            if random.random() > 0.5:
                scale = random.uniform(*self.scale_range)
                min_size = min(img.shape[1:])
                if scale * min_size > self.patch_size:
                    # Scale image and points
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0), scale_factor=scale).squeeze(0)
                    points *= scale
            
            # Random cropping for training
            if self.num_patches > 0:
                return self._random_crop(img, points)
            
            # Random flipping
            if self.flip and random.random() > 0.5:
                img = torch.flip(img, dims=[2])  # Flip horizontally
                if len(points) > 0:
                    points[:, 0] = img.shape[2] - points[:, 0]  # Flip x-coordinates
        
        # Prepare target dictionary
        target = {
            'point': torch.Tensor(points),
            'image_id': torch.tensor([index]),
            'labels': torch.ones(points.shape[0], dtype=torch.long)
        }
        
        return img, [target]
    
    def _load_points_from_mat(self, mat_path):
        """Load crowd points from mat file"""
        try:
            mat = io.loadmat(mat_path)
            # The structure of the .mat file can vary between datasets
            # For ShanghaiTech dataset, typically points are stored in 'image_info'
            try:
                points = mat['image_info'][0, 0]['location'][0, 0]
            except:
                # Alternative structure for some datasets
                if 'annPoints' in mat:
                    points = mat['annPoints']
                else:
                    # Fallback - find the first array with shape (n, 2) where n > 0
                    points = []
                    for key in mat.keys():
                        if isinstance(mat[key], np.ndarray) and len(mat[key].shape) == 2 and mat[key].shape[1] == 2:
                            points = mat[key]
                            break
            
            return np.array(points)
        except Exception as e:
            print(f"Error loading mat file {mat_path}: {e}")
            return np.array([])  # Return empty array if file cannot be loaded
    
    def _random_crop(self, img, points):
        """Extract random patches from the image for training"""
        h, w = img.shape[1:]
        patches = []
        targets = []
        
        for _ in range(self.num_patches):
            # Make sure the image is large enough to crop
            if h <= self.patch_size or w <= self.patch_size:
                # If image is too small, use the original
                patch = img
                patch_points = points
            else:
                # Random crop location
                x1 = random.randint(0, w - self.patch_size)
                y1 = random.randint(0, h - self.patch_size)
                x2 = x1 + self.patch_size
                y2 = y1 + self.patch_size
                
                # Crop the image
                patch = img[:, y1:y2, x1:x2]
                
                # Select points that fall within the crop
                mask = (points[:, 0] >= x1) & (points[:, 0] < x2) & \
                       (points[:, 1] >= y1) & (points[:, 1] < y2)
                patch_points = points[mask].copy()
                
                # Adjust coordinates relative to the crop
                if len(patch_points) > 0:
                    patch_points[:, 0] -= x1
                    patch_points[:, 1] -= y1
            
            # Apply random flip to the patch
            if self.flip and random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])  # Flip horizontally
                if len(patch_points) > 0:
                    patch_points[:, 0] = self.patch_size - patch_points[:, 0]
            
            patches.append(patch)
            targets.append({
                'point': torch.Tensor(patch_points),
                'image_id': torch.tensor([random.randint(0, 10000)]),
                'labels': torch.ones(patch_points.shape[0], dtype=torch.long)
            })
        
        return patches, targets

def build_transforms(augment=False):
    """
    Build image transformations pipeline.
    
    Args:
        augment: Whether to include additional augmentations for training
        
    Returns:
        transform: The composed transforms
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Add additional augmentations for training
    if augment:
        pre_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
        ]
        transform_list = pre_transforms + transform_list
    
    return transforms.Compose(transform_list)

def build_crowd_dataset(data_root, dataset='SHHA', train=True):
    """
    Build a dataset for crowd counting.
    
    Args:
        data_root: Root directory containing the dataset
        dataset: Dataset name (e.g., 'SHHA', 'SHHB')
        train: Whether to build training set (vs validation)
        
    Returns:
        dataset: The constructed dataset
    """
    # Build transforms
    transform = build_transforms(augment=train)
    
    # Create dataset
    dataset = CrowdDataset(
        data_root=data_root,
        dataset=dataset,
        transform=transform,
        train=train,
        patch_size=128,
        num_patches=4 if train else 0,
        flip=train,
        scale_range=(0.7, 1.3)
    )
    
    return dataset

def collate_fn(batch):
    """Custom collate function to handle variable sized images and patches"""
    # Flatten the batch if it contains patches
    flattened_batch = []
    for item in batch:
        imgs, targets = item
        if isinstance(imgs, list):
            # This is a batch of patches
            for i in range(len(imgs)):
                flattened_batch.append((imgs[i], targets[i]))
        else:
            # Single image
            flattened_batch.append((imgs, targets))
    
    # Group images and targets
    images = [item[0] for item in flattened_batch]
    targets = [item[1] for item in flattened_batch]
    
    # Handle variable-sized images by padding
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])
    
    # Round to multiple of 32 for better GPU utilization
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32
    
    # Pad images to same size
    padded_images = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded_img = torch.nn.functional.pad(
            img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_img)
    
    # Stack images into a batch tensor
    batched_images = torch.stack(padded_images)
    
    return batched_images, targets