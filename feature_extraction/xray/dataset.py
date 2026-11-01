import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from preprocessing.xray_preprocessing.preprocessor import KneeXrayPreprocessor

class KneeXrayDataset(Dataset):
    def __init__(self, cache_npz_path, tda_scaler=None, split="train", transform=None, tda_only=False):
        """
        Loads pre-cached image paths, TDA features, and labels.
        Args:
            cache_npz_path: Path to the cached npz file.
            tda_scaler: Fitted scaler (e.g. StandardScaler) to normalize TDA features.
            split: Split type ("train", "val", "test").
            transform: Custom image transform (augmentation).
        """
        cache = np.load(cache_npz_path, allow_pickle=True)
        self.image_paths = cache["image_paths"]
        self.labels = cache["labels"]
        self.patient_ids = cache["patient_ids"]
        self.tda_features = cache["tda_features"]
        self.split = split
        self.tda_only = tda_only
        
        # Check subset mode in configuration
        import yaml
        try:
            with open("configs/config_xray.yaml", "r") as f:
                config_data = yaml.safe_load(f)
            mode = config_data.get("mode", "full_training")
            subset_mode = config_data.get(mode, {}).get("subset_mode", False)
        except Exception:
            subset_mode = False
            
        # Check preprocessing options in configuration
        try:
            with open("configs/config_xray.yaml", "r") as f:
                config_data = yaml.safe_load(f)
            self.use_clahe = config_data.get("preprocessing", {}).get("use_clahe", True)
            self.use_roi = config_data.get("preprocessing", {}).get("use_roi", False)
        except Exception:
            self.use_clahe = True
            self.use_roi = False
            
        if subset_mode:
            max_samples = 300 if split == "train" else 100
            if len(self.image_paths) > max_samples:
                step = len(self.image_paths) // max_samples
                self.image_paths = self.image_paths[::step][:max_samples]
                self.labels = self.labels[::step][:max_samples]
                self.patient_ids = self.patient_ids[::step][:max_samples]
                self.tda_features = self.tda_features[::step][:max_samples]
                
        # Fit or apply TDA scaler
        self.tda_scaler = tda_scaler
        if self.tda_scaler is not None:
            self.normalized_tda = self.tda_scaler.transform(self.tda_features)
        else:
            self.normalized_tda = self.tda_features
            
        # Define standard augmentations for training split only
        if self.split == "train" and transform is None:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=10),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                # ImageNet normalizer values
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.split != "train":
            # For validation and test splits: No augmentation, only simple normalization
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        tda_feat = self.normalized_tda[idx]
        
        if self.tda_only:
            # Bypass image loading entirely
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            file_path = self.image_paths[idx]
            try:
                import cv2
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError()
                if self.use_roi:
                    h, w = img.shape
                    start_h, start_w = int(h * 0.1), int(w * 0.1)
                    img = img[start_h:int(h * 0.9), start_w:int(w * 0.9)]
                if self.use_clahe:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img = clahe.apply(img)
                img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            except Exception:
                img_resized = np.zeros((224, 224), dtype=np.uint8)
                
            img_rgb = np.stack([img_resized] * 3, axis=2) # Shape: (H, W, 3)
            
            if self.transform is not None:
                img_tensor = self.transform(img_rgb)
            else:
                img_norm = img_rgb.astype(np.float32) / 255.0
                img_tensor = torch.tensor(np.transpose(img_norm, (2, 0, 1)), dtype=torch.float32)
                
        return img_tensor, torch.tensor(tda_feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
