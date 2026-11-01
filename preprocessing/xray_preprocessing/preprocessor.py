import cv2
import numpy as np

class KneeXrayPreprocessor:
    def __init__(self, target_size=(224, 224), use_clahe=False, use_roi=False, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.use_roi = use_roi
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            
    def apply_roi_crop(self, img):
        """
        Knee X-rays in KneeXrayMini are already cropped around the joint, 
        but we can crop the central 85% area to focus on the joint gap 
        and remove excessive black borders or labels if present.
        """
        h, w = img.shape[:2]
        crop_h = int(h * 0.85)
        crop_w = int(w * 0.85)
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        return img[start_y:start_y+crop_h, start_x:start_x+crop_w]
        
    def preprocess_image(self, file_path):
        """
        Executes X-ray preprocessing pipeline:
        Grayscale handling -> ROI cropping -> CLAHE -> Resize -> Intensity scale.
        Returns:
            img_gray: Grayscale 2D array of target_size (for TDA and display).
            img_rgb: RGB 3-channel normalized array of target_size (for CNN).
        """
        # Read image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
            
        # 1. ROI handling
        if self.use_roi:
            img = self.apply_roi_crop(img)
            
        # 2. Contrast enhancement (CLAHE)
        if self.use_clahe:
            img = self.clahe.apply(img)
            
        # 3. Resize
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 4. Intensity normalization [0, 1]
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Create 3-channel version for ImageNet ResNet50
        img_rgb = np.stack([img_norm] * 3, axis=0) # Shape: (3, H, W)
        
        # ImageNet normalization: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_rgb = (img_rgb - mean) / std
        
        return img_resized, img_rgb
