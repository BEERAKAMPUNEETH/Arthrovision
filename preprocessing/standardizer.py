import numpy as np
import cv2
import SimpleITK as sitk
from scipy.ndimage import median_filter

class ImageStandardizer:
    def __init__(self, target_img_size=(224, 224), target_vol_size=(64, 64, 32)):
        self.target_img_size = target_img_size
        self.target_vol_size = target_vol_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
    def read_volume(self, file_path):
        """Reads a 3D medical volume using SimpleITK."""
        sitk_img = sitk.ReadImage(file_path)
        # Convert to numpy array (Z, Y, X) and transpose to (X, Y, Z)
        vol = sitk.GetArrayFromImage(sitk_img).transpose(2, 1, 0)
        return vol, sitk_img
        
    def preprocess_volume(self, vol):
        """Standardizes intensity and applies noise reduction to the 3D volume."""
        # 1. Intensity normalization (Min-Max)
        vol_min = vol.min()
        vol_max = vol.max()
        if vol_max - vol_min > 0:
            vol_normalized = (vol - vol_min) / (vol_max - vol_min)
        else:
            vol_normalized = vol.copy()
            
        # 2. Denoising using a 3D median filter (mild)
        vol_denoised = median_filter(vol_normalized, size=3)
        return vol_denoised

    def extract_joint_roi(self, vol, mask, margin=2):
        """
        Crops the 3D volume to the bounding box containing the knee joint anatomy (femur, tibia, cartilage).
        """
        non_zero = np.argwhere(mask > 0)
        if len(non_zero) == 0:
            return vol, mask
            
        x_min, y_min, z_min = non_zero.min(axis=0)
        x_max, y_max, z_max = non_zero.max(axis=0)
        
        # Add margin with boundary constraints
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        z_min = max(0, z_min - margin)
        x_max = min(vol.shape[0], x_max + margin + 1)
        y_max = min(vol.shape[1], y_max + margin + 1)
        z_max = min(vol.shape[2], z_max + margin + 1)
        
        cropped_vol = vol[x_min:x_max, y_min:y_max, z_min:z_max]
        cropped_mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
        
        return cropped_vol, cropped_mask

    def select_top_k_cartilage_slices(self, vol, mask, k=5):
        """
        Selects K slices along the Z axis that contain the highest amount of joint space cartilage (class 3).
        If cartilage is absent, falls back to slices containing any bones (class 1 or 2) or central slices.
        """
        z_dim = vol.shape[2]
        cartilage_counts = []
        
        for z in range(z_dim):
            # Cartilage is class 3 in mask
            count = np.sum(mask[:, :, z] == 3)
            cartilage_counts.append((z, count))
            
        # Sort slices by cartilage presence
        sorted_slices = sorted(cartilage_counts, key=lambda x: x[1], reverse=True)
        
        # If cartilage is missing or very sparse, fallback to bone density or general central slices
        if sorted_slices[0][1] == 0:
            bone_counts = []
            for z in range(z_dim):
                count = np.sum((mask[:, :, z] == 1) | (mask[:, :, z] == 2))
                bone_counts.append((z, count))
            sorted_slices = sorted(bone_counts, key=lambda x: x[1], reverse=True)
            
        # Get top K slice indices
        top_k_indices = [idx for idx, _ in sorted_slices[:k]]
        
        # If the volume is thinner than K slices, pad indices by duplication
        while len(top_k_indices) < k:
            top_k_indices.append(top_k_indices[-1] if len(top_k_indices) > 0 else 0)
            
        # Sort indices to maintain spatial ordering (bottom to top)
        top_k_indices.sort()
        return top_k_indices
        
    def extract_visual_slices(self, vol, mask, k=5):
        """
        Extracts and standardizes K clinically relevant 2D joint slices for the ResNet50 visual branch.
        Also returns the raw central slice at its native resolution for Grad-CAM display.
        """
        # Crop to joint space ROI
        cropped_vol, cropped_mask = self.extract_joint_roi(vol, mask)
        
        # Select K slice indices based on cartilage density
        slice_indices = self.select_top_k_cartilage_slices(cropped_vol, cropped_mask, k=k)
        
        # Extract and preprocess each slice
        preprocessed_slices = []
        for idx in slice_indices:
            raw_slice = cropped_vol[:, :, idx].copy()
            
            # Map intensity to [0, 255]
            slice_uint8 = (raw_slice * 255.0).astype(np.uint8)
            
            # Apply CLAHE enhancement
            slice_enhanced = self.clahe.apply(slice_uint8)
            
            # Resize
            slice_resized = cv2.resize(slice_enhanced, self.target_img_size, interpolation=cv2.INTER_LINEAR)
            
            # Scale to [0, 1]
            img_input = slice_resized.astype(np.float32) / 255.0
            
            # Repeat to 3 channels (RGB) for ResNet50
            img_input = np.stack([img_input] * 3, axis=0) # (3, H, W)
            
            # ImageNet normalize
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_input = (img_input - mean) / std
            
            preprocessed_slices.append(img_input)
            
        # Format as stacked array: (K, 3, 224, 224)
        stacked_slices = np.stack(preprocessed_slices, axis=0)
        
        # Expose middle slice from the original uncropped volume for high-res visualization
        mid_z = vol.shape[2] // 2
        orig_vis = (vol[:, :, mid_z].copy() * 255.0).astype(np.uint8)
        
        return stacked_slices, orig_vis

    def extract_visual_slice(self, vol):
        """Deprecated: Fallback for single slice extraction."""
        mid_z = vol.shape[2] // 2
        raw_slice = vol[:, :, mid_z].copy()
        orig_vis = (raw_slice * 255.0).astype(np.uint8)
        slice_enhanced = self.clahe.apply(orig_vis)
        slice_resized = cv2.resize(slice_enhanced, self.target_img_size, interpolation=cv2.INTER_LINEAR)
        img_input = slice_resized.astype(np.float32) / 255.0
        img_input = np.stack([img_input] * 3, axis=0)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_input = (img_input - mean) / std
        return img_input, orig_vis
