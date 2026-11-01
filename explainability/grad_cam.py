import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activations)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradients)
        else:
            self.backward_hook = self.target_layer.register_backward_hook(self.save_gradients)
            
    def save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
        
    def generate_heatmap(self, input_tensor, class_idx):
        """
        Generates the 2D activation heatmap for the given target class index.
        """
        # Reset gradients
        self.gradients = None
        self.activations = None
        
        # Forward pass on the model (or backbone)
        self.model.zero_grad()
        
        if isinstance(input_tensor, tuple):
            logits = self.model(*input_tensor)
        else:
            try:
                logits = self.model(input_tensor)
            except Exception:
                # Try passing dummy TDA input if it's the hybrid model
                dummy_tda = torch.zeros((input_tensor.size(0), 16), device=input_tensor.device)
                logits = self.model(input_tensor, dummy_tda)
                
        if isinstance(logits, tuple):
            logits = logits[0]
            
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        score = logits[0, class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            # Fallback if hooks fail to record
            return np.ones((7, 7), dtype=np.float32)
            
        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True) # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        weighted_activation = torch.sum(weights * self.activations, dim=1).squeeze(0) # (H, W)
        
        # Apply ReLU to retain positive activations contributing to class score
        heatmap = torch.clamp(weighted_activation, min=0)
        
        # Normalize between [0, 1]
        max_val = torch.max(heatmap).item()
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap.cpu().numpy()
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Overlays the Grad-CAM activation heatmap onto the original image at its native resolution.
    original_img: grayscale image as uint8 numpy array.
    """
    # Resize heatmap to match the original image dimensions
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Scale to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap JET
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert grayscale original to BGR
    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_img.copy()
        
    # Fuse them
    overlay = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1.0 - alpha, 0)
    return overlay
