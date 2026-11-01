import torch
import torch.nn as nn
import torchvision.models as models

class KneeResNet50(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # Load ResNet50 with weights fallback for older torchvision versions
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            self.backbone = models.resnet50(pretrained=pretrained)
            
        self.feature_dim = self.backbone.fc.in_features # 2048
        self.backbone.fc = nn.Identity() # Remove default FC layer
        
        # Expose a projection head for visual-only baseline classification
        self.fc_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: input image tensor of shape (B, 3, 224, 224)
        Returns:
            logits: classification logits for KL grade prediction
            features: 2048-dimensional visual feature representation
        """
        features = self.backbone(x) # Shape: (B, 2048)
        logits = self.fc_head(features)
        return logits, features
