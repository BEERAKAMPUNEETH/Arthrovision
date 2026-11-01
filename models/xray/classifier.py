import torch
import torch.nn as nn
import torchvision.models as models

# --- Ordinal Head (CORAL) ---
class CoralOrdinalHead(nn.Module):
    def __init__(self, in_features, num_classes=5):
        super().__init__()
        # For K classes, CORAL uses K-1 outputs (representing cumulative binary classifiers)
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes - 1, bias=False)
        # CORAL has individual threshold bias parameters
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, x):
        logits = self.fc(x) + self.bias
        return logits
        
    def predict_probabilities(self, logits):
        """Converts cumulative binary logits into a probability distribution over classes."""
        # Shape: (batch, num_classes - 1)
        probs_cumulative = torch.sigmoid(logits)
        
        # P(Y = 0) = 1 - P(Y > 0)
        p_0 = 1.0 - probs_cumulative[:, 0:1]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        p_middle = probs_cumulative[:, :-1] - probs_cumulative[:, 1:]
        
        # P(Y = K-1) = P(Y > K-2)
        p_last = probs_cumulative[:, -1:]
        
        probs = torch.cat([p_0, p_middle, p_last], dim=1)
        return probs

def coral_loss(logits, targets, num_classes=5):
    """
    Computes Consistent Rank Logits (CORAL) loss.
    targets: shape (batch,) with class labels 0 to K-1.
    """
    batch_size = targets.size(0)
    device = targets.device
    
    # Convert integer targets into cumulative binary targets
    # e.g. target = 2 -> [1, 1, 0, 0] for K=5
    levels = torch.arange(num_classes - 1, device=device).unsqueeze(0).expand(batch_size, -1)
    binary_targets = (targets.unsqueeze(1) > levels).float()
    
    # Compute binary cross entropy loss
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(logits, binary_targets)
    return loss.sum(dim=1).mean()


# --- Parameterized Visual Backbone ---
class KneeXrayVisualBackbone(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True):
        super().__init__()
        self.model_name = model_name.lower()
        if self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.model_name == "efficientnet":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif self.model_name == "convnext":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            self.in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")
            
    def forward(self, x):
        feats = self.backbone(x)
        if len(feats.shape) > 2:
            feats = torch.flatten(feats, 1)
        return feats


# --- Topological Embedder ---
class XrayTDAEmbedder(nn.Module):
    def __init__(self, d_tda=16, d_emb=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_tda, d_emb),
            nn.BatchNorm1d(d_emb),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return self.net(x)


# --- TAFF Attention Fusion ---
class TopologyAwareFeatureFusion(nn.Module):
    def __init__(self, d_vis_emb=128, d_tda_emb=32, d_fusion=64, num_heads=4):
        super().__init__()
        self.d_fusion = d_fusion
        
        # Project visual and topological embeddings to shared fusion dimension
        self.vis_proj = nn.Linear(d_vis_emb, d_fusion)
        self.tda_proj = nn.Linear(d_tda_emb, d_fusion)
        
        # Multi-head Self-Attention layer
        self.mha = nn.MultiheadAttention(embed_dim=d_fusion, num_heads=num_heads, batch_first=True)
        
        # Fusion MLP
        self.net = nn.Sequential(
            nn.Linear(d_fusion * 2, d_fusion),
            nn.LayerNorm(d_fusion),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, vis_emb, tda_emb):
        # Project to shared space
        # shape: (batch, d_fusion)
        v_p = self.vis_proj(vis_emb)
        t_p = self.tda_proj(tda_emb)
        
        # Format as tokens: shape (batch, 2, d_fusion)
        tokens = torch.stack([v_p, t_p], dim=1)
        
        # Self-attention over visual and topological tokens
        attn_out, _ = self.mha(tokens, tokens, tokens)
        
        # Residual connection
        attn_out = attn_out + tokens
        
        # Flatten tokens to shape (batch, d_fusion * 2) and project
        flattened = attn_out.reshape(attn_out.size(0), -1)
        out = self.net(flattened)
        return out


# --- Gated Feature Fusion ---
class GatedFeatureFusion(nn.Module):
    def __init__(self, d_vis, d_tda, d_out):
        super().__init__()
        self.proj_vis = nn.Linear(d_vis, d_out)
        self.proj_tda = nn.Linear(d_tda, d_out)
        self.gate = nn.Sequential(
            nn.Linear(d_vis + d_tda, d_out),
            nn.Sigmoid()
        )
        
    def forward(self, vis, tda):
        v_proj = torch.relu(self.proj_vis(vis))
        t_proj = torch.relu(self.proj_tda(tda))
        gate_w = self.gate(torch.cat([vis, tda], dim=1))
        return gate_w * v_proj + (1.0 - gate_w) * t_proj


# --- Complete Multimodal Classifier ---
class HybridXrayClassifier(nn.Module):
    def __init__(self, d_tda=16, d_tda_emb=32, d_vis_emb=128, d_fusion=64, num_classes=5, fusion_type="taff", ordinal=False, backbone_type="resnet50"):
        """
        Args:
            fusion_type: "taff" (Attention), "concat" (simple Concatenation), "tda_only", "image_only", "gated", "late"
            ordinal: If True, uses CORAL ordinal head, else standard softmax classification
            backbone_type: "resnet50", "efficientnet", "convnext"
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.ordinal = ordinal
        self.num_classes = num_classes
        
        # Visual branch
        self.vis_backbone = KneeXrayVisualBackbone(model_name=backbone_type, pretrained=True)
        self.vis_proj = nn.Sequential(
            nn.Linear(self.vis_backbone.in_features, d_vis_emb),
            nn.BatchNorm1d(d_vis_emb),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Topological branch
        self.tda_embedder = XrayTDAEmbedder(d_tda=d_tda, d_emb=d_tda_emb)
        
        # Fusion Selection
        if self.fusion_type == "taff":
            self.fusion = TopologyAwareFeatureFusion(d_vis_emb=d_vis_emb, d_tda_emb=d_tda_emb, d_fusion=d_fusion)
            classifier_in = d_fusion
        elif self.fusion_type == "concat":
            classifier_in = d_vis_emb + d_tda_emb
        elif self.fusion_type == "gated":
            self.fusion = GatedFeatureFusion(d_vis=d_vis_emb, d_tda=d_tda_emb, d_out=d_fusion)
            classifier_in = d_fusion
        elif self.fusion_type == "late":
            out_dim = num_classes - 1 if ordinal else num_classes
            self.vis_classifier = nn.Linear(d_vis_emb, out_dim)
            self.tda_classifier = nn.Linear(d_tda_emb, out_dim)
            classifier_in = d_vis_emb  # dummy
        elif self.fusion_type == "tda_only":
            classifier_in = d_tda_emb
        elif self.fusion_type == "image_only":
            classifier_in = d_vis_emb
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
        # Classification Head
        if self.fusion_type != "late":
            if self.ordinal:
                self.classifier_head = CoralOrdinalHead(classifier_in, num_classes=num_classes)
            else:
                if self.fusion_type in ["tda_only", "concat", "gated"]:
                    hidden_dim = d_fusion
                    self.classifier_head = nn.Sequential(
                        nn.Linear(classifier_in, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dim, num_classes)
                    )
                else:
                    self.classifier_head = nn.Linear(classifier_in, num_classes)
            
    def forward(self, img_x, tda_x):
        # 1. Visual Feature extraction
        if self.fusion_type != "tda_only":
            vis_feats = self.vis_backbone(img_x)
            vis_emb = self.vis_proj(vis_feats)
            
        # 2. Topological Feature extraction
        if self.fusion_type != "image_only":
            tda_emb = self.tda_embedder(tda_x)
            
        # 3. Fusion and Classification
        if self.fusion_type == "taff":
            fused = self.fusion(vis_emb, tda_emb)
            logits = self.classifier_head(fused)
        elif self.fusion_type == "concat":
            fused = torch.cat([vis_emb, tda_emb], dim=1)
            logits = self.classifier_head(fused)
        elif self.fusion_type == "gated":
            fused = self.fusion(vis_emb, tda_emb)
            logits = self.classifier_head(fused)
        elif self.fusion_type == "late":
            logits_vis = self.vis_classifier(vis_emb)
            logits_tda = self.tda_classifier(tda_emb)
            logits = 0.5 * logits_vis + 0.5 * logits_tda
        elif self.fusion_type == "tda_only":
            fused = tda_emb
            logits = self.classifier_head(fused)
        elif self.fusion_type == "image_only":
            fused = vis_emb
            logits = self.classifier_head(fused)
            
        return logits
        
    def predict_probabilities(self, logits):
        """Converts logits to probability distribution over KL grades."""
        if self.ordinal:
            if self.fusion_type == "late":
                cum_probs = torch.sigmoid(logits)
                probs = torch.zeros(logits.size(0), self.num_classes, device=logits.device)
                probs[:, 0] = 1.0 - cum_probs[:, 0]
                for i in range(1, self.num_classes - 1):
                    probs[:, i] = cum_probs[:, i - 1] - cum_probs[:, i]
                probs[:, self.num_classes - 1] = cum_probs[:, -1]
                return probs
            else:
                return self.classifier_head.predict_probabilities(logits)
        else:
            return torch.softmax(logits, dim=1)
