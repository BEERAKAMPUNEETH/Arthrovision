import torch
import torch.nn as nn
from fusion.multi_modal_fusion import SimpleConcatFusion, TopologyAwareFusion

class HybridKneeClassifier(nn.Module):
    def __init__(
        self, 
        d_vis=2048, 
        d_topo=16, 
        d_clin=4, 
        num_classes=5, 
        fusion_type="topology_aware", 
        embed_dim=64,
        dropout=0.3,
        use_visual=True,
        use_topological=True,
        use_clinical=True,
        ordinal=False
    ):
        super().__init__()
        self.use_visual = use_visual
        self.use_topological = use_topological
        self.use_clinical = use_clinical
        self.fusion_type = fusion_type
        self.ordinal = ordinal
        self.num_classes = num_classes
        
        # Calculate active dimensions
        active_vis = d_vis if use_visual else 0
        active_topo = d_topo if use_topological else 0
        active_clin = d_clin if use_clinical else 0
        
        if fusion_type == "topology_aware" and use_visual and use_topological and use_clinical:
            self.fusion = TopologyAwareFusion(
                d_vis=d_vis, 
                d_topo=d_topo, 
                d_clin=d_clin, 
                d_embed=embed_dim
            )
            in_features = self.fusion.output_dim
        else:
            total_dim = active_vis + active_topo + active_clin
            self.fusion = SimpleConcatFusion(
                d_vis=active_vis, 
                d_topo=active_topo, 
                d_clin=active_clin
            )
            in_features = self.fusion.output_dim
            
        # In CORAL ordinal classification, we predict num_classes - 1 binary cuts
        out_features = (num_classes - 1) if ordinal else num_classes
        
        # Compact Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_features)
        )
        
    def forward(self, f_vis=None, f_topo=None, f_clin=None):
        """
        Fuses available features and predicts KL grade.
        Inputs:
          f_vis: (B, d_vis) visual feature tensor
          f_topo: (B, d_topo) topological feature tensor
          f_clin: (B, d_clin) clinical feature tensor
        """
        inputs = []
        if self.use_visual:
            assert f_vis is not None, "Visual features must be provided."
            inputs.append(f_vis)
        if self.use_topological:
            assert f_topo is not None, "Topological features must be provided."
            inputs.append(f_topo)
        if self.use_clinical:
            assert f_clin is not None, "Clinical features must be provided."
            inputs.append(f_clin)
            
        if self.fusion_type == "topology_aware" and self.use_visual and self.use_topological and self.use_clinical:
            fused = self.fusion(f_vis, f_topo, f_clin)
        else:
            fused = torch.cat(inputs, dim=1)
            
        logits = self.classifier(fused)
        return logits

    def predict_probabilities(self, logits):
        """
        Helper to convert raw network outputs to 5 class probabilities.
        Handles both Standard Softmax and CORAL Ordinal representations.
        """
        if not self.ordinal:
            return torch.softmax(logits, dim=-1)
        else:
            # CORAL logits: each represents logits of P(y > k) for k = 0, 1, 2, 3
            # Sigmoid outputs P(y > k)
            cum_probs = torch.sigmoid(logits) # (B, 4)
            
            # Reconstruct class probabilities:
            # P(y = 0) = 1 - P(y > 0)
            # P(y = k) = P(y > k-1) - P(y > k)
            # P(y = 4) = P(y > 3)
            probs = torch.zeros((logits.shape[0], self.num_classes), device=logits.device)
            probs[:, 0] = 1.0 - cum_probs[:, 0]
            for k in range(1, self.num_classes - 1):
                probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]
            probs[:, self.num_classes - 1] = cum_probs[:, -1]
            
            # Clamp to prevent tiny negative numbers due to precision limits
            probs = torch.clamp(probs, min=0.0, max=1.0)
            # Renormalize to ensure they sum to 1.0
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
            return probs

class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, logits, targets):
        """
        Computes the CORAL ordinal loss.
        logits: (B, num_classes - 1)
        targets: (B,) long integer labels in [0, num_classes - 1]
        """
        # Convert target index to CORAL binary representation
        # Example: target = 2 -> [1, 1, 0, 0] for classes >0, >1, >2, >3
        B = targets.shape[0]
        ordinal_targets = torch.zeros_like(logits, dtype=torch.float32)
        for i in range(self.num_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()
            
        losses = self.bce(logits, ordinal_targets)
        return losses.sum(dim=1).mean()
