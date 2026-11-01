import torch
import torch.nn as nn

class SimpleConcatFusion(nn.Module):
    def __init__(self, d_vis, d_topo, d_clin):
        super().__init__()
        self.output_dim = d_vis + d_topo + d_clin
        
    def forward(self, f_vis, f_topo, f_clin):
        """
        Simple concatenation fusion.
        f_vis: (B, d_vis), f_topo: (B, d_topo), f_clin: (B, d_clin)
        """
        return torch.cat([f_vis, f_topo, f_clin], dim=1)

class TopologyAwareFusion(nn.Module):
    def __init__(self, d_vis, d_topo, d_clin, d_embed=64, num_heads=2):
        super().__init__()
        self.proj_vis = nn.Sequential(
            nn.Linear(d_vis, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU()
        )
        self.proj_topo = nn.Sequential(
            nn.Linear(d_topo, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU()
        )
        self.proj_clin = nn.Sequential(
            nn.Linear(d_clin, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU()
        )
        
        # Multi-Head Attention to capture associations across modalities
        self.mha = nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, batch_first=True)
        
        # Final embedding mapping
        self.fc = nn.Sequential(
            nn.Linear(d_embed * 3, d_embed),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = d_embed
        
    def forward(self, f_vis, f_topo, f_clin):
        """
        Topology-Aware Feature Fusion (TAFF) module.
        f_vis: (B, d_vis), f_topo: (B, d_topo), f_clin: (B, d_clin)
        """
        # Map each feature vector to the shared embedding space
        e_vis = self.proj_vis(f_vis).unsqueeze(1)    # (B, 1, d_embed)
        e_topo = self.proj_topo(f_topo).unsqueeze(1)  # (B, 1, d_embed)
        e_clin = self.proj_clin(f_clin).unsqueeze(1)  # (B, 1, d_embed)
        
        # Concatenate tokens: (B, 3, d_embed)
        x = torch.cat([e_vis, e_topo, e_clin], dim=1)
        
        # Self-attention over modalities
        attn_out, attn_weights = self.mha(x, x, x)
        
        # Flatten and down-project
        fused = attn_out.reshape(attn_out.size(0), -1)  # (B, 3 * d_embed)
        return self.fc(fused)
