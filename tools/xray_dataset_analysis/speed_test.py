import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from feature_extraction.xray.dataset import KneeXrayDataset
from models.xray.classifier import HybridXrayClassifier

def speed_test():
    print("Running speed test...")
    train_cache = "data/processed_xray/train_cache.npz"
    ds = KneeXrayDataset(train_cache, split="train")
    
    # Let's force subset_mode: false to test full data loading speed
    ds.image_paths = ds.image_paths[:100]
    ds.labels = ds.labels[:100]
    ds.patient_ids = ds.patient_ids[:100]
    ds.tda_features = ds.tda_features[:100]
    ds.normalized_tda = ds.normalized_tda[:100]
    
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HybridXrayClassifier(
        d_tda=16, d_tda_emb=32, d_vis_emb=128, d_fusion=64, num_classes=5, fusion_type="image_only"
    ).to(device)
    
    # Freeze backbone
    for param in model.vis_backbone.parameters():
        param.requires_grad = False
        
    start_time = time.time()
    
    for i, (imgs, tda, lbls) in enumerate(loader):
        imgs = imgs.to(device)
        tda = tda.to(device)
        lbls = lbls.to(device)
        
        logits = model(imgs, tda)
        if i % 2 == 0:
            print(f"Batch {i}: shape={imgs.shape}, time={time.time() - start_time:.4f}s")
            
    total_time = time.time() - start_time
    print(f"Finished 100 samples in {total_time:.4f}s")
    print(f"Estimated time for 5778 samples: {total_time * 57.78:.4f}s ({(total_time * 57.78)/60:.2f} minutes)")

if __name__ == "__main__":
    speed_test()
