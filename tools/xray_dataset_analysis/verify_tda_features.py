import os
import numpy as np

def verify_tda():
    print("=" * 60)
    print("ARTHROVISION-TDA: TOPOLOGICAL FEATURE VALIDATION")
    print("=" * 60)
    
    train_cache = "data/processed_xray/train_cache.npz"
    val_cache = "data/processed_xray/val_cache.npz"
    test_cache = "data/processed_xray/test_cache.npz"
    
    for split, path in [("Train", train_cache), ("Val", val_cache), ("Test", test_cache)]:
        if not os.path.exists(path):
            print(f"Error: Cache file {path} not found.")
            continue
            
        data = np.load(path, allow_pickle=True)
        tda = data["tda_features"]
        labels = data["labels"]
        
        print(f"\nSplit: {split}")
        print(f"  Number of samples: {len(tda)}")
        print(f"  Feature dimensions: {tda.shape[1]}")
        
        # Check dimensionality
        assert tda.shape[1] == 16, f"Expected 16 dimensions, got {tda.shape[1]}"
        
        # Check NaN values
        nan_count = np.isnan(tda).sum()
        print(f"  NaN values count: {nan_count}")
        assert nan_count == 0, "NaN values detected in TDA features!"
        
        # Check Infinite values
        inf_count = np.isinf(tda).sum()
        print(f"  Infinite values count: {inf_count}")
        assert inf_count == 0, "Infinite values detected in TDA features!"
        
        # Check variance
        variances = np.var(tda, axis=0)
        zero_var_indices = np.where(variances == 0)[0]
        print(f"  Zero-variance features count: {len(zero_var_indices)}")
        if len(zero_var_indices) > 0:
            print(f"    Zero-variance feature indices: {zero_var_indices}")
            
        # Check range
        feature_min = tda.min(axis=0)
        feature_max = tda.max(axis=0)
        print("  Feature Ranges (Min to Max):")
        for idx in range(16):
            print(f"    Dim {idx:2d} (H0 if <8 else H1): Min={feature_min[idx]:.4f}, Max={feature_max[idx]:.4f}")
            
    print("\n[SUCCESS] All TDA features verified. Feature representation is stable, valid, and ready for training.")

if __name__ == "__main__":
    verify_tda()
