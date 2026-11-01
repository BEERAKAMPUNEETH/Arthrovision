import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

from feature_extraction.xray.dataset import KneeXrayDataset
from feature_extraction.xray.scaler import XrayTopologicalScaler
from models.xray.classifier import HybridXrayClassifier

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode", "full_training")
    mode_settings = config.get(mode, {})
    for k, v in mode_settings.items():
        config["training"][k] = v
    return config

def run_cross_validation(cv_folds=5):
    print("=" * 60)
    print(f"RUNNING PATIENT-DISJOINT STRATIFIED GROUP {cv_folds}-FOLD CV")
    print("=" * 60)
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware device: {device}")
    
    # 1. Load cached training data
    train_cache = "data/processed_xray/train_cache.npz"
    if not os.path.exists(train_cache):
        raise FileNotFoundError(f"Cache file {train_cache} not found.")
        
    cache = np.load(train_cache, allow_pickle=True)
    image_paths = cache["image_paths"]
    labels = cache["labels"]
    patient_ids = cache["patient_ids"]
    tda_features = cache["tda_features"]
    
    # Apply subset mode limit if active
    if config["training"].get("subset_mode", False):
        max_samples = 100
        if len(image_paths) > max_samples:
            step = len(image_paths) // max_samples
            image_paths = image_paths[::step][:max_samples]
            labels = labels[::step][:max_samples]
            patient_ids = patient_ids[::step][:max_samples]
            tda_features = tda_features[::step][:max_samples]
            
    # Load scaler and transform TDA features
    tda_scaler = XrayTopologicalScaler()
    tda_scaler.load("data/processed_xray/scalers/topological_scaler.npz")
    normalized_tda = tda_scaler.transform(tda_features)
    
    # 2. Extract visual and topological embeddings using fine-tuned models
    best_weights_path = "models/xray/resnet_tda_concat_full.pth"
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Fine-tuned concatenation weights {best_weights_path} not found. Please train concat first.")
        
    print(f"Loading visual and topological projection modules from: {best_weights_path}")
    model = HybridXrayClassifier(
        d_tda=16,
        d_tda_emb=config["model"]["topological_embedding_dim"], # 32
        d_vis_emb=config["model"]["visual_embedding_dim"],       # 128
        d_fusion=config["model"]["fusion_embedding_dim"],        # 64
        num_classes=config["model"]["num_classes"],
        fusion_type="concat",
        ordinal=False
    ).to(device)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    
    print("Extracting pre-trained features (visual 128D + topological 32D) for all samples...")
    dataset = KneeXrayDataset(train_cache, tda_scaler=tda_scaler, split="val")
    if config["training"].get("subset_mode", False):
        dataset.image_paths = image_paths
        dataset.labels = labels
        dataset.patient_ids = patient_ids
        dataset.tda_features = tda_features
        dataset.normalized_tda = normalized_tda
        
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    fused_features = []
    with torch.no_grad():
        for imgs, tda_feats, _ in loader:
            imgs = imgs.to(device)
            tda_feats = tda_feats.to(device)
            
            # Extract 128D visual embedding
            vis_feats = model.vis_backbone(imgs)
            vis_emb = model.vis_proj(vis_feats)
            
            # Extract 32D TDA embedding
            tda_emb = model.tda_embedder(tda_feats)
            
            # Concatenate (128 + 32 = 160D)
            fused = torch.cat([vis_emb, tda_emb], dim=1)
            fused_features.append(fused.cpu().numpy())
            
    fused_features = np.concatenate(fused_features, axis=0)
    print(f"Fused Embeddings Extracted: {fused_features.shape}")
    
    # 3. StratifiedGroupKFold splitting based on patient_ids to prevent split leakage
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=config["training"].get("random_seed", 42))
    
    fold_metrics = []
    
    # StratifiedGroupKFold split
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(fused_features, labels, groups=patient_ids)):
        print(f"\nTraining Fold {fold + 1}/{cv_folds}...")
        
        # Verify patient-disjoint partition constraint
        train_pats = set(patient_ids[train_idx])
        val_pats = set(patient_ids[val_idx])
        overlap = train_pats.intersection(val_pats)
        assert len(overlap) == 0, f"Leakage detected! Patient overlap: {overlap}"
        
        # Tensors
        fold_train_feats = torch.tensor(fused_features[train_idx], dtype=torch.float32)
        fold_train_lbl = torch.tensor(labels[train_idx], dtype=torch.long)
        
        fold_val_feats = torch.tensor(fused_features[val_idx], dtype=torch.float32)
        fold_val_lbl = torch.tensor(labels[val_idx], dtype=torch.long)
        
        # DataLoaders
        train_ds = TensorDataset(fold_train_feats, fold_train_lbl)
        val_ds = TensorDataset(fold_val_feats, fold_val_lbl)
        
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)
        
        # Initialize MLP head classifier
        # Input size is 160D -> 64 -> 5 classes
        clf_head = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)
        ).to(device)
        
        # Train fold classifier using Standard Cross Entropy
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(clf_head.parameters(), lr=config["training"]["learning_rate"])
        
        best_val_loss = float("inf")
        best_preds = None
        best_lbls = None
        
        for epoch in range(10):
            clf_head.train()
            for feats, lbl in train_loader:
                feats, lbl = feats.to(device), lbl.to(device)
                optimizer.zero_grad()
                logits = clf_head(feats)
                loss = criterion(logits, lbl)
                loss.backward()
                optimizer.step()
                
            # Eval
            clf_head.eval()
            val_loss = 0.0
            val_preds = []
            val_lbls = []
            with torch.no_grad():
                for feats, lbl in val_loader:
                    feats, lbl = feats.to(device), lbl.to(device)
                    logits = clf_head(feats)
                    loss = criterion(logits, lbl)
                    val_loss += loss.item() * feats.size(0)
                    
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_lbls.extend(lbl.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds
                best_lbls = val_lbls
                
        # Compute fold metrics
        acc = accuracy_score(best_lbls, best_preds)
        bal_acc = balanced_accuracy_score(best_lbls, best_preds)
        f1_macro = f1_score(best_lbls, best_preds, average='macro')
        kappa = cohen_kappa_score(best_lbls, best_preds, weights='quadratic')
        mae = mean_absolute_error(best_lbls, best_preds)
        
        print(f"  Fold {fold + 1} - Val Loss: {best_val_loss:.4f} | Acc: {acc:.2%} | Macro F1: {f1_macro:.4f} | Kappa: {kappa:.4f}")
        
        fold_metrics.append({
            "Fold": fold + 1,
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "Macro F1": f1_macro,
            "Kappa": kappa,
            "MAE": mae
        })
        
    # 4. Save and report cross-validation metrics
    os.makedirs("results/cross_validation", exist_ok=True)
    df_metrics = pd.DataFrame(fold_metrics)
    df_metrics.to_csv("results/cross_validation/fold_metrics.csv", index=False)
    
    summary = {
        "mean_accuracy": float(df_metrics["Accuracy"].mean()),
        "std_accuracy": float(df_metrics["Accuracy"].std()),
        "mean_balanced_accuracy": float(df_metrics["Balanced Accuracy"].mean()),
        "std_balanced_accuracy": float(df_metrics["Balanced Accuracy"].std()),
        "mean_f1_macro": float(df_metrics["Macro F1"].mean()),
        "std_f1_macro": float(df_metrics["Macro F1"].std()),
        "mean_kappa": float(df_metrics["Kappa"].mean()),
        "std_kappa": float(df_metrics["Kappa"].std()),
        "mean_mae": float(df_metrics["MAE"].mean()),
        "std_mae": float(df_metrics["MAE"].std())
    }
    
    summary_path = "results/cross_validation/cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "=" * 60)
    print("5-FOLD PATIENT-LEVEL CV RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Accuracy:          {summary['mean_accuracy']:.2%} ± {summary['std_accuracy']:.2%}")
    print(f"  Balanced Accuracy: {summary['mean_balanced_accuracy']:.2%} ± {summary['std_balanced_accuracy']:.2%}")
    print(f"  Macro F1:          {summary['mean_f1_macro']:.4f} ± {summary['std_f1_macro']:.4f}")
    print(f"  Quadratic Kappa:   {summary['mean_kappa']:.4f} ± {summary['std_kappa']:.4f}")
    print(f"  MAE:               {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    run_cross_validation()
