import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

from feature_extraction.xray.dataset import KneeXrayDataset
from feature_extraction.xray.scaler import XrayTopologicalScaler
from models.xray.classifier import HybridXrayClassifier, coral_loss

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode", "full_training")
    mode_settings = config.get(mode, {})
    for k, v in mode_settings.items():
        config["training"][k] = v
    return config

def compute_class_weights(labels, num_classes=5):
    counts = np.zeros(num_classes)
    unique_classes, unique_counts = np.unique(labels, return_counts=True)
    for c, cnt in zip(unique_classes, unique_counts):
        if c < num_classes:
            counts[c] = cnt
            
    total = len(labels)
    class_weights = np.ones(num_classes, dtype=np.float32)
    num_active_classes = np.sum(counts > 0)
    if num_active_classes > 0 and total > 0:
        for i in range(num_classes):
            if counts[i] > 0:
                class_weights[i] = total / (num_active_classes * counts[i])
                
    return torch.tensor(class_weights, dtype=torch.float32)

def get_weighted_sampler(labels):
    class_weights = compute_class_weights(labels)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def train_model(model_name, fusion_type, ordinal=False, loss_type="cross_entropy", use_sampler=False, eval_test=True, ablation_type=None, backbone_type="resnet50", aug_type=None):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected hardware device: {device}")
    print("\n" + "=" * 50)
    print(f"TRAINING MODEL: {model_name} (Fusion: {fusion_type.upper()}, Ordinal: {ordinal}, Backbone: {backbone_type}, Augmentation: {aug_type})")
    print("=" * 50)
    
    # 1. Load Caches
    train_cache = "data/processed_xray/train_cache.npz"
    val_cache = "data/processed_xray/val_cache.npz"
    test_cache = "data/processed_xray/test_cache.npz"
    
    if not os.path.exists(train_cache):
        raise FileNotFoundError("Cache files not found. Run preprocessing/xray_preprocessing/extract_features.py first.")
        
    # Fit topological scaler on train split only
    train_raw = np.load(train_cache, allow_pickle=True)
    tda_scaler = XrayTopologicalScaler()
    tda_scaler.fit(train_raw["tda_features"])
    os.makedirs("data/processed_xray/scalers", exist_ok=True)
    tda_scaler.save("data/processed_xray/scalers/topological_scaler.npz")
    
    # Datasets
    tda_only = (fusion_type == "tda_only")
    
    # Custom Augmentation Construction
    custom_transform = None
    if aug_type is not None:
        import torchvision.transforms as T
        if aug_type == "none":
            custom_transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug_type == "geometric":
            custom_transform = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=10),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug_type == "intensity":
            custom_transform = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug_type == "all":
            custom_transform = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=10),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    train_ds = KneeXrayDataset(train_cache, tda_scaler=tda_scaler, split="train", transform=custom_transform, tda_only=tda_only)
    val_ds = KneeXrayDataset(val_cache, tda_scaler=tda_scaler, split="val", tda_only=tda_only)
    if eval_test:
        test_ds = KneeXrayDataset(test_cache, tda_scaler=tda_scaler, split="test", tda_only=tda_only)
    else:
        test_ds = None
    
    # DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    pin_mem = True if num_workers > 0 else False
    if use_sampler:
        sampler = get_weighted_sampler(train_ds.labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    if eval_test:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    else:
        test_loader = None
    
    # Compute Class Weights for Loss
    class_weights = None
    if loss_type in ["weighted_cross_entropy", "focal_loss"]:
        class_weights = compute_class_weights(train_ds.labels).to(device)
        print(f"Applying Class-weighted Loss: {class_weights.cpu().numpy()}")
        
    total_tda_dim = train_ds.tda_features.shape[1]
    d_tda = total_tda_dim // 2 if ablation_type in ["h0_only", "h1_only"] else total_tda_dim
    model = HybridXrayClassifier(
        d_tda=d_tda,
        d_tda_emb=config["model"]["topological_embedding_dim"],
        d_vis_emb=config["model"]["visual_embedding_dim"],
        d_fusion=config["model"]["fusion_embedding_dim"],
        num_classes=config["model"]["num_classes"],
        fusion_type=fusion_type,
        ordinal=ordinal,
        backbone_type=backbone_type
    ).to(device)
    
    # Setup Optimization - Stage A (Warmup phase)
    # Freeze visual backbone completely
    if fusion_type != "tda_only":
        for name, param in model.vis_backbone.named_parameters():
            param.requires_grad = False
            
    # Collect only the trainable head / projection parameters
    head_params = [p for p in model.parameters() if p.requires_grad]
        
    optimizer = optim.AdamW(
        head_params, 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Loss selection
    if ordinal:
        criterion = lambda outputs, targets: coral_loss(outputs, targets, num_classes=5)
    else:
        if loss_type == "weighted_cross_entropy":
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "focal_loss":
            criterion = FocalLoss(alpha=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
            
    # Training Loop
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    
    os.makedirs("models/xray", exist_ok=True)
    best_weights_path = f"models/xray/{model_name}.pth"
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    
    for epoch in range(epochs):
        # Stage B Transition: Unfreeze later blocks/stages depending on backbone_type
        if fusion_type != "tda_only" and epoch == warmup_epochs and warmup_epochs > 0:
            print(f"\n[STAGE B] Transitioning to fine-tuning. Unfreezing later blocks of {backbone_type}...")
            for name, param in model.vis_backbone.named_parameters():
                unfreeze = False
                if backbone_type == "resnet50":
                    if "layer3" in name or "layer4" in name:
                        unfreeze = True
                elif backbone_type == "efficientnet":
                    if any(f"features.{i}" in name for i in [6, 7, 8]):
                        unfreeze = True
                elif backbone_type == "convnext":
                    if any(f"features.{i}" in name for i in [5, 6, 7]):
                        unfreeze = True
                if unfreeze:
                    param.requires_grad = True
                    
            # Reconstruct the optimizer with discriminative learning rates
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "vis_backbone" in name:
                        backbone_params.append(param)
                    else:
                        head_params.append(param)
                        
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": config["training"]["backbone_learning_rate"]},
                {"params": head_params, "lr": config["training"]["learning_rate"]}
            ], weight_decay=config["training"]["weight_decay"])
            
            # Re-initialize scheduler for Stage B
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            
        model.train()
        train_loss = 0.0
        train_preds = []
        train_lbls_list = []
        
        for imgs, tda_feats, lbls in train_loader:
            if ablation_type == "h0_only":
                tda_feats = tda_feats[:, :tda_feats.shape[1] // 2]
            elif ablation_type == "h1_only":
                tda_feats = tda_feats[:, tda_feats.shape[1] // 2:]
            imgs = imgs.to(device)
            tda_feats = tda_feats.to(device)
            lbls = lbls.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs, tda_feats)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
            probs = model.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_lbls_list.extend(lbls.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_lbls_list, train_preds)
        
        # Validation epoch
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_lbls = []
        
        with torch.no_grad():
            for imgs, tda_feats, lbls in val_loader:
                if ablation_type == "h0_only":
                    tda_feats = tda_feats[:, :tda_feats.shape[1] // 2]
                elif ablation_type == "h1_only":
                    tda_feats = tda_feats[:, tda_feats.shape[1] // 2:]
                imgs = imgs.to(device)
                tda_feats = tda_feats.to(device)
                lbls = lbls.to(device)
                
                logits = model(imgs, tda_feats)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
                
                probs = model.predict_probabilities(logits)
                preds = torch.argmax(probs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_lbls.extend(lbls.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_lbls, val_preds)
        val_f1 = f1_score(val_lbls, val_preds, average='macro')
        
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2%}) | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | Val F1: {val_f1:.4f}")
        
        # Save best weights (Early stopping check)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)
            patience_counter = 0
            print("  [SAVED] Best validation checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                 print(f"Early stopping triggered after {epoch+1} epochs.")
                 break
                 
    # Save training history to appropriate results subdirectory
    if "temp" in model_name:
        history_dir = "results/development_subset"
    elif "visual_baseline" in model_name:
        history_dir = "results/visual_baselines"
    elif "baseline" in model_name:
        history_dir = "results/full_baseline"
    elif "tda_only" in model_name:
        history_dir = "results/full_tda"
    elif "taff_ordinal" in model_name:
        history_dir = "results/full_ordinal"
    elif "taff" in model_name:
        history_dir = "results/full_taff"
    elif "ablation" in model_name:
        history_dir = "results/ablation"
    else:
        history_dir = "results"
        
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_path}")

    # Plot training curves
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title("Loss Curves")
        
        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.set_title("Accuracy Curves")
        
        plt.tight_layout()
        curves_path = os.path.join(history_dir, f"{model_name}_curves.png")
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Saved training curves to {curves_path}")
    except Exception as e:
        print(f"Could not plot training curves: {e}")
                 
    # 3. Load best weights and evaluate
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
    if not eval_test:
        val_preds_best = []
        val_lbls_best = []
        with torch.no_grad():
            for imgs, tda_feats, lbls in val_loader:
                if ablation_type == "h0_only":
                    tda_feats = tda_feats[:, :tda_feats.shape[1] // 2]
                elif ablation_type == "h1_only":
                    tda_feats = tda_feats[:, tda_feats.shape[1] // 2:]
                imgs = imgs.to(device)
                tda_feats = tda_feats.to(device)
                logits = model(imgs, tda_feats)
                probs = model.predict_probabilities(logits)
                preds = torch.argmax(probs, dim=1)
                val_preds_best.extend(preds.cpu().numpy())
                val_lbls_best.extend(lbls.cpu().numpy())
                
        val_acc_best = accuracy_score(val_lbls_best, val_preds_best)
        val_bal_acc_best = balanced_accuracy_score(val_lbls_best, val_preds_best)
        val_f1_macro_best = f1_score(val_lbls_best, val_preds_best, average='macro')
        val_f1_weighted_best = f1_score(val_lbls_best, val_preds_best, average='weighted')
        val_kappa_best = cohen_kappa_score(val_lbls_best, val_preds_best, weights='quadratic')
        val_mae_best = mean_absolute_error(val_lbls_best, val_preds_best)
        val_cm_best = confusion_matrix(val_lbls_best, val_preds_best, labels=[0, 1, 2, 3, 4])
        
        print(f"\n[EVALUATION] Validation Set (Best Checkpoint) - {model_name}:")
        print(f"  Accuracy: {val_acc_best:.2%}")
        print(f"  Balanced Accuracy: {val_bal_acc_best:.2%}")
        print(f"  Macro F1: {val_f1_macro_best:.4f}")
        print(f"  Weighted F1: {val_f1_weighted_best:.4f}")
        print(f"  Kappa (Quadratic): {val_kappa_best:.4f}")
        print(f"  MAE: {val_mae_best:.4f}")
        print(f"  Confusion Matrix:\n{val_cm_best}")
        
        # Save confusion matrix plot for validation
        os.makedirs("results/confusion_matrices", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        cax = ax.matshow(val_cm_best, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        labels = ["KL0", "KL1", "KL2", "KL3", "KL4"]
        for i in range(val_cm_best.shape[0]):
            for j in range(val_cm_best.shape[1]):
                ax.text(j, i, str(val_cm_best[i, j]), va='center', ha='center', color='black' if val_cm_best[i, j] < val_cm_best.max()/2 else 'white')
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted KL Grade")
        ax.set_ylabel("Actual KL Grade")
        ax.xaxis.set_ticks_position('bottom')
        val_cm_path = os.path.join(history_dir, f"{model_name}_val_cm.png")
        plt.savefig(val_cm_path, dpi=150)
        plt.savefig(f"results/confusion_matrices/{model_name}_val_cm.png", dpi=150)
        plt.close()
        
        return {
            "accuracy": float(val_acc_best),
            "balanced_accuracy": float(val_bal_acc_best),
            "f1_macro": float(val_f1_macro_best),
            "f1_weighted": float(val_f1_weighted_best),
            "cohen_kappa_quadratic": float(val_kappa_best),
            "mae": float(val_mae_best),
            "confusion_matrix": val_cm_best.tolist()
        }
    
    test_preds = []
    test_lbls = []
    test_probs = []
    
    with torch.no_grad():
        for imgs, tda_feats, lbls in test_loader:
            if ablation_type == "h0_only":
                tda_feats = tda_feats[:, :tda_feats.shape[1] // 2]
            elif ablation_type == "h1_only":
                tda_feats = tda_feats[:, tda_feats.shape[1] // 2:]
            imgs = imgs.to(device)
            tda_feats = tda_feats.to(device)
            logits = model(imgs, tda_feats)
            probs = model.predict_probabilities(logits)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_lbls.extend(lbls.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            
    test_acc = accuracy_score(test_lbls, test_preds)
    test_bal_acc = balanced_accuracy_score(test_lbls, test_preds)
    test_f1_macro = f1_score(test_lbls, test_preds, average='macro')
    test_f1_weighted = f1_score(test_lbls, test_preds, average='weighted')
    test_kappa = cohen_kappa_score(test_lbls, test_preds, weights='quadratic')
    test_mae = mean_absolute_error(test_lbls, test_preds)
    cm = confusion_matrix(test_lbls, test_preds, labels=[0, 1, 2, 3, 4])
    
    print(f"\n[EVALUATION]Hold-out Test Set - {model_name}:")
    print(f"  Accuracy: {test_acc:.2%}")
    print(f"  Balanced Accuracy: {test_bal_acc:.2%}")
    print(f"  Macro F1: {test_f1_macro:.4f}")
    print(f"  Weighted F1: {test_f1_weighted:.4f}")
    print(f"  Kappa (Quadratic): {test_kappa:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Save confusion matrix plot
    os.makedirs("results/confusion_matrices", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    labels = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted KL Grade")
    ax.set_ylabel("Actual KL Grade")
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrices/{model_name}_cm.png", dpi=150)
    plt.close()
    
    return {
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal_acc),
        "f1_macro": float(test_f1_macro),
        "f1_weighted": float(test_f1_weighted),
        "cohen_kappa_quadratic": float(test_kappa),
        "mae": float(test_mae),
        "confusion_matrix": cm.tolist()
    }
