import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torchvision import transforms

from preprocessing.standardizer import ImageStandardizer
from reconstruction.reconstructor import JointReconstructor
from topology.persistent_homology import TopologicalAnalyzer
from feature_extraction.topology import TopologicalFeatureExtractor, TopologicalFeatureScaler
from feature_extraction.image import KneeResNet50
from feature_extraction.clinical import ClinicalFeaturePreprocessor
from models.classifier import HybridKneeClassifier, OrdinalLoss
from evaluation.metrics import evaluate_predictions

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def verify_data_splits(df):
    """
    Asserts strict patient-level splitting to prevent data leakage,
    displays split statistics, and outputs warnings for underrepresented classes.
    """
    print("\n================ DATASET SPLIT INTEGRITY CHECK ================")
    
    splits = ["train", "val", "test"]
    patient_sets = {}
    
    for s in splits:
        split_df = df[df["split"] == s]
        patient_sets[s] = set(split_df["patient_id"].unique())
        
        # Calculate scan and patient counts
        n_pats = len(patient_sets[s])
        n_scans = len(split_df)
        print(f"\nSplit '{s}': {n_pats} Patients, {n_scans} Scans")
        
        # KL distribution
        kl_counts = split_df["kl_grade"].value_counts().sort_index()
        total_scans = len(split_df)
        print("KL Grade Distribution:")
        for grade in range(5):
            count = kl_counts.get(grade, 0)
            pct = (count / total_scans) * 100 if total_scans > 0 else 0
            print(f"  KL {grade}: {count} ({pct:.1f}%)")
            
            # Check representation
            if count == 0:
                print(f"  [WARNING] KL {grade} is MISSING from '{s}' split!")
            elif pct < 10.0:
                print(f"  [WARNING] KL {grade} is underrepresented (<10%) in '{s}' split ({pct:.1f}%)")

    # Data Leakage checks (overlapping patients across splits)
    overlap_train_val = patient_sets["train"].intersection(patient_sets["val"])
    overlap_train_test = patient_sets["train"].intersection(patient_sets["test"])
    overlap_val_test = patient_sets["val"].intersection(patient_sets["test"])
    
    if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
        print("\n  [CRITICAL WARNING] DATA LEAKAGE DETECTED!")
        print(f"  Overlapping patients between Train & Val: {overlap_train_val}")
        print(f"  Overlapping patients between Train & Test: {overlap_train_test}")
        print(f"  Overlapping patients between Val & Test: {overlap_val_test}")
        raise ValueError("Data leakage: patient IDs overlap between dataset splits!")
    else:
        print("\n[SUCCESS] No patient-level data leakage detected across splits.")
    print("===============================================================\n")

class KneeSliceDataset(Dataset):
    """
    Dataset of individual preprocessed 2D slices for ResNet50 pre-training.
    """
    def __init__(self, slices, labels, transform=None):
        self.slices = slices # numpy array (N, 3, 224, 224)
        self.labels = labels # list/array of scan labels
        self.transform = transform
        
    def __len__(self):
        return len(self.slices)
        
    def __getitem__(self, idx):
        slice_img = self.slices[idx]
        label = self.labels[idx]
        
        # Convert to torch tensor if not already
        slice_tensor = torch.tensor(slice_img, dtype=torch.float32)
        
        # Apply training augmentations
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
            
        return slice_tensor, torch.tensor(label, dtype=torch.long)

def fine_tune_resnet_on_slices(config, df, raw_dir, device):
    """
    Fine-tunes the ResNet50 visual branch (layer4 and fc_head) on preprocessed 2D knee slices,
    using train-set class-weighted loss and early stopping.
    """
    print("\n--- Starting ResNet50 Visual Branch Fine-Tuning Stage ---")
    
    standardizer = ImageStandardizer()
    num_slices = config["model"]["visual"].get("num_slices", 5)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    def gather_slices(split_df):
        all_slices = []
        all_labels = []
        for idx, row in split_df.iterrows():
            img_path = os.path.join(raw_dir, row["image_path"])
            mask_path = os.path.join(raw_dir, row["mask_path"])
            
            # Load volume and segmentation mask
            vol, _ = standardizer.read_volume(img_path)
            vol_pre = standardizer.preprocess_volume(vol)
            sitk_mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(sitk_mask).transpose(2, 1, 0)
            
            # Extract slices
            slices, _ = standardizer.extract_visual_slices(vol_pre, mask_arr, k=num_slices)
            for s in slices:
                all_slices.append(s)
                all_labels.append(row["kl_grade"])
        return np.array(all_slices, dtype=np.float32), np.array(all_labels, dtype=np.int64)
        
    print("Extracting training slice images...")
    X_train_slices, y_train_slices = gather_slices(train_df)
    print("Extracting validation slice images...")
    X_val_slices, y_val_slices = gather_slices(val_df)
    
    # Calculate inverse frequency class weights to address class imbalance
    class_counts = np.bincount(y_train_slices, minlength=5)
    total_samples = len(y_train_slices)
    class_weights = total_samples / (5.0 * (class_counts + 1e-5))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Computed training split class weights: {class_weights}")
    
    # Training augmentations (rotation, scaling, translation)
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ])
    
    train_ds = KneeSliceDataset(X_train_slices, y_train_slices, transform=train_transform)
    val_ds = KneeSliceDataset(X_val_slices, y_val_slices)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    # Instantiate resnet
    resnet = KneeResNet50(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Helper function for training epochs under different unfreezing configurations
    def run_fine_tune_epochs(model, optimizer, start_epoch, num_epochs):
        best_loss = float("inf")
        best_state = None
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits, _ = model(inputs)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
                    
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            actual_epoch = start_epoch + epoch + 1
            print(f"Fine-tuning ResNet Epoch {actual_epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return best_state, best_loss

    # Staged Fine-Tuning Strategy:
    # Stage 1: Freeze all backbone layers and train classification head
    print("Stage 1: Training classifier projection head (backbone frozen)")
    for name, param in resnet.backbone.named_parameters():
        param.requires_grad = False
    optimizer_s1 = torch.optim.AdamW(resnet.fc_head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state_s1, _ = run_fine_tune_epochs(resnet, optimizer_s1, start_epoch=0, num_epochs=2)
    if best_state_s1 is not None:
        resnet.load_state_dict({k: v.to(device) for k, v in best_state_s1.items()})

    # Stage 2: Unfreeze layer4
    print("Stage 2: Fine-tuning layer4 (lower learning rate)")
    for name, param in resnet.backbone.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
    optimizer_s2 = torch.optim.AdamW([p for p in resnet.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    best_state_s2, _ = run_fine_tune_epochs(resnet, optimizer_s2, start_epoch=2, num_epochs=2)
    if best_state_s2 is not None:
        resnet.load_state_dict({k: v.to(device) for k, v in best_state_s2.items()})

    # Stage 3: Unfreeze layer3 & layer4
    print("Stage 3: Fine-tuning layer3 & layer4 (very low learning rate)")
    for name, param in resnet.backbone.named_parameters():
        if "layer3" in name or "layer4" in name:
            param.requires_grad = True
    optimizer_s3 = torch.optim.AdamW([p for p in resnet.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-4)
    best_state_s3, _ = run_fine_tune_epochs(resnet, optimizer_s3, start_epoch=4, num_epochs=2)
    if best_state_s3 is not None:
        resnet.load_state_dict({k: v.to(device) for k, v in best_state_s3.items()})
        
    # Save the fine-tuned model
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(resnet.state_dict(), os.path.join(models_dir, "resnet50_finetuned.pth"))
    print("Fine-tuning completed. Saved resnet50_finetuned.pth weights.")
    return resnet


def extract_and_cache_features(config):
    """
    Extracts multi-slice visual features, normalized GUDHI topological vectors,
    and standardized clinical features split-by-split.
    """
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    split_csv = os.path.join(config["data"]["split_dir"], "dataset_splits.csv")
    
    os.makedirs(processed_dir, exist_ok=True)
    df = pd.read_csv(split_csv)
    
    # 1. Verification of Split Leakage and Class Representation
    verify_data_splits(df)
    
    device = torch.device(config["training"]["device"])
    
    # 2. ResNet fine-tuning stage
    resnet = fine_tune_resnet_on_slices(config, df, raw_dir, device)
    resnet.eval()
    
    # Initialize Preprocessors and Scalers
    standardizer = ImageStandardizer()
    reconstructor = JointReconstructor()
    topo_analyzer = TopologicalAnalyzer()
    topo_extractor = TopologicalFeatureExtractor()
    
    # Clinical Preprocessor: Fit only on training split
    train_df = df[df["split"] == "train"]
    clinical_preprocessor = ClinicalFeaturePreprocessor()
    clinical_preprocessor.fit(train_df)
    
    num_slices = config["model"]["visual"].get("num_slices", 5)
    
    # Gather train topological features to fit the scaler
    print("Calculating training topological scale parameters...")
    train_topo_list = []
    for idx, row in train_df.iterrows():
        mask_path = os.path.join(raw_dir, row["mask_path"])
        sitk_mask = sitk.ReadImage(mask_path)
        mask_arr = sitk.GetArrayFromImage(sitk_mask).transpose(2, 1, 0)
        points, _ = reconstructor.reconstruct_joint_surface(mask_arr)
        simplex_tree, _ = topo_analyzer.compute_persistence(points)
        topo_feat = topo_extractor.extract_features(simplex_tree)
        train_topo_list.append(topo_feat)
        
    topo_scaler = TopologicalFeatureScaler()
    topo_scaler.fit(train_topo_list)
    
    # Save the clinical and topological scalers for runtime FastAPI/Streamlit prediction
    os.makedirs(os.path.join(processed_dir, "scalers"), exist_ok=True)
    torch.save(clinical_preprocessor, os.path.join(processed_dir, "scalers", "clinical_preprocessor.pth"))
    torch.save(topo_scaler, os.path.join(processed_dir, "scalers", "topological_scaler.pth"))
    
    # Extract split by split
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        print(f"\nCaching features for the {split} split ({len(split_df)} scans)...")
        
        vis_feats_list = []
        topo_feats_list = []
        clin_feats = clinical_preprocessor.transform(split_df)
        labels = split_df["kl_grade"].values
        scan_ids = split_df["scan_id"].values
        
        for idx, row in split_df.reset_index(drop=True).iterrows():
            img_path = os.path.join(raw_dir, row["image_path"])
            mask_path = os.path.join(raw_dir, row["mask_path"])
            
            # Read volume and segmentation mask
            vol, _ = standardizer.read_volume(img_path)
            vol_preprocessed = standardizer.preprocess_volume(vol)
            sitk_mask = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(sitk_mask).transpose(2, 1, 0)
            
            # 1. Multi-Slice Visual Feature aggregation (Mean Pooling)
            img_inputs, _ = standardizer.extract_visual_slices(vol_preprocessed, mask_arr, k=num_slices)
            
            # Pass all K slices through ResNet50
            slice_features = []
            for img_in in img_inputs:
                img_tensor = torch.tensor(img_in, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, vis_feat = resnet(img_tensor)
                slice_features.append(vis_feat.squeeze().cpu().numpy())
                
            # Aggregate slices using mean pooling
            mean_visual_feat = np.mean(slice_features, axis=0)
            vis_feats_list.append(mean_visual_feat)
            
            # 2. Point cloud geometry reconstruction
            points, _ = reconstructor.reconstruct_joint_surface(mask_arr)
            
            # 3. Persistent Homology
            simplex_tree, _ = topo_analyzer.compute_persistence(points)
            
            # 4. GUDHI Descriptors & Scaler normalization
            topo_raw = topo_extractor.extract_features(simplex_tree)
            topo_feats_list.append(topo_raw)
            
            if (idx + 1) % 5 == 0 or (idx + 1) == len(split_df):
                print(f"Processed {idx + 1}/{len(split_df)} scans")
                
        # Scale topological features
        topo_feats_scaled = topo_scaler.transform(topo_feats_list)
        
        npz_path = os.path.join(processed_dir, f"{split}_features.npz")
        np.savez(
            npz_path,
            vis_feats=np.array(vis_feats_list, dtype=np.float32),
            topo_feats=np.array(topo_feats_scaled, dtype=np.float32),
            clin_feats=np.array(clin_feats, dtype=np.float32),
            labels=labels,
            scan_ids=scan_ids
        )
        print(f"Saved cached feature matrices to {npz_path}")

def train_classifier(config, model_config, train_npz, val_npz):
    """
    Trains a classification model using pre-cached features, supporting
    inverse class weighting, WeightedRandomSampler, and Ordinal CORAL regression.
    """
    device = torch.device(config["training"]["device"])
    
    # Load cached features
    train_data = np.load(train_npz)
    val_data = np.load(val_npz)
    
    # Unpack tensors
    X_train_vis = torch.tensor(train_data["vis_feats"]).to(device)
    X_train_topo = torch.tensor(train_data["topo_feats"]).to(device)
    X_train_clin = torch.tensor(train_data["clin_feats"]).to(device)
    labels_train = train_data["labels"]
    y_train = torch.tensor(labels_train, dtype=torch.long).to(device)
    
    X_val_vis = torch.tensor(val_data["vis_feats"]).to(device)
    X_val_topo = torch.tensor(val_data["topo_feats"]).to(device)
    X_val_clin = torch.tensor(val_data["clin_feats"]).to(device)
    y_val = torch.tensor(val_data["labels"], dtype=torch.long).to(device)
    
    # Setup class weights (for standard softmax baseline)
    class_counts = np.bincount(labels_train, minlength=5)
    total_samples = len(labels_train)
    class_weights = total_samples / (5.0 * (class_counts + 1e-5))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Setup WeightedRandomSampler to balance batches during training
    # Probability of picking a sample is proportional to its class weight
    sample_weights = [class_weights[label] for label in labels_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=total_samples, 
        replacement=True
    )
    
    # Build batch loader of indices
    # Using batches to match PyTorch training structures
    batch_size = config["training"]["batch_size"]
    train_indices_loader = DataLoader(
        range(total_samples), 
        batch_size=batch_size, 
        sampler=sampler
    )
    
    is_ordinal = model_config.get("ordinal", False)
    
    # Initialize hybrid model
    model = HybridKneeClassifier(
        d_vis=X_train_vis.shape[1],
        d_topo=X_train_topo.shape[1],
        d_clin=X_train_clin.shape[1],
        num_classes=5,
        fusion_type=model_config.get("fusion_type", "topology_aware"),
        embed_dim=config["model"]["fusion"]["embed_dim"],
        dropout=config["model"]["classification"]["dropout"],
        use_visual=model_config.get("use_visual", True),
        use_topological=model_config.get("use_topological", True),
        use_clinical=model_config.get("use_clinical", True),
        ordinal=is_ordinal
    ).to(device)
    
    # Select loss function
    if is_ordinal:
        criterion = OrdinalLoss(num_classes=5)
    else:
        # standard weighted cross-entropy
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=1e-4)
    
    epochs = config["training"]["epochs"]
    best_val_f1 = -1.0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Iterate using the WeightedRandomSampler index batches
        for indices in train_indices_loader:
            optimizer.zero_grad()
            
            b_vis = X_train_vis[indices] if model_config.get("use_visual", True) else None
            b_topo = X_train_topo[indices] if model_config.get("use_topological", True) else None
            b_clin = X_train_clin[indices] if model_config.get("use_clinical", True) else None
            b_y = y_train[indices]
            
            logits = model(b_vis, b_topo, b_clin)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation early stopping check
        model.eval()
        with torch.no_grad():
            v_vis = X_val_vis if model_config.get("use_visual", True) else None
            v_topo = X_val_topo if model_config.get("use_topological", True) else None
            v_clin = X_val_clin if model_config.get("use_clinical", True) else None
            
            val_logits = model(v_vis, v_topo, v_clin)
            
            # Predict and evaluate
            val_probs = model.predict_probabilities(val_logits).cpu().numpy()
            val_preds = np.argmax(val_probs, axis=1)
            
            metrics = evaluate_predictions(y_val.cpu().numpy(), val_preds, val_probs)
            val_f1 = metrics["f1_macro"]
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    return model

if __name__ == "__main__":
    cfg = load_config()
    extract_and_cache_features(cfg)
