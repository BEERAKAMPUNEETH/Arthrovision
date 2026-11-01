import os
import yaml
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from torch.utils.data import Dataset, DataLoader

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class KneeSegmentationDataset(Dataset):
    def __init__(self, df, raw_dir):
        self.df = df
        self.raw_dir = raw_dir
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.raw_dir, row["image_path"])
        mask_path = os.path.join(self.raw_dir, row["mask_path"])
        
        # Load SimpleITK images
        img_sitk = sitk.ReadImage(img_path)
        mask_sitk = sitk.ReadImage(mask_path)
        
        # Get arrays (Z, Y, X) and convert to (C, X, Y, Z)
        img_arr = sitk.GetArrayFromImage(img_sitk).transpose(2, 1, 0)
        mask_arr = sitk.GetArrayFromImage(mask_sitk).transpose(2, 1, 0)
        
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0) # (1, X, Y, Z)
        mask_tensor = torch.tensor(mask_arr, dtype=torch.long) # (X, Y, Z)
        
        return img_tensor, mask_tensor, row["scan_id"]

def compute_dice_iou(pred, target, num_classes=4):
    """
    Computes class-wise Dice and IoU coefficients.
    pred: (B, H, W, D) argmax predictions
    target: (B, H, W, D) ground truth labels
    """
    dices = []
    ious = []
    
    for c in range(1, num_classes): # skip background class 0
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float().item()
        union = (pred_c | target_c).sum().float().item()
        total_sum = pred_c.sum().float().item() + target_c.sum().float().item()
        
        if total_sum > 0:
            dice = (2.0 * intersection) / total_sum
        else:
            dice = 1.0 if target_c.sum() == 0 else 0.0
            
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if target_c.sum() == 0 else 0.0
            
        dices.append(dice)
        ious.append(iou)
        
    return dices, ious

def train_segmentation():
    config = load_config()
    torch.manual_seed(config["seed"])
    
    raw_dir = config["data"]["raw_dir"]
    split_csv = os.path.join(config["data"]["split_dir"], "dataset_splits.csv")
    
    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    train_ds = KneeSegmentationDataset(train_df, raw_dir)
    val_ds = KneeSegmentationDataset(val_df, raw_dir)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # Initialize MONAI 3D U-Net
    device = torch.device(config["training"]["device"])
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1
    ).to(device)
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training MONAI 3D U-Net segmentation model...")
    num_epochs = 4 # small number of epochs for quick synthesis validation
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for step, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # targets needs to be (B, 1, X, Y, Z) for the loss function or similar depending on settings
            loss = loss_function(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Validation metrics
    model.eval()
    all_dices = []
    all_ious = []
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1) # (B, X, Y, Z)
            
            dices, ious = compute_dice_iou(pred, targets)
            all_dices.append(dices)
            all_ious.append(ious)
            
    mean_dices = np.mean(all_dices, axis=0)
    mean_ious = np.mean(all_ious, axis=0)
    
    print("\nMONAI 3D U-Net Validation Metrics:")
    print(f"Femur     - Dice: {mean_dices[0]:.4f}, IoU: {mean_ious[0]:.4f}")
    print(f"Tibia     - Dice: {mean_dices[1]:.4f}, IoU: {mean_ious[1]:.4f}")
    print(f"Cartilage - Dice: {mean_dices[2]:.4f}, IoU: {mean_ious[2]:.4f}")
    
    # Save the model
    os.makedirs(os.path.join(raw_dir, "..", "models"), exist_ok=True)
    model_path = os.path.join(raw_dir, "..", "models", "unet_segmentation.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved segmentation model to {model_path}")

if __name__ == "__main__":
    train_segmentation()
