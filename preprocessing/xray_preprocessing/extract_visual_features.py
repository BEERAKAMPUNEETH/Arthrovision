import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.models as models
from preprocessing.xray_preprocessing.preprocessor import KneeXrayPreprocessor

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        return yaml.safe_load(f)

class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.preprocessor = KneeXrayPreprocessor(target_size=(224, 224), use_clahe=True, use_roi=False)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        file_path = self.image_paths[idx]
        try:
            _, img_rgb = self.preprocessor.preprocess_image(file_path)
        except Exception:
            img_rgb = np.zeros((3, 224, 224), dtype=np.float32)
        return torch.tensor(img_rgb, dtype=torch.float32)

def extract_visual_embeddings():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting visual features on device: {device}")
    
    # Load ResNet50 model pretrained on ImageNet
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    # Remove final FC layer
    resnet.fc = nn = torch.nn.Identity()
    resnet.eval()
    
    splits = ["train", "val", "test"]
    for split in splits:
        cache_path = f"data/processed_xray/{split}_cache.npz"
        if not os.path.exists(cache_path):
            print(f"Cache file {cache_path} not found.")
            continue
            
        print(f"\nProcessing split: {split.upper()}")
        data = np.load(cache_path, allow_pickle=True)
        image_paths = data["image_paths"]
        
        dataset = ImageOnlyDataset(image_paths)
        # Use num_workers=0 on windows to avoid multiprocessing errors
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Extracting"):
                batch = batch.to(device)
                feats = resnet(batch)
                embeddings.append(feats.cpu().numpy())
                
        embeddings = np.concatenate(embeddings, axis=0)
        
        # Save updated cache with visual features
        np.savez_compressed(
            cache_path,
            image_paths=image_paths,
            labels=data["labels"],
            patient_ids=data["patient_ids"],
            lateralities=data["lateralities"],
            tda_features=data["tda_features"],
            visual_features=embeddings.astype(np.float32)
        )
        print(f"Updated cache for split {split.upper()} with visual features of shape {embeddings.shape}")

if __name__ == "__main__":
    extract_visual_embeddings()
