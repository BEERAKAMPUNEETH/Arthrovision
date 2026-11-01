import os
import yaml
import torch
import numpy as np
import cv2
import base64
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from preprocessing.xray_preprocessing.preprocessor import KneeXrayPreprocessor
from topology.xray.persistent_homology import XrayTopologicalAnalyzer
from feature_extraction.xray.scaler import XrayTopologicalScaler
from models.xray.classifier import HybridXrayClassifier
from explainability.grad_cam import GradCAM, overlay_heatmap

app = FastAPI(title="ArthroVision-TDA 2D X-Ray API Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_config():
    with open("configs/config_xray.yaml", "r") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode", "full_training")
    mode_settings = config.get(mode, {})
    for k, v in mode_settings.items():
        config["training"][k] = v
    return config

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load topological scaler
scaler_path = "data/processed_xray/scalers/topological_scaler.npz"
tda_scaler = XrayTopologicalScaler()
if os.path.exists(scaler_path):
    tda_scaler.load(scaler_path)
    tda_dim = len(tda_scaler.mean)
else:
    # Initialize a default scaler in case API is started before training
    tda_scaler.mean = np.zeros(16)
    tda_scaler.std = np.ones(16)
    tda_dim = 16

# Load hybrid classifier model (Winner Model: ResNet50 + Late Probability Fusion)
weights_path = "models/xray/resnet50_fusion_late.pth"
fusion_type = "late"
if not os.path.exists(weights_path):
    weights_path = "models/xray/resnet50_fusion_gated.pth"
    fusion_type = "gated"
    if not os.path.exists(weights_path):
        weights_path = "models/xray/resnet50_visual_tda_concat.pth"
        fusion_type = "concat"

model = HybridXrayClassifier(
    d_tda=tda_dim,
    d_tda_emb=config["model"]["topological_embedding_dim"],
    d_vis_emb=config["model"]["visual_embedding_dim"],
    d_fusion=config["model"]["fusion_embedding_dim"],
    num_classes=config["model"]["num_classes"],
    fusion_type=fusion_type,
    ordinal=False,
    backbone_type="resnet50"
).to(device)

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"[SUCCESS] Loaded fine-tuned winner hybrid model weights from {weights_path} (Fusion: {fusion_type}).")
else:
    print(f"[WARNING] Fine-tuned model weights not found at {weights_path}. Using initialized model.")
model.eval()

@app.get("/health")
def health_check():
    mode = "DEMO (Synthetic)" if config.get("demo_mode", True) else "REAL-DATA RESEARCH"
    return {
        "status": "healthy",
        "device": str(device),
        "modality": "2D X-ray",
        "mode": mode,
        "clinical_modality": "disabled (no matched clinical data available)"
    }

@app.post("/predict")
async def predict_xray(
    image: UploadFile = File(...)
):
    try:
        temp_dir = "data/processed_xray/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, image.filename)
        
        with open(temp_file_path, "wb") as f:
            f.write(await image.read())
            
        # 1. Preprocess X-ray
        preprocessor = KneeXrayPreprocessor(
            target_size=tuple(config["preprocessing"]["target_size"]),
            use_clahe=config["preprocessing"]["use_clahe"],
            use_roi=config["preprocessing"]["use_roi"]
        )
        
        try:
            img_gray, img_rgb = preprocessor.preprocess_image(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not preprocess image: {e}")
            
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        # 2. Extract topological features
        tda = XrayTopologicalAnalyzer(
            max_points=config["tda"]["max_points"],
            downsample_size=tuple(config["tda"]["downsample_size"])
        )
        
        rep_type = config["tda"]["representation"]
        tda_mode = "expanded" if tda_dim == 32 else "standard"
        if rep_type == "alpha":
            points = tda.extract_point_cloud_from_edges(img_gray)
            complex_obj, _ = tda.compute_persistence_alpha(points)
            tda_raw = tda.extract_topological_features(complex_obj, is_cubical=False, mode=tda_mode)
        else:
            complex_obj, _ = tda.compute_persistence_cubical(img_gray)
            tda_raw = tda.extract_topological_features(complex_obj, is_cubical=True, mode=tda_mode)
            
        # Normalize TDA features
        tda_norm = tda_scaler.transform([tda_raw])[0]
        
        # 3. Model Inference
        # Convert to tensors
        t_img = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0).to(device)
        t_tda = torch.tensor(tda_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(t_img, t_tda)
            probs = model.predict_probabilities(logits).squeeze(0).cpu().numpy()
            predicted_class = int(np.argmax(probs))
            
        # 4. Explainable AI: Grad-CAM
        backbone_name = model.vis_backbone.model_name
        if backbone_name in ["convnext", "efficientnet"]:
            target_layer = model.vis_backbone.backbone.features[-1]
        else:
            target_layer = model.vis_backbone.backbone.layer4

        grad_cam = GradCAM(model, target_layer)
        # Pass visual and topological inputs as tuple
        heatmap = grad_cam.generate_heatmap((t_img, t_tda), predicted_class)
        grad_cam.remove_hooks()
        
        cam_overlay = overlay_heatmap(heatmap, img_gray)
        
        # 5. Persistence plots
        fig_diag_path = io.BytesIO()
        tda.plot_persistence_diagram(complex_obj, fig_diag_path, is_cubical=(rep_type == "cubical"))
        fig_diag_path.seek(0)
        
        fig_bar_path = io.BytesIO()
        tda.plot_persistence_barcode(complex_obj, fig_bar_path, is_cubical=(rep_type == "cubical"))
        fig_bar_path.seek(0)
        
        # Encode outputs as base64
        _, buffer_cam = cv2.imencode('.png', cam_overlay)
        base64_cam = base64.b64encode(buffer_cam).decode('utf-8')
        
        _, buffer_orig = cv2.imencode('.png', cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR))
        base64_orig = base64.b64encode(buffer_orig).decode('utf-8')
        
        base64_diag = base64.b64encode(fig_diag_path.read()).decode('utf-8')
        base64_bar = base64.b64encode(fig_bar_path.read()).decode('utf-8')
        
        kl_grades_mapping = {
            0: "Healthy (KL 0)",
            1: "Doubtful (KL 1)",
            2: "Mild (KL 2)",
            3: "Moderate (KL 3)",
            4: "Severe (KL 4)"
        }
        
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "predicted_grade": kl_grades_mapping[predicted_class],
            "probabilities": probs.tolist(),
            "topological_features": tda_norm.tolist(),
            "original_image_b64": base64_orig,
            "grad_cam_b64": base64_cam,
            "persistence_diagram_b64": base64_diag,
            "persistence_barcode_b64": base64_bar,
            "clinical_modality_disabled": True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
