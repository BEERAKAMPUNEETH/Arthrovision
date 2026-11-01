import os
import io
import yaml
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import base64
import cv2
import matplotlib.pyplot as plt

from preprocessing.standardizer import ImageStandardizer
from reconstruction.reconstructor import JointReconstructor
from topology.persistent_homology import TopologicalAnalyzer
from feature_extraction.topology import TopologicalFeatureExtractor, TopologicalFeatureScaler
from feature_extraction.image import KneeResNet50
from feature_extraction.clinical import ClinicalFeaturePreprocessor
from models.classifier import HybridKneeClassifier
from explainability.grad_cam import GradCAM, overlay_heatmap

app = FastAPI(title="ArthroVision-TDA Academic Research API", version="1.1.0")

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
device = torch.device(config["training"]["device"])

# Global models dictionary
models = {}

@app.on_event("startup")
def load_models():
    """Load pre-trained parameters and scalers on startup."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    processed_dir = config["data"]["processed_dir"]
    
    # 1. Load fitted scalers to prevent data leakage during inference
    clinical_scaler_path = os.path.join(processed_dir, "scalers", "clinical_preprocessor.pth")
    topo_scaler_path = os.path.join(processed_dir, "scalers", "topological_scaler.pth")
    
    if os.path.exists(clinical_scaler_path):
        models["clinical_preprocessor"] = torch.load(clinical_scaler_path, weights_only=False)
        print("Loaded fitted clinical preprocessor scaler.")
    else:
        # Fallback dummy
        models["clinical_preprocessor"] = ClinicalFeaturePreprocessor()
        dummy_df = pd.DataFrame({"age": [45, 80], "bmi": [18.0, 45.0], "pain_score": [0, 10], "gender": [0, 1]})
        models["clinical_preprocessor"].fit(dummy_df)
        
    if os.path.exists(topo_scaler_path):
        models["topo_scaler"] = torch.load(topo_scaler_path, weights_only=False)
        print("Loaded fitted GUDHI topological scaler.")
    else:
        # Fallback dummy
        models["topo_scaler"] = TopologicalFeatureScaler()
        dummy_topo = np.random.rand(10, 16)
        models["topo_scaler"].fit(dummy_topo)

    # 2. Load fine-tuned ResNet50
    resnet = KneeResNet50(num_classes=5).to(device)
    finetuned_resnet_path = os.path.join(models_dir, "resnet50_finetuned.pth")
    if os.path.exists(finetuned_resnet_path):
        resnet.load_state_dict(torch.load(finetuned_resnet_path, map_location=device))
        print("Loaded fine-tuned ResNet50 visual backbone.")
    resnet.eval()
    models["resnet"] = resnet

    # 3. Load hybrid multimodal classifier (checks if ordinal version is available, fallback to proposed)
    is_ordinal = False
    classifier_path = os.path.join(models_dir, "proposed_ordinal.pth")
    if os.path.exists(classifier_path):
        is_ordinal = True
    else:
        classifier_path = os.path.join(models_dir, "proposed_multimodal.pth")
        
    classifier = HybridKneeClassifier(
        d_vis=2048,
        d_topo=16,
        d_clin=4,
        num_classes=5,
        fusion_type="topology_aware",
        ordinal=is_ordinal
    ).to(device)
    
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        print(f"Loaded multimodal classifier weights from {os.path.basename(classifier_path)} (ordinal={is_ordinal}).")
    classifier.eval()
    models["classifier"] = classifier
    models["is_ordinal"] = is_ordinal

@app.get("/health")
def health_check():
    mode = "DEMO (Synthetic)" if config.get("demo_mode", True) else "RESEARCH (Real OAI)"
    return {"status": "healthy", "device": str(device), "mode": mode}

@app.post("/predict")
async def predict_osteoarthritis(
    image: UploadFile = File(...),
    age: float = Form(...),
    bmi: float = Form(...),
    pain_score: float = Form(...),
    gender: int = Form(...) # 0: Female, 1: Male
):
    try:
        temp_dir = os.path.join(config["data"]["processed_dir"], "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, image.filename)
        
        with open(temp_file_path, "wb") as f:
            f.write(await image.read())
            
        # 1. Read volume via SimpleITK
        standardizer = ImageStandardizer()
        try:
            vol, _ = standardizer.read_volume(temp_file_path)
        except Exception as e:
            # Grayscale fallback
            raw_img = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
            vol = np.stack([raw_img] * 32, axis=-1).astype(np.float32) / 255.0
            
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        vol_preprocessed = standardizer.preprocess_volume(vol)
        
        # Heuristic/mock segmentation for prediction if ground-truth mask is not supplied in request
        # In deployment/UI, we segment the joint space using simple bounding thresholds
        size = vol.shape
        mask = np.zeros(size, dtype=np.uint8)
        mask[vol_preprocessed > 0.6] = 1 # bone structure
        mask[vol_preprocessed > 0.4] = 2 # bone structure
        mask[(vol_preprocessed > 0.2) & (vol_preprocessed <= 0.4)] = 3 # cartilage
        
        # 2. Multi-Slice Visual Feature extraction
        num_slices = config["model"]["visual"].get("num_slices", 5)
        img_inputs, orig_vis = standardizer.extract_visual_slices(vol_preprocessed, mask, k=num_slices)
        
        resnet = models["resnet"]
        slice_features = []
        for img_in in img_inputs:
            img_tensor = torch.tensor(img_in, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, vis_feat = resnet(img_tensor)
            slice_features.append(vis_feat.squeeze().cpu().numpy())
            
        # Mean pooling aggregation
        mean_vis_feat = np.mean(slice_features, axis=0)
        t_vis = torch.tensor(mean_vis_feat, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 3. 3D Surface Reconstruction
        reconstructor = JointReconstructor()
        points, _ = reconstructor.reconstruct_joint_surface(mask)
        
        # 4. GUDHI Persistent Homology
        topo_analyzer = TopologicalAnalyzer()
        simplex_tree, _ = topo_analyzer.compute_persistence(points)
        
        # 5. Extract and normalize topological features
        topo_extractor = TopologicalFeatureExtractor()
        topo_raw = topo_extractor.extract_features(simplex_tree)
        topo_norm = models["topo_scaler"].transform([topo_raw])[0]
        t_topo = torch.tensor(topo_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 6. Normalize clinical features
        clin_df = pd.DataFrame([{
            "age": age, "bmi": bmi, "pain_score": pain_score, "gender": gender
        }])
        clin_norm = models["clinical_preprocessor"].transform(clin_df)[0]
        t_clin = torch.tensor(clin_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 7. Classification Prediction
        classifier = models["classifier"]
        with torch.no_grad():
            logits = classifier(t_vis, t_topo, t_clin)
            probs = classifier.predict_probabilities(logits).squeeze().cpu().numpy()
            predicted_class = int(np.argmax(probs))
            
        # 8. Grad-CAM on central visual slice
        mid_slice_idx = num_slices // 2
        central_slice_input = img_inputs[mid_slice_idx]
        central_slice_tensor = torch.tensor(central_slice_input, dtype=torch.float32).unsqueeze(0).to(device)
        
        grad_cam = GradCAM(resnet, resnet.backbone.layer4)
        heatmap = grad_cam.generate_heatmap(central_slice_tensor, predicted_class)
        grad_cam.remove_hooks()
        
        cam_overlay = overlay_heatmap(heatmap, orig_vis)
        
        # Base64 encodings
        _, buffer_cam = cv2.imencode('.png', cam_overlay)
        base64_cam = base64.b64encode(buffer_cam).decode('utf-8')
        
        _, buffer_orig = cv2.imencode('.png', cv2.cvtColor(orig_vis, cv2.COLOR_GRAY2BGR))
        base64_orig = base64.b64encode(buffer_orig).decode('utf-8')
        
        fig_diag_path = io.BytesIO()
        topo_analyzer.plot_persistence_diagram(simplex_tree, fig_diag_path)
        fig_diag_path.seek(0)
        base64_diag = base64.b64encode(fig_diag_path.read()).decode('utf-8')
        
        fig_bar_path = io.BytesIO()
        topo_analyzer.plot_persistence_barcode(simplex_tree, fig_bar_path)
        fig_bar_path.seek(0)
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
            "clinical_features": clin_norm.tolist(),
            "topological_features": topo_norm.tolist(),
            "original_image_b64": base64_orig,
            "grad_cam_b64": base64_cam,
            "persistence_diagram_b64": base64_diag,
            "persistence_barcode_b64": base64_bar
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
