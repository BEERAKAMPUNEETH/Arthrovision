import os
import numpy as np
from preprocessing.xray_preprocessing.preprocessor import KneeXrayPreprocessor
from topology.xray.persistent_homology import XrayTopologicalAnalyzer

def test_xray_modules():
    print("=" * 60)
    print("TESTING 2D X-RAY PREPROCESSING & TDA EXTRACTOR")
    print("=" * 60)
    
    dataset_dir = "C:/Users/punee/Downloads/archive/KneeXrayMini"
    # Find first image in train/0
    class_0_dir = os.path.join(dataset_dir, "train", "0")
    if not os.path.exists(class_0_dir):
        print(f"Error: {class_0_dir} does not exist.")
        return
        
    filenames = os.listdir(class_0_dir)
    if len(filenames) == 0:
        print(f"Error: No files found in {class_0_dir}.")
        return
        
    sample_path = os.path.join(class_0_dir, filenames[0])
    print(f"Loading sample image: {sample_path}")
    
    # 1. Test Preprocessor
    preprocessor = KneeXrayPreprocessor(target_size=(224, 224), use_clahe=True, use_roi=True)
    img_gray, img_rgb = preprocessor.preprocess_image(sample_path)
    
    print(f"Preprocessed Grayscale Shape: {img_gray.shape}, dtype: {img_gray.dtype}, Range: [{img_gray.min()}, {img_gray.max()}]")
    print(f"Preprocessed RGB Shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
    
    assert img_gray.shape == (224, 224), "Grayscale image shape should be (224, 224)"
    assert img_rgb.shape == (3, 224, 224), "RGB image shape should be (3, 224, 224)"
    print("[PASSED] Preprocessor")
    
    # 2. Test TDA Edge (Alpha Complex)
    tda = XrayTopologicalAnalyzer()
    points = tda.extract_point_cloud_from_edges(img_gray)
    print(f"Extracted edge point cloud shape: {points.shape}")
    assert len(points) > 0, "Should extract edge points"
    
    simplex_tree, persistence = tda.compute_persistence_alpha(points)
    features_alpha = tda.extract_topological_features(simplex_tree, is_cubical=False)
    print(f"Alpha topological features vector shape: {features_alpha.shape}")
    print(f"Alpha features summary: H0 mean birth = {features_alpha[5]:.4f}, H1 mean birth = {features_alpha[13]:.4f}")
    assert len(features_alpha) == 16, "Should yield a 16-dimensional feature vector"
    
    # 3. Test TDA Cubical Complex
    cubical_complex, persistence_cub = tda.compute_persistence_cubical(img_gray)
    features_cub = tda.extract_topological_features(cubical_complex, is_cubical=True)
    print(f"Cubical topological features vector shape: {features_cub.shape}")
    print(f"Cubical features summary: H0 mean birth = {features_cub[5]:.4f}, H1 mean birth = {features_cub[13]:.4f}")
    assert len(features_cub) == 16, "Should yield a 16-dimensional feature vector"
    print("[PASSED] Topological Analyzer")
    
if __name__ == "__main__":
    test_xray_modules()
