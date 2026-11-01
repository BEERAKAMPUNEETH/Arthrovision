import os
import numpy as np
import torch
import SimpleITK as sitk

from preprocessing.data_generator import generate_synthetic_3d_knee
from preprocessing.standardizer import ImageStandardizer
from reconstruction.reconstructor import JointReconstructor
from topology.persistent_homology import TopologicalAnalyzer
from feature_extraction.topology import TopologicalFeatureExtractor, TopologicalFeatureScaler
from fusion.multi_modal_fusion import SimpleConcatFusion, TopologyAwareFusion
from models.classifier import HybridKneeClassifier, OrdinalLoss

def test_data_generator_output():
    """Verify synthetic 3D volume shapes and label ranges."""
    img, mask = generate_synthetic_3d_knee(kl_grade=2, size=(32, 32, 16))
    
    assert img.shape == (32, 32, 16)
    assert mask.shape == (32, 32, 16)
    assert np.all(img >= 0.0) and np.all(img <= 1.0)
    assert set(np.unique(mask)).issubset({0, 1, 2, 3})

def test_image_standardizer():
    """Verify multi-slice joint ROI extraction and resizing."""
    standardizer = ImageStandardizer(target_img_size=(64, 64))
    
    # Create test 3D volume & mock cartilage mask
    vol = np.random.rand(32, 32, 16).astype(np.float32)
    mask = np.zeros((32, 32, 16), dtype=np.uint8)
    mask[8:16, 8:16, 4:12] = 3 # cartilage
    
    vol_preproc = standardizer.preprocess_volume(vol)
    assert vol_preproc.shape == (32, 32, 16)
    
    # Extract K=3 slices
    img_inputs, orig_vis = standardizer.extract_visual_slices(vol_preproc, mask, k=3)
    
    # img_inputs shape: (K, 3, H, W) -> (3, 3, 64, 64)
    assert img_inputs.shape == (3, 3, 64, 64)
    assert orig_vis.shape == (32, 32)
    assert orig_vis.dtype == np.uint8

def test_joint_reconstruction():
    """Verify Marching Cubes and Open3D sample extraction."""
    mask = np.zeros((32, 32, 16), dtype=np.uint8)
    mask[16:, :, :] = 1 # simulated femur
    mask[:8, :, :] = 2  # simulated tibia
    mask[8:16, :, :] = 3 # simulated cartilage
    
    reconstructor = JointReconstructor(step_size=1)
    points, mesh = reconstructor.reconstruct_joint_surface(mask)
    
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3 # x, y, z coordinates
    assert len(mesh.vertices) > 0

def test_persistent_homology_and_features():
    """Verify GUDHI filtration and 16-dimensional topological feature vector extraction and scaling."""
    # Generate random points representing a hollow sphere surface
    theta = np.random.uniform(0, 2*np.pi, 100)
    phi = np.random.uniform(0, np.pi, 100)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1) * 10.0 + 15.0
    
    topo_analyzer = TopologicalAnalyzer(max_alpha_square=150.0)
    simplex_tree, persistence = topo_analyzer.compute_persistence(points)
    
    assert len(persistence) > 0
    
    topo_extractor = TopologicalFeatureExtractor()
    features = topo_extractor.extract_features(simplex_tree)
    
    assert isinstance(features, np.ndarray)
    assert features.shape == (16,)
    
    # Test topological scaler
    scaler = TopologicalFeatureScaler()
    dummy_feats = np.random.rand(5, 16)
    scaler.fit(dummy_feats)
    scaled = scaler.transform([features])
    assert scaled.shape == (1, 16)

def test_multimodal_fusion():
    """Verify simple concat and attention-based fusion shape logic."""
    f_vis = torch.randn(4, 256)
    f_topo = torch.randn(4, 16)
    f_clin = torch.randn(4, 4)
    
    concat_fusion = SimpleConcatFusion(d_vis=256, d_topo=16, d_clin=4)
    out_concat = concat_fusion(f_vis, f_topo, f_clin)
    assert out_concat.shape == (4, 276)
    
    taff_fusion = TopologyAwareFusion(d_vis=256, d_topo=16, d_clin=4, d_embed=32, num_heads=2)
    out_taff = taff_fusion(f_vis, f_topo, f_clin)
    assert out_taff.shape == (4, 32)

def test_hybrid_classifier_and_ordinal():
    """Verify standard and CORAL ordinal classification outputs and losses."""
    f_vis = torch.randn(2, 2048)
    f_topo = torch.randn(2, 16)
    f_clin = torch.randn(2, 4)
    
    # 1. Standard classification
    model_std = HybridKneeClassifier(
        d_vis=2048, d_topo=16, d_clin=4, num_classes=5, fusion_type="topology_aware", ordinal=False
    )
    logits_std = model_std(f_vis, f_topo, f_clin)
    assert logits_std.shape == (2, 5)
    probs_std = model_std.predict_probabilities(logits_std)
    assert probs_std.shape == (2, 5)
    assert torch.allclose(probs_std.sum(dim=-1), torch.ones(2))
    
    # 2. Ordinal classification
    model_ord = HybridKneeClassifier(
        d_vis=2048, d_topo=16, d_clin=4, num_classes=5, fusion_type="topology_aware", ordinal=True
    )
    logits_ord = model_ord(f_vis, f_topo, f_clin)
    assert logits_ord.shape == (2, 4) # K - 1 = 4 cutpoints
    probs_ord = model_ord.predict_probabilities(logits_ord)
    assert probs_ord.shape == (2, 5)
    assert torch.allclose(probs_ord.sum(dim=-1), torch.ones(2))
    
    # Test ordinal loss
    criterion = OrdinalLoss(num_classes=5)
    targets = torch.tensor([1, 4], dtype=torch.long)
    loss = criterion(logits_ord, targets)
    assert loss.dim() == 0 # Scalar loss
