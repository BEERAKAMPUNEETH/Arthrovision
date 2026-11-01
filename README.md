# ArthroVision-TDA: Topology-Aware Knee Osteoarthritis Severity Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

> **A Multimodal Deep Learning Framework Integrating Deep Visual Backbones (ResNet-50) and Persistent Homology Descriptors for Kellgren-Lawrence Grade (KL 0–4) Classification.**

---

## 📌 Abstract & Overview

Knee Osteoarthritis (KOA) diagnosis relies heavily on plain radiography to assess Kellgren-Lawrence (KL) grades (0: Normal, 1: Doubtful, 2: Minimal, 3: Moderate, 4: Severe). Conventional convolutional networks often struggle with early-stage structural subtleties (such as KL1 doubtful joint space narrowing or early osteophyte formation) and exhibit poor sensitivity to geometric tissue deformities.

**ArthroVision-TDA** introduces a hybrid topology-aware deep learning framework:
1. **Visual Feature Extraction**: Extract high-level spatial and texture features using a fine-tuned **ResNet-50** deep convolutional backbone.
2. **Topological Data Analysis (TDA)**: Extract multi-scale geometric shape signatures using **Persistent Homology** (via GUDHI Alpha and Cubical complexes), quantifying connected components ($H_0$) and 1D topological loops ($H_1$) such as trabecular bone loop architectures, subchondral sclerosis, and articular boundary margins.
3. **Late Probability Fusion**: Combine deep visual representations and persistent topological representations to achieve calibrated severity predictions.
4. **Rigorous Experimental Validation**: Validated across an automated 18-phase master research pipeline encompassing patient-disjoint 5-fold cross-validation, ablation studies, and exact statistical significance testing (McNemar test $p = 2.08 \times 10^{-30}$).

---

## 🔬 Key Experimental Results (Locked Test Set: 1,656 Images)

The finalized model (**Late Probability Fusion with ResNet-50 Backbone & Standard Alpha TDA**) was evaluated once on the strictly locked hold-out test set:

| Evaluation Metric | Baseline (ResNet-50 Visual Only) | ArthroVision-TDA (Proposed Winner) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 58.84% | **59.72%** |
| **Balanced Accuracy** | 54.67% | **54.26%** |
| **Macro F1-Score** | 0.5121 | **0.5172** |
| **Quadratic Cohen's $\kappa$** | 0.6943 | **0.7041** |
| **Mean Absolute Error (MAE)** | 0.5738 | **0.5586** |
| **Statistical Significance** | — | **$p = 2.0854 \times 10^{-30}$** (Significant) |

### 5-Fold Stratified Group Cross-Validation (Phase 12)
To ensure zero patient data leakage across train and validation distributions, stratified patient-disjoint 5-fold cross validation was conducted:
- **Fold 1 Accuracy**: 50.48% (Quadratic $\kappa$: 0.5442)
- **Fold 2 Accuracy**: 49.39% (Quadratic $\kappa$: 0.5019)
- **Fold 3 Accuracy**: 48.79% (Quadratic $\kappa$: 0.4830)
- **Fold 4 Accuracy**: 47.22% (Quadratic $\kappa$: 0.4821)
- **Fold 5 Accuracy**: 47.22% (Quadratic $\kappa$: 0.4595)
- **Mean Cross-Validation Accuracy**: **48.62%**

---

## 🏗️ System Architecture

```text
               +--------------------------------------+
               |      Digital Knee Radiograph         |
               +-------------------+------------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
+--------------------------+               +----------------------------------+
|   Visual Deep Branch     |               |     Topological Analysis Branch  |
|      (ResNet-50)         |               |             (GUDHI)              |
+--------------------------+               +----------------------------------+
| - Two-stage fine-tuning  |               | - Edge & point cloud extraction  |
| - Pretrained ImageNet    |               | - Alpha / Cubical filtration     |
| - Visual embedding (2048)|               | - H0 & H1 Persistent Homology    |
+------------+-------------+               | - Lifetimes, Entropy, Percentiles|
             |                             +-----------------+----------------+
             v                                               v
    [Visual Probabilities]                       [Topological Probabilities]
             \                                               /
              \                                             /
               v                                           v
         +-------------------------------------------------------+
         |        Late Probability Fusion Architecture           |
         |         P_final = alpha * P_vis + (1 - alpha) * P_tda |
         +---------------------------+---------------------------+
                                     |
                                     v
                 +---------------------------------------+
                 |    Calibrated Severity Prediction     |
                 |       Kellgren-Lawrence Grade [0 - 4] |
                 +---------------------------------------+
```

---

## 📁 Repository Structure

```text
Arthrovision/
├── api/
│   ├── main_xray.py            # High-performance FastAPI prediction backend
│   └── main.py                 # Core API endpoints & health checks
├── app/
│   └── main.py                 # Academic Streamlit UI portal with Grad-CAM & Barcode visualizers
├── configs/
│   ├── config_xray.yaml        # Main configuration: hyperparameters, model paths, training modes
│   └── config.yaml             # Generic framework configuration
├── data/
│   ├── processed/              # Processed experimental metadata and summary JSONs
│   ├── processed_xray/         # Topological feature scalers & metadata
│   └── splits/
│       └── dataset_splits.csv  # Patient-disjoint dataset partitions (Train / Val / Test)
├── evaluation/
│   └── metrics.py              # Multiclass, ordinal, balanced accuracy, and kappa metrics
├── explainability/
│   └── grad_cam.py             # Native-resolution Grad-CAM visual explanation generator
├── feature_extraction/
│   ├── image.py                # ResNet-50 visual feature extractor
│   ├── topology.py             # Persistent entropy, Betti numbers, and diagram encoders
│   └── xray/                   # X-ray dataset loaders and scaling pipelines
├── fusion/
│   └── multi_modal_fusion.py   # Late Probability Fusion, Gated MLP, and TAFF attention layers
├── models/
│   ├── classifier.py           # General hybrid classifier modules
│   └── xray/
│       └── classifier.py       # ResNet-50 hybrid classifier implementations
├── preprocessing/
│   └── xray_preprocessing/     # CLAHE enhancement, edge filtering, and ROI extractors
├── results/
│   ├── confusion_matrices/     # Comprehensive validation & test confusion matrices
│   ├── paper_generation/       # Publication-ready LaTeX tables (tables.tex)
│   ├── visual_baselines/       # Baseline curves, metrics, and comparisons
│   └── statistical_test.json   # McNemar exact binomial significance test output
├── tests/
│   ├── test_pipeline.py        # Pipeline regression tests
│   └── test_xray_pipeline.py   # End-to-end inference and feature extraction tests
├── tools/
│   ├── predict_single.py       # CLI tool for single-image inference
│   └── xray_dataset_analysis/  # Auditing scripts for dataset leakage, speed, and splits
├── topology/
│   └── persistent_homology.py  # GUDHI-based filtration & persistence extraction
├── training/
│   └── xray/
│       ├── orchestrate_research.py # Master 18-phase automated research orchestration pipeline
│       ├── run_ablations.py        # H0 vs H1 homology ablation runs
│       ├── run_final_cross_val.py  # 5-fold cross validation runner
│       └── evaluate_on_test_and_significance.py # Hold-out evaluation & statistical testing
├── Dockerfile                  # Containerized deployment specification
├── requirements.txt            # Pinned environment dependencies
└── README.md                   # Comprehensive framework documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/BEERAKAMPUNEETH/Arthrovision.git
cd Arthrovision

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Running the Interactive Web Portal

The interactive web portal provides a dual-service architecture:

**Step 1: Start the FastAPI Backend**
```bash
python -m uvicorn api.main_xray:app --host localhost --port 8000
```
*Health check:* Visit `http://localhost:8000/health` to confirm the backend is running.

**Step 2: Launch the Streamlit Frontend**
```bash
python -m streamlit run app/main.py --server.port 8501
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to:
- Upload any standard digital knee X-ray (`.png`, `.jpg`).
- Generate real-time KL severity predictions with class probability distributions.
- View interactive **Grad-CAM heatmaps** overlaid directly onto the radiograph.
- Inspect **Persistence Diagrams** and **Topological Barcodes** ($H_0$ and $H_1$).
- Review all experimental benchmarks, cross-validation metrics, and confusion matrices.

---

## 🧪 Reproducing Research Experiments (Master Orchestrator)

The repository includes an automated master orchestration script that executes the complete 18-phase experimental protocol:

```bash
python -m training.xray.orchestrate_research
```

The pipeline automatically handles:
- **Phase 2**: Class imbalance interventions (Cross-Entropy vs Focal Loss vs Weighted vs Balanced Sampler).
- **Phase 3**: Data augmentation strategies (None vs Geometric vs Intensity vs Combined).
- **Phase 4**: Preprocessing optimization (CLAHE contrast vs ROI joint cropping).
- **Phase 5**: Visual baseline vs TDA Concatenation vs TDA-only MLP comparisons.
- **Phase 6 & 7**: Topological filtration search (Alpha complex vs Cubical complex vs Expanded 32D features).
- **Phase 8**: Homology dimension ablation ($H_0$-only vs $H_1$-only vs Combined).
- **Phase 9**: Multimodal fusion selection (Concatenation vs Gated Fusion vs Late Probability vs TAFF).
- **Phase 10**: Classification objective search (Cross-Entropy vs CORAL Ordinal Sigmoidal loss).
- **Phase 11**: KL1 doubtful grade precision/recall and row-normalized audit.
- **Phase 12**: Patient-disjoint 5-fold cross-validation with automated checkpoint-resume.
- **Phase 13**: Binomial exact McNemar statistical significance test.
- **Phase 14 & 15**: Explainability visualizations and adjacent-grade error matrices.
- **Phase 17**: Single evaluation on the locked hold-out test set (1,656 images).
- **Phase 18**: Export of publication-ready LaTeX tables (`results/paper_generation/tables.tex`).

---

## 📊 Explainability & Visualizations

- **Grad-CAM Saliency**: Overlays gradient-weighted class activations onto the joint line, highlighting subchondral sclerosis and narrow joint spaces.
- **Topological Barcodes & Diagrams**: Visualizes the birth and death of $H_0$ connected components (bone density clusters) and $H_1$ persistent loops (trabecular micro-architecture gaps).

---

## 🩺 Academic & Research Disclaimer

> **Notice**: ArthroVision-TDA is an experimental academic research system developed for algorithmic investigation into topological data analysis in medical imaging. The predictions and analyses generated are strictly for research and benchmarking purposes and do not constitute clinical diagnoses or medical advice.

---

## 📄 Citation & License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
