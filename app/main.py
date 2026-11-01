import os
import yaml
import json
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import requests
import base64
import io
import torch
import matplotlib.pyplot as plt

# Config & Page Setup
st.set_page_config(
    page_title="ArthroVision-TDA Academic Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config_xray.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    mode = config.get("mode", "full_training")
    mode_settings = config.get(mode, {})
    for k, v in mode_settings.items():
        config["training"][k] = v
    return config

config = load_config()

# Restrained Academic Custom CSS (White/Light Background, Charcoal Texts, Times/Georgia Fonts)
st.markdown("""
<style>
    .reportview-container { background-color: #ffffff; }
    .main .block-container { padding-top: 1.5rem; }
    body, p, div { color: #1f2937; font-family: 'Times New Roman', Georgia, serif; font-size: 1.05rem; }
    h1 { color: #0f172a; font-family: 'Times New Roman', Georgia, serif; font-weight: bold; font-size: 2.1rem; border-bottom: 2px solid #0f172a; padding-bottom: 0.4rem; }
    h2 { color: #1e293b; font-family: 'Times New Roman', Georgia, serif; font-weight: bold; font-size: 1.5rem; margin-top: 1.5rem; border-bottom: 1px solid #cbd5e1; padding-bottom: 0.2rem; }
    h3 { color: #334155; font-family: 'Times New Roman', Georgia, serif; font-weight: bold; font-size: 1.2rem; margin-top: 1.1rem; }
    
    .stButton>button { background-color: #0f172a; color: white; border-radius: 0px; font-weight: bold; border: 1px solid #0f172a; padding: 0.5rem 1.5rem; }
    .stButton>button:hover { background-color: #ffffff; color: #0f172a; border: 1px solid #0f172a; }
    
    .academic-disclaimer {
        background-color: #f8fafc;
        padding: 0.8rem;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #475569;
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    
    .demo-notice {
        background-color: #fffbeb;
        padding: 0.8rem;
        border: 1px solid #fef3c7;
        border-left: 4px solid #d97706;
        color: #b45309;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .research-notice {
        background-color: #f0fdf4;
        padding: 0.8rem;
        border: 1px solid #dcfce7;
        border-left: 4px solid #16a34a;
        color: #15803d;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: bold;
        color: #0f172a;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("ArthroVision-TDA: Topology-Aware Knee Severity Grading (2D X-Ray)")
st.markdown("""
*Multimodal deep learning framework integrating visual features and topological descriptors for Kellgren-Lawrence grade classification.*
""")

# Research Disclaimer (Requirement 10)
st.markdown("""
<div class="academic-disclaimer">
    <strong>Research prototype disclaimer:</strong> predictions are for experimental/research purposes and are not clinical diagnoses.
</div>
""", unsafe_allow_html=True)

# Mode Notification
subset_active = config["training"].get("subset_mode", False)
if subset_active:
    st.markdown("""
    <div class="demo-notice">
        <strong>[SUBSET DEVELOPMENT MODE ACTIVE]</strong> Model running on a representative subset of the KneeXrayMini dataset for rapid CPU testing and verification.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="research-notice">
        <strong>[MODE: REAL-DATA RESEARCH MODE]</strong> Running on full KneeXrayMini dataset.
    </div>
    """, unsafe_allow_html=True)

# Sidebar UI
st.sidebar.header("Clinical Modality")
st.sidebar.warning("Clinical modality is disabled because matched clinical data were not available for the current X-ray cohort.")

st.sidebar.header("Medical Image Input")
st.sidebar.caption("Primary modality: 2D Knee X-Ray (.png/.jpg).")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray File", type=["png", "jpg", "jpeg"])

# Baseline results loading
results_path = "results/experimental_results_xray.json"

# Initialize session state for prediction results persistence
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
if "predicted_class" not in st.session_state:
    st.session_state["predicted_class"] = None
if "predicted_grade" not in st.session_state:
    st.session_state["predicted_grade"] = None
if "probs" not in st.session_state:
    st.session_state["probs"] = None
if "f_topo" not in st.session_state:
    st.session_state["f_topo"] = None
if "orig_arr" not in st.session_state:
    st.session_state["orig_arr"] = None
if "cam_overlay" not in st.session_state:
    st.session_state["cam_overlay"] = None
if "diag_bytes" not in st.session_state:
    st.session_state["diag_bytes"] = None
if "bar_bytes" not in st.session_state:
    st.session_state["bar_bytes"] = None
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

# Clear results if a new file is uploaded
if uploaded_file != st.session_state["last_uploaded_file"]:
    st.session_state["prediction_done"] = False
    st.session_state["last_uploaded_file"] = uploaded_file

# THREE MAIN TABS
tab1, tab2, tab3 = st.tabs(["Prediction", "Experimental Evaluation", "Research Details"])

with tab1:
    if uploaded_file is None:
        st.info("Upload a Knee X-Ray image in the sidebar to execute prediction.")
    else:
        if st.button("Run Prediction"):
            with st.spinner("Processing Knee X-Ray and extracting topological descriptors..."):
                img_bytes = uploaded_file.read()
                filename = uploaded_file.name
                
                try:
                    files = {"image": (filename, img_bytes)}
                    response = requests.post("http://localhost:8000/predict", files=files, timeout=20)
                    
                    if response.status_code == 200:
                        res = response.json()
                        st.session_state["predicted_class"] = res["predicted_class"]
                        st.session_state["predicted_grade"] = res["predicted_grade"]
                        st.session_state["probs"] = np.array(res["probabilities"])
                        st.session_state["f_topo"] = np.array(res["topological_features"])
                        
                        st.session_state["orig_arr"] = base64.b64decode(res["original_image_b64"])
                        st.session_state["cam_overlay"] = base64.b64decode(res["grad_cam_b64"])
                        st.session_state["diag_bytes"] = base64.b64decode(res["persistence_diagram_b64"])
                        st.session_state["bar_bytes"] = base64.b64decode(res["persistence_barcode_b64"])
                        st.session_state["prediction_done"] = True
                    else:
                        st.error(f"Error from FastAPI backend: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to the API backend at http://localhost:8000. Ensure uvicorn api.main_xray:app is running. Details: {e}")

        # Render outputs if prediction has been performed
        if st.session_state["prediction_done"]:
            st.subheader("Severity Prediction")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                kl_descriptions = {
                    0: "KL 0 — Healthy",
                    1: "KL 1 — Doubtful",
                    2: "KL 2 — Mild",
                    3: "KL 3 — Moderate",
                    4: "KL 4 — Severe"
                }
                confidence_pct = st.session_state["probs"][st.session_state["predicted_class"]] * 100
                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; padding: 1.2rem; margin-bottom:1rem;">
                    <span style="font-size:0.9rem; text-transform:uppercase; color:#64748b; font-weight:bold;">Predicted KL Grade</span><br>
                    <span style="font-size:1.8rem; font-weight:bold; color:#0f172a;">{kl_descriptions[st.session_state["predicted_class"]]}</span><br><br>
                    <span style="font-size:0.9rem; text-transform:uppercase; color:#64748b; font-weight:bold;">PREDICTED-CLASS PROBABILITY</span><br>
                    <span style="font-size:1.8rem; font-weight:bold; color:#0f172a;">{confidence_pct:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Predicted-class probability represents the model's output probability and should not be interpreted as clinical certainty.")

            with col_res2:
                # Compact Horizontal Bar Chart
                probs_df = pd.DataFrame({
                    "KL Grade": ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"],
                    "Probability": st.session_state["probs"]
                })
                st.write("**Model Probability Distribution:**")
                st.bar_chart(probs_df.set_index("KL Grade"))
                
            st.divider()

            # Explainable AI Panels
            st.subheader("Explainable AI & Topological Outputs")
            col_x1, col_x2, col_x3 = st.columns(3)
            
            with col_x1:
                st.write("**A. Preprocessed X-Ray**")
                if st.session_state["orig_arr"] is not None:
                    st.image(st.session_state["orig_arr"], use_container_width=True)
                
            with col_x2:
                st.write("**B. ResNet50 Grad-CAM**")
                if st.session_state["cam_overlay"] is not None:
                    st.image(st.session_state["cam_overlay"], use_container_width=True)
                st.caption("Regions contributing to the visual backbone prediction.")
                
            with col_x3:
                st.write("**C. GUDHI Persistence Diagram**")
                if st.session_state["diag_bytes"] is not None:
                    st.image(st.session_state["diag_bytes"], use_container_width=True)
                st.caption("Topological features extracted via 2D Persistent Homology.")
                
            st.divider()
            
            # Topological Feature Details
            st.subheader("Topological Analysis")
            col_topo1, col_topo2 = st.columns([2, 1])
            with col_topo1:
                st.write("**Persistence Barcode**")
                if st.session_state["bar_bytes"] is not None:
                    st.image(st.session_state["bar_bytes"], width=520)
            with col_topo2:
                st.write("**Feature Vector Summary:**")
                st.caption("Standardized shape features representing persistent loops and components.")
                st.write(f"- **H0 Connected Components**: `{', '.join([f'{x:.2f}' for x in st.session_state['f_topo'][:8]])}`")
                st.write(f"- **H1 One-Dimensional Loops**: `{', '.join([f'{x:.2f}' for x in st.session_state['f_topo'][8:]])}`")
                
                with st.expander("View 16-Dimensional TDA Vector"):
                    st.json(st.session_state["f_topo"].tolist())

with tab2:
    st.subheader("Experimental Evaluation")
    st.markdown("""
    *Quantitative evaluation of the proposed multimodal architecture and its component contributions on the 2D Knee X-Ray dataset.*
    """)
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            outcomes = json.load(f)
            
        experiments = outcomes.get("experiments", {})
        cv_summary = outcomes.get("cross_validation", {})
        stats_val = outcomes.get("statistical_validation", {})
        
        # Primary Performance metric cards (Hold-out/Test set)
        st.write("### HOLD-OUT / TEST SET PERFORMANCE")
        st.caption("Performance metrics evaluated on the independent hold-out test split.")
        proposed_model_key = "4. Proposed ResNet50 + TDA + TAFF"
        if proposed_model_key in experiments:
            metrics_prop = experiments[proposed_model_key]
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Accuracy</div>
                    <div class="metric-val">{metrics_prop['accuracy']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Macro F1</div>
                    <div class="metric-val">{metrics_prop['f1_macro']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Quadratic Kappa</div>
                    <div class="metric-val">{metrics_prop['cohen_kappa_quadratic']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">MAE</div>
                    <div class="metric-val">{metrics_prop['mae']:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                
        st.write("---")
        
        # 5-Fold Cross Validation Performance
        if cv_summary:
            st.write("### 5-FOLD PATIENT-LEVEL CROSS-VALIDATION PERFORMANCE")
            st.caption("Mean performance metrics computed across patient-level cross-validation folds.")
            
            col_cv1, col_cv2, col_cv3, col_cv4 = st.columns(4)
            with col_cv1:
                sd_pp = cv_summary['accuracy_std'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">ACCURACY</div>
                    <div class="metric-val">{cv_summary['accuracy_mean']:.2%} ± {sd_pp:.2f} pp</div>
                </div>
                """, unsafe_allow_html=True)
            with col_cv2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">MACRO F1</div>
                    <div class="metric-val">{cv_summary['f1_macro_mean']:.4f} ± {cv_summary['f1_macro_std']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_cv3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">COHEN'S KAPPA</div>
                    <div class="metric-val">{cv_summary['kappa_mean']:.4f} ± {cv_summary['kappa_std']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_cv4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">MAE</div>
                    <div class="metric-val">{cv_summary['mae_mean']:.4f} ± {cv_summary['mae_std']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                
        st.write("---")

        # Baseline Comparison Matrix
        st.write("### Baseline Comparison Matrix")
        records = []
        for name, metrics in experiments.items():
            records.append({
                "Model Configuration": name,
                "Accuracy": f"{metrics['accuracy']:.2%}",
                "Balanced Acc.": f"{metrics['balanced_accuracy']:.2%}",
                "Macro F1": f"{metrics['f1_macro']:.4f}",
                "Weighted F1": f"{metrics['f1_weighted']:.4f}",
                "Quadratic Cohen's Kappa": f"{metrics['cohen_kappa_quadratic']:.4f}",
                "MAE": f"{metrics['mae']:.4f}"
            })
        results_df = pd.DataFrame(records)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Graphical Confusion Matrix
        st.write("### Confusion Matrix (Proposed Model)")
        if proposed_model_key in experiments:
            cm = np.array(experiments[proposed_model_key]["confusion_matrix"])
            # The CM size might be 4 or 5 depending on subset classes present in test split
            num_classes = cm.shape[0]
            labels = ["KL0", "KL1", "KL2", "KL3", "KL4"][:num_classes]
            
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
                    
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels(labels)
            ax.set_yticks(range(num_classes))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted KL Grade")
            ax.set_ylabel("Actual KL Grade")
            ax.xaxis.set_ticks_position('bottom')
            st.pyplot(fig)
            st.caption("Diagonal = correct predictions | Off-diagonal = misclassifications")
            
        # Statistical Analysis Expandable Dropdown
        if stats_val:
            with st.expander("Statistical Significance Analysis ▸"):
                st.markdown(f"""
                * **Proposed Model vs ResNet50 Baseline**: t-statistic = `{stats_val['resnet_vs_proposed']['t_statistic']:.4f}`, p-value = `{stats_val['resnet_vs_proposed']['p_value']:.4f}`
                * **Proposed Model vs Simple Concatenation**: t-statistic = `{stats_val['concat_vs_proposed']['t_statistic']:.4f}`, p-value = `{stats_val['concat_vs_proposed']['p_value']:.4f}`
                """)
                p_val = stats_val['resnet_vs_proposed']['p_value']
                if p_val < 0.05:
                    st.success(f"✔️ The improvement of the Proposed Model over the Image-only baseline is statistically significant (p = {p_val:.4f} < 0.05 under alpha=0.05).")
                else:
                    st.warning(f"⚠️ The performance difference is not statistically significant (p = {p_val:.4f} >= 0.05).")
    else:
        st.info("Run `python -m training.xray.run_experiments` in your terminal to cache baseline metrics.")

with tab3:
    st.subheader("Research Details & Methodology")
    
    with st.expander("A. System Architecture Diagram"):
        st.graphviz_chart("""
        digraph G {
            rankdir=LR;
            node [shape=box, fontname="Times-Roman", fontsize=10, style=filled, fillcolor="#f8fafc", color="#475569"];
            edge [fontname="Times-Roman", fontsize=9, color="#64748b"];
            
            subgraph cluster_visual {
                label = "Visual Branch";
                fontname = "Times-Bold";
                fontsize = 10;
                style = dashed;
                color = "#94a3b8";
                
                m_img [label="Knee X-Ray Input"];
                img_pre [label="CLAHE + Resize (224x224)"];
                resnet [label="ResNet50 Backbone"];
                vis_feats [label="Visual Features (128)"];
                
                m_img -> img_pre -> resnet -> vis_feats;
            }
            
            subgraph cluster_topo {
                label = "Topological Branch";
                fontname = "Times-Bold";
                fontsize = 10;
                style = dashed;
                color = "#94a3b8";
                
                canny [label="Canny Edge Detection"];
                gudhi [label="Alpha Complex Filtration"];
                topo_feats [label="Topological Features (16)"];
                
                m_img -> canny -> gudhi -> topo_feats;
            }
            
            taff [label="TAFF Self-Attention Fusion", fillcolor="#cbd5e1"];
            clf_head [label="Classification Head (FC / CORAL)"];
            pred [label="KL Grade Prediction (KL0-KL4)", fillcolor="#e2e8f0"];
            
            vis_feats -> taff;
            topo_feats -> taff;
            
            taff -> clf_head -> pred;
        }
        """)
        
    with st.expander("B. Image Preprocessing Pipeline"):
        st.markdown("""
        The input 2D Knee X-Ray is preprocessed using the following sequence:
        `Read Image (Grayscale)` → `CLAHE Enhancement` → `Standard Resize (224x224)` → `ImageNet Normalization`.
        * **CLAHE (Contrast Limited Adaptive Histogram Equalization)** improves local contrast around the joint space while limiting noise amplification.
        """)
        
    with st.expander("C. GUDHI Persistent Homology Methodology (2D X-Ray)"):
        st.markdown("""
        **Persistent Homology** tracks topological features across spatial resolutions (filtrations).
        * **Alpha Complex**: Built on 2D boundary points extracted from edge-maps (Canny filtering). Tracks $H_0$ components (bone structures) and $H_1$ loops (joint space spaces).
        * **Cubical Complex**: Built on downsampled intensity grid ($64 \times 64$). Tracks intensity fluctuations across gray levels.
        """)
        
    with st.expander("D. Multimodal Fusion Details (TAFF)"):
        st.markdown("""
        Complementary information from visual and topological branches are mapped to a joint embedding space using projection layers:
        * **Visual features** capture micro-texture, osteophytes, and gross bone shapes.
        * **Topological features** represent shape invariants of the joint space boundaries.
        
        The **Topology-Aware Feature Fusion (TAFF)** projects these embeddings to a shared 64D token representation and computes cross-attention tokens using Multi-Head Self-Attention layers.
        """)
        
    with st.expander("E. Model Training Configuration"):
        st.markdown("""
        | Training Parameter | Configured Value |
        | :--- | :--- |
        | **Optimizer** | AdamW |
        | **Initial Learning Rate** | 0.001 |
        | **Batch Size** | 16 |
        | **Max Epochs** | 2 (Subset mode) |
        | **Loss Function** | Inverse-Frequency Weighted Cross-Entropy / CORAL cumulative BCE |
        | **Regularization** | Dropout 0.3 + Weight Decay 1e-4 |
        | **Seed** | 42 |
        """)
        
    with st.expander("F. Dataset Specifications"):
        st.markdown("""
        * **Dataset Name**: KneeXrayMini
        * **Total Scans**: 8,260
        * **Train split**: 5,778 scans (2,889 patients)
        * **Validation split**: 826 scans (413 patients)
        * **Test split**: 1,656 scans (828 patients)
        * **Split Protocol**: Patient-level (strictly patient-disjoint splits to prevent leakage)
        """)
        
    with st.expander("G. Target Severity Class Distribution"):
        # Display the distribution count in KneeXrayMini train split
        fig, ax = plt.subplots(figsize=(5, 3))
        grades = ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"]
        counts = [2286, 1046, 1516, 737, 193]  # train split numbers from check_dataset.py
        ax.bar(grades, counts, color="#475569")
        ax.set_ylabel("Scans Count")
        st.pyplot(fig)
        st.caption("Distribution of scan samples across Kellgren-Lawrence grades in train split.")
        
    with st.expander("H. Evaluation Metrics Definitions"):
        st.markdown("""
        * **Accuracy**: Fraction of correctly classified samples.
        * **Balanced Accuracy**: Average recall across classes, giving each class equal importance.
        * **Macro F1**: Average F1 score across classes with equal class weighting.
        * **Weighted F1**: F1 score weighted according to class frequency.
        * **Quadratic Cohen's Kappa**: Agreement between predictions and ground truth considering the ordinal distance between grades.
        * **MAE**: Average absolute difference between predicted and actual KL grade.
        """)
