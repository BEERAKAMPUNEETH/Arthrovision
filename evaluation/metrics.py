import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score,
    cohen_kappa_score,
    mean_absolute_error
)

def compute_class_wise_sensitivity_specificity(y_true, y_pred, num_classes=5):
    """
    Computes sensitivity (Recall) and specificity for each class individually.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    sensitivities = []
    specificities = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
    return sensitivities, specificities

def evaluate_predictions(y_true, y_pred, y_probs=None, num_classes=5):
    """
    Evaluates classification predictions using multiple medical imaging and ordinal metrics.
    """
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, specs = compute_class_wise_sensitivity_specificity(y_true, y_pred, num_classes)
    
    # Ordinal specific metrics
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "cohen_kappa_quadratic": float(kappa),
        "mae": float(mae),
        "confusion_matrix": cm.tolist(),
        "class_wise_sensitivity": [float(s) for s in sens],
        "class_wise_specificity": [float(sp) for sp in specs]
    }
    
    if y_probs is not None:
        y_probs = np.array(y_probs)
        try:
            # Multi-class ROC-AUC (One-vs-Rest)
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", labels=list(range(num_classes)))
            metrics["auc_ovr"] = float(auc)
        except Exception:
            metrics["auc_ovr"] = 0.0
            
    return metrics
