import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)


def get_metrics(model, X_val, y_val):
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "accuracy": accuracy_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, proba),
        "preds": preds,
        "proba": proba,
        "report": classification_report(y_val, preds, target_names=["Genuine", "Hallucinated"])
    }


def print_results(name, metrics):
    print(f"\n{'='*50}")
    print(f"  {name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['report']}")


def compare_models(lgbm_metrics, cat_metrics):
    print(f"\n{'='*55}")
    print(f"{'Metric':<15} {'LightGBM':>15} {'CatBoost':>15}")
    print(f"{'-'*55}")
    for key, label in [("accuracy", "Accuracy"), ("f1", "F1 Score"), ("roc_auc", "ROC-AUC")]:
        l = lgbm_metrics[key]
        c = cat_metrics[key]
        winner = " <-- BETTER" if l >= c else "            <-- BETTER"
        print(f"{label:<15} {l:>15.4f} {c:>15.4f}{winner if l != c else ''}")
    print(f"{'='*55}")
    winner = "LightGBM" if lgbm_metrics["roc_auc"] >= cat_metrics["roc_auc"] else "CatBoost"
    print(f"\n  Best Model (ROC-AUC): {winner}")
