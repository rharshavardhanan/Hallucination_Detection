import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import OUTPUT_DIR
import os

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_class_distribution(y):
    values = [sum(y == 0), sum(y == 1)]
    labels = ['Genuine (0)', 'Hallucinated (1)']
    colors = ['#4C72B0', '#DD8452']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title("Class Distribution in Training Data", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()
    print("Saved: class_distribution.png")


def plot_answer_length_distribution(train_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    genuine = train_df[train_df['Target'] == 0]['answer_len']
    hallucinated = train_df[train_df['Target'] == 1]['answer_len']
    ax.hist(genuine.clip(upper=2000), bins=60, alpha=0.6, color='#4C72B0', label='Genuine')
    ax.hist(hallucinated.clip(upper=2000), bins=60, alpha=0.6, color='#DD8452', label='Hallucinated')
    ax.set_title("Answer Length Distribution by Class", fontsize=14, fontweight='bold')
    ax.set_xlabel("Character Length (clipped at 2000)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "answer_length_distribution.png"))
    plt.close()
    print("Saved: answer_length_distribution.png")


def plot_confusion_matrices(y_val, lgbm_preds, cat_preds):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, preds, title, cmap in zip(
        axes,
        [lgbm_preds, cat_preds],
        ["LightGBM", "CatBoost"],
        ["Blues", "Oranges"]
    ):
        cm = confusion_matrix(y_val, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=["Genuine", "Hallucinated"],
                    yticklabels=["Genuine", "Hallucinated"])
        ax.set_title(f"{title} — Confusion Matrix", fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"))
    plt.close()
    print("Saved: confusion_matrices.png")


def plot_roc_curves(y_val, lgbm_proba, cat_proba):
    fig, ax = plt.subplots(figsize=(8, 6))
    for proba, label, color in zip(
        [lgbm_proba, cat_proba],
        ["LightGBM", "CatBoost"],
        ["#4C72B0", "#DD8452"]
    ):
        fpr, tpr, _ = roc_curve(y_val, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plt.close()
    print("Saved: roc_curves.png")


def plot_comparison_bar(lgbm_metrics, cat_metrics):
    metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
    lgbm_scores = [lgbm_metrics['accuracy'], lgbm_metrics['f1'], lgbm_metrics['roc_auc']]
    cat_scores = [cat_metrics['accuracy'], cat_metrics['f1'], cat_metrics['roc_auc']]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, lgbm_scores, width, label='LightGBM', color='#4C72B0', edgecolor='black')
    b2 = ax.bar(x + width / 2, cat_scores, width, label='CatBoost', color='#DD8452', edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('LightGBM vs CatBoost — Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.bar_label(b1, fmt='%.3f', padding=3)
    ax.bar_label(b2, fmt='%.3f', padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_bar.png"))
    plt.close()
    print("Saved: comparison_bar.png")


def plot_feature_importance(lgbm_model, tfidf):
    importance = lgbm_model.feature_importances_
    n_tfidf = len(tfidf.get_feature_names_out())
    numeric_names = ['answer_len', 'prompt_len', 'answer_word_count',
                     'answer_sentence_count', 'has_no_answer', 'starts_with_step']
    feature_names = list(tfidf.get_feature_names_out()) + numeric_names
    feature_names = feature_names[:len(importance)]

    indices = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importance[indices], color='#4C72B0', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title("LightGBM — Top 20 Feature Importances", fontsize=14, fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    print("Saved: feature_importance.png")
