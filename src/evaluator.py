import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def evaluate_all_models(results, y_test, feature_names):
    """
    Computes statistical evaluation metrics for all benchmark models and generates
    high-resolution clinical diagnostic plots.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "figure.dpi": 300})

    metrics_list = []

    print("\n" + "="*80)
    print(f"{'Model':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("="*80)

    for name, data in results.items():
        y_pred = data["y_pred"]
        y_prob = data["y_prob"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        print(f"{name:<35} {acc*100:6.2f}%   {prec*100:6.2f}%   {rec*100:6.2f}%   {f1:6.4f}     {auc:6.4f}")

    print("="*80)
    metrics_df = pd.DataFrame(metrics_list)

    # 1. ROC Curves Plot
    plt.figure(figsize=(9, 7))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
        auc = roc_auc_score(y_test, data["y_prob"])
        linewidth = 2.5 if "Champion" in name else 1.5
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=linewidth)

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[OK] Saved ROC Curves to: {roc_path}")

    # 2. Confusion Matrix Heatmap (Champion Model)
    champion_name = "Tuned Gradient Boosting (Champion)" if "Tuned Gradient Boosting (Champion)" in results else list(results.keys())[0]
    cm = confusion_matrix(y_test, results[champion_name]["y_pred"])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(7, 6))
    annot_matrix = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=True,
        xticklabels=["Low/Moderate Risk", "High Risk"],
        yticklabels=["Low/Moderate Risk", "High Risk"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix — {champion_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Clinical Diagnosis", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Patient Health State", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Confusion Matrix to: {cm_path}")

    # 3. Feature Importance Plot (Champion Model)
    champ_model = results[champion_name]["model"]
    if hasattr(champ_model, "feature_importances_"):
        importances = champ_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": [f.replace("_", " ").title() for f in feature_names],
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(
            x="Importance", y="Feature", data=feat_df,
            palette="crest"
        )
        for p in barplot.patches:
            barplot.annotate(
                f"{p.get_width()*100:.1f}%",
                (p.get_width() + 0.005, p.get_y() + p.get_height() / 2),
                ha="left", va="center", fontsize=10, fontweight="bold"
            )
        plt.title("Clinical Biomarker Feature Importance (Gini Impurity Reduction)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Relative Importance Weight", fontsize=11, fontweight="bold")
        plt.ylabel("Clinical Predictor", fontsize=11, fontweight="bold")
        plt.xlim(0, max(importances) * 1.18)
        plt.tight_layout()
        fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[OK] Saved Feature Importance to: {fi_path}")

    # 4. Model Comparison Bar Chart
    plt.figure(figsize=(11, 6))
    melted_df = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "ROC-AUC"],
                                var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=melted_df, palette="viridis")
    plt.xticks(rotation=20, ha="right", fontweight="bold")
    plt.ylim(0.5, 1.02)
    plt.title("Comparative Evaluation Across Machine Learning Algorithms", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Performance Score", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    comp_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Model Comparison to: {comp_path}")

    return metrics_df
