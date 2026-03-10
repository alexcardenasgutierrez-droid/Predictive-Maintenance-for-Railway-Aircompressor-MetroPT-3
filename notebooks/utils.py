from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    experiment_name="Unknown",
    explanation=None,
    threshold=0.5
):
    # -------------------------
    # Create results folder
    # -------------------------
    save_dir = Path("../results")
    save_dir.mkdir(exist_ok=True)

    metrics_path = save_dir / "metrics.csv"
    cm_path = save_dir / "confusion_matrices.csv"

    # -------------------------
    # Predictions
    # -------------------------
    y_train_pred = model.predict(X_train)

    # Use threshold on test instead of model.predict()
    try:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_test_proba)
        # Adjust decision threshold — default 0.5, lower = more failures flagged
        y_test_pred = (y_test_proba >= threshold).astype(int)
    except:
        # Fallback for models without predict_proba (e.g. SVM without probability=True)
        y_test_pred = model.predict(X_test)
        roc_auc = None

    # -------------------------
    # Model name
    # -------------------------
    model_name = model.__class__.__name__

    # -------------------------
    # Build results row
    # -------------------------
    results_row = pd.DataFrame([{
        "experiment_name": experiment_name,
        "explanation":     explanation,
        "model":           model_name,
        "threshold":       threshold,

        # Failure class (class 1) metrics — the ones that actually matter
        "F1_Train(Class1)":        round(f1_score(y_train, y_train_pred, average="binary", zero_division=0), 4),
        "F1_Test(Class1)":         round(f1_score(y_test, y_test_pred,  average="binary", zero_division=0), 4),

        "Precision_Test(Class1)":  round(precision_score(y_test, y_test_pred,  average="binary", zero_division=0), 4),

        "Recall_Test(Class1)":     round(recall_score(y_test, y_test_pred,  average="binary", zero_division=0), 4),

        "Overfit_Gap(F1)":     round(
            f1_score(y_train, y_train_pred, average="binary", zero_division=0) -
            f1_score(y_test,  y_test_pred,  average="binary", zero_division=0), 4
        ),  # ← now based on F1 instead of accuracy
    
        "ROC_AUC":         round(roc_auc, 4) if roc_auc is not None else None,
    }])

    # -------------------------
    # Append metrics to CSV
    # -------------------------
    if metrics_path.exists():
        old_results = pd.read_csv(metrics_path)
        results_df = pd.concat([old_results, results_row], ignore_index=True)
    else:
        results_df = results_row

    results_df.to_csv(metrics_path, index=False)

    # -------------------------
    # Confusion Matrix — save as CSV row
    # -------------------------
    cm = confusion_matrix(y_test, y_test_pred)

    cm_row = pd.DataFrame([{
        "experiment_name": experiment_name,
        "model": model_name,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    }])

    if cm_path.exists():
        old_cm = pd.read_csv(cm_path)
        cm_df = pd.concat([old_cm, cm_row], ignore_index=True)
    else:
        cm_df = cm_row

    cm_df.to_csv(cm_path, index=False)

    # -------------------------
    # Confusion Matrix — save as image
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=['No Failure', 'Failure'],
        yticklabels=['No Failure', 'Failure']
    )
    ax.set_title(f'Confusion Matrix — {experiment_name} (threshold={threshold})')  # ← NEW: show threshold in title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_dir / f"{experiment_name}.png", dpi=150)
    plt.show()
    plt.close()

    return results_row