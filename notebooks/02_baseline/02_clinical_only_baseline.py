"""
================================================================================
Phase 3 — Baseline Models
Script 02: Clinical-Only Baseline
================================================================================

GOAL:
    Train three models using ONLY the 18 clinical features.
    This establishes the FLOOR — the minimum performance we expect
    from our cross-modal model in Phase 4.

MODELS:
    1. Logistic Regression  — linear, interpretable, fast
    2. Random Forest        — non-linear, ensemble of trees
    3. XGBoost              — gradient boosting, usually best on tabular data

WHY class_weight='balanced':
    Our dataset has imbalance ratio 1.93x (Immunotherapy 43% vs Targeted 23%).
    Without balancing, models learn to favor the majority class.
    class_weight='balanced' automatically adjusts loss weights so each
    class contributes equally to training, regardless of its sample size.

KEY METRIC: Macro F1
    We use Macro F1 (not accuracy) because:
    - Accuracy is misleading with imbalanced classes
    - Macro F1 gives equal weight to all three classes
    - A model that ignores Targeted (22%) would still get 78% accuracy
      but would have a poor Macro F1
================================================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path('data/splits/')
RESULTS_DIR = Path('experiments/02_clinical_only/')
FIG_DIR     = Path('figures/baseline/')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP   = {0: 'Immunotherapy', 1: 'Chemotherapy', 2: 'Targeted'}
CLASS_NAMES = ['Immunotherapy', 'Chemotherapy', 'Targeted']

# ================================================================================
# SECTION 1: Load Data
# ================================================================================
print("Loading preprocessed data...")

X_train = pd.read_csv(SPLITS_DIR / 'X_train_clinical.csv')
X_val   = pd.read_csv(SPLITS_DIR / 'X_val_clinical.csv')
X_test  = pd.read_csv(SPLITS_DIR / 'X_test_clinical.csv')
y_train = np.load(SPLITS_DIR / 'y_train.npy')
y_val   = np.load(SPLITS_DIR / 'y_val.npy')
y_test  = np.load(SPLITS_DIR / 'y_test.npy')

print(f"✓ Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"  Features: {X_train.shape[1]} clinical features")

# ================================================================================
# SECTION 2: Define Models
# ================================================================================
# Each model uses class_weight='balanced' to handle class imbalance.
# We use default hyperparameters first — these are already reasonable baselines.
# In a real paper, we would tune these with grid search on the val set.

models = {

    'Logistic Regression': LogisticRegression(
        # C: inverse of regularization strength
        #   Small C → strong regularization → simpler model
        #   Large C → weak regularization → fits training data more closely
        C=1.0,
        # max_iter: maximum number of optimization steps
        # 1000 is safe for convergence on this dataset size
        max_iter=1000,
        # solver='lbfgs' handles multiclass natively in newer sklearn versions
        # multi_class parameter was removed in sklearn >= 1.5
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
    ),

    'Random Forest': RandomForestClassifier(
        # n_estimators: number of trees in the forest
        # More trees = more stable predictions, slower training
        n_estimators=300,
        # max_depth: maximum depth of each tree
        # None = grow until all leaves are pure (may overfit)
        # We limit to prevent overfitting on 4,154 training samples
        max_depth=10,
        # min_samples_leaf: minimum samples required at a leaf node
        # Higher = smoother decision boundaries, less overfitting
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,    # use all CPU cores
    ),

    'XGBoost': XGBClassifier(
        # n_estimators: number of boosting rounds
        n_estimators=500,
        # learning_rate: how much each tree corrects the previous ones
        # Smaller = more conservative, less likely to overfit
        learning_rate=0.05,
        # max_depth: depth of each tree — shallower = less overfit
        max_depth=5,
        # subsample: fraction of training data used per tree
        # 0.8 = each tree sees 80% of training data → reduces overfit
        subsample=0.8,
        # colsample_bytree: fraction of features used per tree
        colsample_bytree=0.8,
        # objective: multiclass classification with softmax
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        # early_stopping_rounds handled separately below
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}

# ================================================================================
# SECTION 3: Train and Evaluate
# ================================================================================

def evaluate(model, X, y, split_name):
    """Compute all metrics for a given split."""
    y_pred = model.predict(X)

    # predict_proba returns probability for each class
    # needed for AUC calculation
    try:
        y_prob = model.predict_proba(X)
    except AttributeError:
        y_prob = None

    metrics = {
        'accuracy'   : round(accuracy_score(y, y_pred), 4),
        'macro_f1'   : round(f1_score(y, y_pred, average='macro'), 4),
        'weighted_f1': round(f1_score(y, y_pred, average='weighted'), 4),
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(y, y_pred, average=None)
    for i, cls in LABEL_MAP.items():
        metrics[f'f1_{cls.lower().replace(" ", "_")}'] = round(per_class_f1[i], 4)

    if y_prob is not None:
        try:
            metrics['auc_ovr'] = round(
                roc_auc_score(y, y_prob, multi_class='ovr', average='macro'), 4)
        except Exception:
            metrics['auc_ovr'] = float('nan')

    return metrics, y_pred


print(f"\n{'='*65}")
print(f"  TRAINING — CLINICAL ONLY BASELINE ({X_train.shape[1]} features)")
print(f"{'='*65}")

all_results = {}

for model_name, model in models.items():
    print(f"\n── {model_name} ──────────────────────────────────")

    # Special handling for XGBoost: use early stopping on val set
    # to prevent overfitting
    if model_name == 'XGBoost':
        model.set_params(early_stopping_rounds=30)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        best_iter = model.best_iteration
        print(f"  Best iteration: {best_iter}")
    else:
        model.fit(X_train, y_train)

    # Evaluate on train, val, test
    train_metrics, _     = evaluate(model, X_train, y_train, 'train')
    val_metrics,   _     = evaluate(model, X_val,   y_val,   'val')
    test_metrics,  y_pred_test = evaluate(model, X_test,  y_test,  'test')

    # Print results
    print(f"\n  {'Metric':<20} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'─'*45}")
    for metric in ['accuracy', 'macro_f1', 'auc_ovr']:
        if metric in train_metrics:
            print(f"  {metric:<20} "
                  f"{train_metrics[metric]:>8.4f} "
                  f"{val_metrics[metric]:>8.4f} "
                  f"{test_metrics[metric]:>8.4f}")

    print(f"\n  Per-class F1 (test):")
    for cls in CLASS_NAMES:
        key = f'f1_{cls.lower().replace(" ", "_")}'
        if key in test_metrics:
            print(f"    {cls:<20} {test_metrics[key]:.4f}")

    # Check for overfitting
    overfit_gap = train_metrics['macro_f1'] - val_metrics['macro_f1']
    if overfit_gap > 0.1:
        print(f"\n  ⚠ Overfitting detected: train-val gap = {overfit_gap:.3f}")
    else:
        print(f"\n  ✓ No significant overfitting (gap = {overfit_gap:.3f})")

    all_results[model_name] = {
        'train': train_metrics,
        'val':   val_metrics,
        'test':  test_metrics,
    }

    # Save model
    joblib.dump(model, RESULTS_DIR / f'{model_name.replace(" ", "_").lower()}.pkl')

# ================================================================================
# SECTION 4: Comparison Table
# ================================================================================
print(f"\n{'='*65}")
print(f"  RESULTS SUMMARY — CLINICAL ONLY BASELINE")
print(f"{'='*65}")
print(f"\n  {'Model':<25} {'Val F1':>8} {'Test F1':>9} {'Test AUC':>10}")
print(f"  {'─'*55}")

best_model_name = None
best_val_f1     = 0

for name, results in all_results.items():
    val_f1  = results['val']['macro_f1']
    test_f1 = results['test']['macro_f1']
    test_auc = results['test'].get('auc_ovr', float('nan'))
    marker  = " ←" if val_f1 > best_val_f1 else ""
    if val_f1 > best_val_f1:
        best_val_f1     = val_f1
        best_model_name = name
    print(f"  {name:<25} {val_f1:>8.4f} {test_f1:>9.4f} {test_auc:>10.4f}{marker}")

print(f"\n  Best model on val: {best_model_name}")

# ================================================================================
# SECTION 5: Confusion Matrix for Best Model
# ================================================================================
best_model = joblib.load(
    RESULTS_DIR / f'{best_model_name.replace(" ", "_").lower()}.pkl')
_, y_pred_best = evaluate(best_model, X_test, y_test, 'test')

cm = confusion_matrix(y_test, y_pred_best)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5)
axes[0].set_title(f'Confusion Matrix — {best_model_name}\n(Clinical Only, counts)',
                  fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Percentages
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5)
axes[1].set_title(f'Confusion Matrix — {best_model_name}\n(Clinical Only, %)',
                  fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrix_clinical_only.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: figures/baseline/confusion_matrix_clinical_only.png")

# ================================================================================
# SECTION 6: Save Results
# ================================================================================
with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*65}")
print(f"✓ Clinical-only baseline complete")
print(f"{'='*65}")
print(f"\n  FLOOR established:")
print(f"  Best model    : {best_model_name}")
print(f"  Val Macro F1  : {best_val_f1:.4f}")
print(f"  Test Macro F1 : {all_results[best_model_name]['test']['macro_f1']:.4f}")
print(f"\n  Next: run 03_all_features_baseline.py")
