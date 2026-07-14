"""
================================================================================
Phase 3 — Baseline Models
Script 03: All Features Upper Bound
================================================================================

GOAL:
    Train the same three models using ALL 64 features (18 clinical + 46 genomic).
    This establishes the CEILING — the best possible performance when
    genomic data is available at inference time.

THE GAP:
    gap = ceiling_F1 - floor_F1

    This gap tells us how much genomic data adds over clinical alone.
    Our cross-modal model (Phase 4) should close most of this gap
    while using only clinical features at inference time.

    gap > 0.05  → genomic data adds significant value → distillation worthwhile
    gap < 0.02  → genomic barely helps → reconsider approach
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
    confusion_matrix
)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR       = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits/')
RESULTS_DIR_ALL  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/experiments/03_all_features/')
RESULTS_DIR_CLIN = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/experiments/02_clinical_only/')
FIG_DIR          = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/figures/baseline/')
RESULTS_DIR_ALL.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP   = {0: 'Immunotherapy', 1: 'Chemotherapy', 2: 'Targeted'}
CLASS_NAMES = ['Immunotherapy', 'Chemotherapy', 'Targeted']

# ================================================================================
# SECTION 1: Load Data
# ================================================================================
print("Loading preprocessed data...")

X_train = pd.read_csv(SPLITS_DIR / 'X_train_all.csv')
X_val   = pd.read_csv(SPLITS_DIR / 'X_val_all.csv')
X_test  = pd.read_csv(SPLITS_DIR / 'X_test_all.csv')
y_train = np.load(SPLITS_DIR / 'y_train.npy')
y_val   = np.load(SPLITS_DIR / 'y_val.npy')
y_test  = np.load(SPLITS_DIR / 'y_test.npy')

print(f"✓ Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"  Features: {X_train.shape[1]} (clinical + genomic)")

# ================================================================================
# SECTION 2: Define Models (identical to clinical-only script)
# ================================================================================
models = {

    'Logistic Regression': LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ),

    'XGBoost': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}

# ================================================================================
# SECTION 3: Train and Evaluate
# ================================================================================

def evaluate(model, X, y):
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except Exception:
        y_prob = None

    metrics = {
        'accuracy'   : round(accuracy_score(y, y_pred), 4),
        'macro_f1'   : round(f1_score(y, y_pred, average='macro'), 4),
        'weighted_f1': round(f1_score(y, y_pred, average='weighted'), 4),
    }
    per_class = f1_score(y, y_pred, average=None)
    for i, cls in LABEL_MAP.items():
        metrics[f'f1_{cls.lower().replace(" ", "_")}'] = round(per_class[i], 4)

    if y_prob is not None:
        try:
            metrics['auc_ovr'] = round(
                roc_auc_score(y, y_prob, multi_class='ovr', average='macro'), 4)
        except Exception:
            metrics['auc_ovr'] = float('nan')

    return metrics, y_pred


print(f"\n{'='*65}")
print(f"  TRAINING — ALL FEATURES ({X_train.shape[1]} features)")
print(f"{'='*65}")

all_results = {}

for model_name, model in models.items():
    print(f"\n── {model_name} ──────────────────────────────────")

    if model_name == 'XGBoost':
        model.set_params(early_stopping_rounds=30)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        print(f"  Best iteration: {model.best_iteration}")
    else:
        model.fit(X_train, y_train)

    train_metrics, _          = evaluate(model, X_train, y_train)
    val_metrics,   _          = evaluate(model, X_val,   y_val)
    test_metrics,  y_pred_test = evaluate(model, X_test,  y_test)

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

    gap = train_metrics['macro_f1'] - val_metrics['macro_f1']
    status = "⚠ Overfitting" if gap > 0.1 else "✓ OK"
    print(f"\n  Train-Val gap: {gap:.3f}  {status}")

    all_results[model_name] = {
        'train': train_metrics,
        'val':   val_metrics,
        'test':  test_metrics,
    }
    joblib.dump(model, RESULTS_DIR_ALL / f'{model_name.replace(" ", "_").lower()}.pkl')

# ================================================================================
# SECTION 4: Gap Analysis — Clinical vs All Features
# ================================================================================
print(f"\n{'='*65}")
print(f"  GAP ANALYSIS — Clinical-Only vs All Features")
print(f"{'='*65}")

# Load clinical-only results
with open(RESULTS_DIR_CLIN / 'results.json') as f:
    clinical_results = json.load(f)

print(f"\n  {'Model':<25} {'Clinical F1':>12} {'All F1':>10} {'Gap':>8} {'Gap %':>8}")
print(f"  {'─'*65}")

best_all_name = None
best_all_val  = 0

for name in models.keys():
    clin_f1 = clinical_results[name]['test']['macro_f1']
    all_f1  = all_results[name]['test']['macro_f1']
    gap     = all_f1 - clin_f1
    gap_pct = gap / clin_f1 * 100
    marker  = " ←" if all_results[name]['val']['macro_f1'] > best_all_val else ""
    if all_results[name]['val']['macro_f1'] > best_all_val:
        best_all_val  = all_results[name]['val']['macro_f1']
        best_all_name = name
    print(f"  {name:<25} {clin_f1:>12.4f} {all_f1:>10.4f} "
          f"{gap:>+8.4f} {gap_pct:>+7.1f}%{marker}")

# Overall gap assessment
avg_gap = np.mean([
    all_results[n]['test']['macro_f1'] - clinical_results[n]['test']['macro_f1']
    for n in models.keys()
])
print(f"\n  Average gap: {avg_gap:+.4f}")
if avg_gap > 0.05:
    print(f"  ✓ Genomic data adds significant value (gap > 0.05)")
    print(f"    → Cross-modal distillation is well-justified")
elif avg_gap > 0.02:
    print(f"  ⚡ Genomic data adds moderate value (gap 0.02–0.05)")
    print(f"    → Distillation may help, but improvement may be modest")
else:
    print(f"  ⚠ Genomic data adds little value (gap < 0.02)")
    print(f"    → Reconsider whether distillation is worthwhile")

# ================================================================================
# SECTION 5: Comparison Bar Chart
# ================================================================================
model_names = list(models.keys())
clin_f1s    = [clinical_results[n]['test']['macro_f1'] for n in model_names]
all_f1s     = [all_results[n]['test']['macro_f1']      for n in model_names]

x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, clin_f1s, width,
               label='Clinical Only (18 features)',
               color='#90CAF9', edgecolor='white')
bars2 = ax.bar(x + width/2, all_f1s, width,
               label='All Features (64 features)',
               color='#1565C0', edgecolor='white')

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Macro F1 (Test)', fontsize=11)
ax.set_title('Clinical-Only vs All Features — Macro F1 Comparison\n'
             '(gap shows value added by genomic features)',
             fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(FIG_DIR / 'gap_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: figures/baseline/gap_analysis.png")

# ================================================================================
# SECTION 6: Confusion Matrix — Best All-Features Model
# ================================================================================
best_model = joblib.load(
    RESULTS_DIR_ALL / f'{best_all_name.replace(" ", "_").lower()}.pkl')
_, y_pred_best = evaluate(best_model, X_test, y_test)

cm     = confusion_matrix(y_test, y_pred_best)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5)
axes[0].set_title(f'Confusion Matrix — {best_all_name}\n(All Features, counts)',
                  fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5)
axes[1].set_title(f'Confusion Matrix — {best_all_name}\n(All Features, %)',
                  fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrix_all_features.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: figures/baseline/confusion_matrix_all_features.png")

# ================================================================================
# SECTION 7: Save and Summary
# ================================================================================
with open(RESULTS_DIR_ALL / 'results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*65}")
print(f"✓ All-features baseline complete")
print(f"{'='*65}")
print(f"\n  FLOOR  (clinical only) : "
      f"{max(clinical_results[n]['test']['macro_f1'] for n in models):.4f}")
print(f"  CEILING (all features) : "
      f"{max(all_results[n]['test']['macro_f1'] for n in models):.4f}")
print(f"  Gap to close           : {avg_gap:+.4f}")
print(f"\n  Next: Phase 4 — Cross-Modal Architecture")
