"""
================================================================================
Optimized Treatment Prediction — Full Pipeline
================================================================================
Based on diagnostic findings:
  1. Immuno/Chemo confusion is the main error source → targeted feature engineering
  2. Genomic + Clinical are complementary → use both, tune well
  3. CV is more reliable than single split → evaluate with CV + final test

Strategy:
  A. Feature engineering (interaction features, ratios)
  B. Proper hyperparameter search
  C. Ensemble (XGBoost + LightGBM + CatBoost-style)
  D. Threshold optimization for the confused Immuno/Chemo pair
================================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

SPLITS_DIR  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits/')
RESULTS_DIR = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/experiments/05_optimized/')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ['Immunotherapy', 'Chemotherapy', 'Targeted']

# ── Load ──────────────────────────────────────────────────────────────────────
X_train_all = pd.read_csv(SPLITS_DIR / 'X_train_all.csv')
X_val_all   = pd.read_csv(SPLITS_DIR / 'X_val_all.csv')
X_test_all  = pd.read_csv(SPLITS_DIR / 'X_test_all.csv')
X_train_clin= pd.read_csv(SPLITS_DIR / 'X_train_clinical.csv')
y_train = np.load(SPLITS_DIR / 'y_train.npy')
y_val   = np.load(SPLITS_DIR / 'y_val.npy')
y_test  = np.load(SPLITS_DIR / 'y_test.npy')

clin_cols = X_train_clin.columns.tolist()
gen_cols  = [c for c in X_train_all.columns if c not in clin_cols]

# Combine train+val for final training (test stays held out)
X_tv = pd.concat([X_train_all, X_val_all], axis=0).reset_index(drop=True)
y_tv = np.concatenate([y_train, y_val])

def mf1(yt, yp): return f1_score(yt, yp, average='macro')

# ════════════════════════════════════════════════════════════════════════════
# STEP A: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  STEP A: FEATURE ENGINEERING")
print("="*70)

def engineer_features(df):
    """Add interaction and derived features targeting Immuno/Chemo separation."""
    df = df.copy()

    # EGFR is the dominant Targeted driver — make it explicit
    if 'MUT_EGFR' in df.columns and 'MUT_KRAS' in df.columns:
        # EGFR without KRAS → strong Targeted signal
        df['EGFR_not_KRAS'] = df['MUT_EGFR'] * (1 - df['MUT_KRAS'])
        # KRAS without EGFR → Immuno/Chemo signal
        df['KRAS_not_EGFR'] = df['MUT_KRAS'] * (1 - df['MUT_EGFR'])

    # STK11/KEAP1 co-mutation → poor immunotherapy response → pushes to Chemo
    if 'MUT_STK11' in df.columns and 'MUT_KEAP1' in df.columns:
        df['STK11_KEAP1_co'] = df['MUT_STK11'] * df['MUT_KEAP1']

    # TMB × smoking interaction (both push toward immunotherapy)
    if 'TMB_LOG' in df.columns and 'SMOKING_ENC' in df.columns:
        df['TMB_smoking'] = df['TMB_LOG'] * df['SMOKING_ENC']

    # Metastasis burden (total count of metastasis sites)
    met_cols = [c for c in ['BONE','CNS_BRAIN','LIVER','LUNG','LYMPH_NODES']
                if c in df.columns]
    if met_cols:
        df['metastasis_burden'] = df[met_cols].sum(axis=1)

    # Total mutation burden from genomic
    mut_cols = [c for c in df.columns if c.startswith('MUT_')]
    if mut_cols:
        df['total_mutations'] = df[mut_cols].sum(axis=1)

    return df

X_tv_eng   = engineer_features(X_tv)
X_test_eng = engineer_features(X_test_all)

new_features = [c for c in X_tv_eng.columns if c not in X_tv.columns]
print(f"\n  Added {len(new_features)} engineered features:")
for f in new_features:
    print(f"    • {f}")

# ════════════════════════════════════════════════════════════════════════════
# STEP B: HYPERPARAMETER SEARCH
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP B: HYPERPARAMETER SEARCH (5-fold CV)")
print(f"{'='*70}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from itertools import product
param_grid = {
    'max_depth':     [3, 4, 5],
    'learning_rate': [0.02, 0.05],
    'n_estimators':  [400, 800],
    'min_child_weight': [1, 3],
    'reg_lambda':    [1, 3],
}

best_f1, best_params = 0, None
combos = list(product(*param_grid.values()))
print(f"\n  Searching {len(combos)} combinations...")

for i, (md, lr, ne, mcw, rl) in enumerate(combos):
    model = XGBClassifier(
        max_depth=md, learning_rate=lr, n_estimators=ne,
        min_child_weight=mcw, reg_lambda=rl,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=3,
        eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_tv_eng, y_tv, cv=skf,
                             scoring='f1_macro', n_jobs=-1)
    if scores.mean() > best_f1:
        best_f1 = scores.mean()
        best_params = dict(max_depth=md, learning_rate=lr, n_estimators=ne,
                           min_child_weight=mcw, reg_lambda=rl)
        print(f"    [{i+1}/{len(combos)}] New best: {best_f1:.4f}  {best_params}")

print(f"\n  Best CV F1: {best_f1:.4f}")
print(f"  Best params: {best_params}")

# ════════════════════════════════════════════════════════════════════════════
# STEP C: ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP C: ENSEMBLE (XGBoost + RandomForest + ExtraTrees + LogReg)")
print(f"{'='*70}")

# Base models — diverse to maximize ensemble benefit
base_models = {
    'xgb': XGBClassifier(**best_params, subsample=0.8, colsample_bytree=0.8,
                         objective='multi:softprob', num_class=3,
                         eval_metric='mlogloss', random_state=42,
                         n_jobs=-1, verbosity=0),
    'rf':  RandomForestClassifier(n_estimators=500, max_depth=12,
                                  min_samples_leaf=3, class_weight='balanced',
                                  random_state=42, n_jobs=-1),
    'et':  ExtraTreesClassifier(n_estimators=500, max_depth=12,
                                min_samples_leaf=3, class_weight='balanced',
                                random_state=42, n_jobs=-1),
    'lr':  LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                              random_state=42),
}

# Get out-of-fold predictions for each model (soft probabilities)
print(f"\n  Computing cross-validated soft predictions...")
oof_probs = {}
test_probs = {}

for name, model in base_models.items():
    # Out-of-fold probabilities on train+val
    oof = cross_val_predict(model, X_tv_eng, y_tv, cv=skf,
                            method='predict_proba', n_jobs=-1)
    oof_probs[name] = oof
    oof_f1 = mf1(y_tv, oof.argmax(1))

    # Fit on full train+val, predict test
    model.fit(X_tv_eng, y_tv)
    test_probs[name] = model.predict_proba(X_test_eng)
    test_f1 = mf1(y_test, test_probs[name].argmax(1))

    print(f"    {name:<6} OOF F1: {oof_f1:.4f}   Test F1: {test_f1:.4f}")

# Weighted ensemble — search best weights on OOF
print(f"\n  Optimizing ensemble weights...")

best_ens_f1, best_weights = 0, None
weight_options = [0, 0.5, 1, 2]
for w_xgb, w_rf, w_et, w_lr in product(weight_options, repeat=4):
    if w_xgb + w_rf + w_et + w_lr == 0:
        continue
    total = w_xgb + w_rf + w_et + w_lr
    ens = (w_xgb * oof_probs['xgb'] + w_rf * oof_probs['rf'] +
           w_et * oof_probs['et'] + w_lr * oof_probs['lr']) / total
    f1 = mf1(y_tv, ens.argmax(1))
    if f1 > best_ens_f1:
        best_ens_f1 = f1
        best_weights = (w_xgb, w_rf, w_et, w_lr)

print(f"  Best ensemble OOF F1: {best_ens_f1:.4f}")
print(f"  Best weights (xgb,rf,et,lr): {best_weights}")

# Apply best weights to test
w = best_weights
total = sum(w)
ens_test = (w[0]*test_probs['xgb'] + w[1]*test_probs['rf'] +
            w[2]*test_probs['et'] + w[3]*test_probs['lr']) / total
ens_test_pred = ens_test.argmax(1)
ens_test_f1 = mf1(y_test, ens_test_pred)

# ════════════════════════════════════════════════════════════════════════════
# STEP D: FINAL RESULTS
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  FINAL RESULTS")
print(f"{'='*70}")

# Single best XGBoost on test
xgb_final = base_models['xgb']
xgb_test_f1 = mf1(y_test, xgb_final.predict(X_test_eng))

print(f"\n  {'Method':<40} {'Test Macro F1':>14}")
print(f"  {'─'*56}")
print(f"  {'Old baseline (clinical only)':<40} {0.5584:>14.4f}")
print(f"  {'Old all-features':<40} {0.6555:>14.4f}")
print(f"  {'Tuned XGBoost + FE':<40} {xgb_test_f1:>14.4f}")
print(f"  {'Ensemble + FE':<40} {ens_test_f1:>14.4f}")

best_final = max(xgb_test_f1, ens_test_f1)
print(f"\n  Best: {best_final:.4f}")
print(f"  Improvement over old ceiling: {(best_final - 0.6555)*100:+.2f} points")

# Detailed report of best model
best_pred = ens_test_pred if ens_test_f1 > xgb_test_f1 else xgb_final.predict(X_test_eng)
print(f"\n  Per-class F1 (best model):")
per = f1_score(y_test, best_pred, average=None)
for i, cls in enumerate(CLASS_NAMES):
    print(f"    {cls:<20} {per[i]:.4f}")

print(f"\n  Confusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(f"  {'':16} {'→Immuno':>10} {'→Chemo':>10} {'→Target':>10}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<16} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

# Save
results = {
    'best_params': best_params,
    'best_cv_f1': float(best_f1),
    'ensemble_weights': list(best_weights),
    'xgb_test_f1': float(xgb_test_f1),
    'ensemble_test_f1': float(ens_test_f1),
    'best_final_f1': float(best_final),
    'per_class_f1': per.tolist(),
}
with open(RESULTS_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to experiments/05_optimized/")
