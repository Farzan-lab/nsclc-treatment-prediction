"""
================================================================================
Diagnostic Analysis — Why is performance stuck?
================================================================================
Systematically identify where we lose accuracy before trying to fix it.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

SPLITS_DIR = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits/')

X_train_clin = pd.read_csv(SPLITS_DIR / 'X_train_clinical.csv')
X_val_clin   = pd.read_csv(SPLITS_DIR / 'X_val_clinical.csv')
X_test_clin  = pd.read_csv(SPLITS_DIR / 'X_test_clinical.csv')
X_train_all  = pd.read_csv(SPLITS_DIR / 'X_train_all.csv')
X_val_all    = pd.read_csv(SPLITS_DIR / 'X_val_all.csv')
X_test_all   = pd.read_csv(SPLITS_DIR / 'X_test_all.csv')
y_train = np.load(SPLITS_DIR / 'y_train.npy')
y_val   = np.load(SPLITS_DIR / 'y_val.npy')
y_test  = np.load(SPLITS_DIR / 'y_test.npy')

clin_cols = X_train_clin.columns.tolist()
gen_cols  = [c for c in X_train_all.columns if c not in clin_cols]
CLASS_NAMES = ['Immunotherapy', 'Chemotherapy', 'Targeted']

def mf1(yt, yp): return round(f1_score(yt, yp, average='macro'), 4)

# Combine train+val for cross-validation (more robust than single split)
X_tv_clin = pd.concat([X_train_clin, X_val_clin], axis=0).reset_index(drop=True)
X_tv_all  = pd.concat([X_train_all,  X_val_all],  axis=0).reset_index(drop=True)
y_tv      = np.concatenate([y_train, y_val])

print("="*70)
print("  DIAGNOSTIC 1: Cross-Validation Stability")
print("="*70)
print("  (single train/test split can be misleading — CV shows true performance)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, X in [('Clinical only', X_tv_clin), ('All features', X_tv_all)]:
    model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          objective='multi:softprob', num_class=3,
                          eval_metric='mlogloss', random_state=42,
                          n_jobs=-1, verbosity=0)
    scores = cross_val_score(model, X, y_tv, cv=skf,
                             scoring='f1_macro', n_jobs=-1)
    print(f"\n  {name}:")
    print(f"    CV Macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    Range: [{scores.min():.4f}, {scores.max():.4f}]")

print("\n" + "="*70)
print("  DIAGNOSTIC 2: Is the test set unusually hard?")
print("="*70)

# Train on train+val, test on test
model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                      subsample=0.8, colsample_bytree=0.8,
                      objective='multi:softprob', num_class=3,
                      eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
model.fit(X_tv_all, y_tv)
test_f1 = mf1(y_test, model.predict(X_test_all))
print(f"\n  All-features test F1: {test_f1:.4f}")
print(f"  CV all-features F1  : (compare with Diagnostic 1)")
print(f"  → If test << CV, the test split is unlucky")

print("\n" + "="*70)
print("  DIAGNOSTIC 3: Per-Class Confusion (all features)")
print("="*70)

cm = confusion_matrix(y_test, model.predict(X_test_all))
print(f"\n  {'':16} {'→Immuno':>10} {'→Chemo':>10} {'→Target':>10}")
for i, cls in enumerate(CLASS_NAMES):
    row = cm[i]
    print(f"  {cls:<16} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# Which class is hardest?
print(f"\n  Most confused pairs:")
for i in range(3):
    for j in range(3):
        if i != j and cm[i][j] > 30:
            print(f"    {CLASS_NAMES[i]} → predicted as {CLASS_NAMES[j]}: {cm[i][j]} times")

print("\n" + "="*70)
print("  DIAGNOSTIC 4: How much does EACH modality contribute?")
print("="*70)

configs = {
    'Clinical only'        : X_tv_clin.columns.tolist(),
    'Genomic only'         : gen_cols,
    'Clinical + Genomic'   : X_tv_all.columns.tolist(),
}

for name, cols in configs.items():
    model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          objective='multi:softprob', num_class=3,
                          eval_metric='mlogloss', random_state=42,
                          n_jobs=-1, verbosity=0)
    scores = cross_val_score(model, X_tv_all[cols], y_tv, cv=skf,
                             scoring='f1_macro', n_jobs=-1)
    print(f"  {name:<22} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

print("\n" + "="*70)
print("  DIAGNOSTIC 5: Genomic-only feature importance")
print("="*70)
print("  (which genes drive treatment prediction?)")

model_gen = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          objective='multi:softprob', num_class=3,
                          eval_metric='mlogloss', random_state=42,
                          n_jobs=-1, verbosity=0)
model_gen.fit(X_tv_all[gen_cols], y_tv)
importances = pd.Series(model_gen.feature_importances_, index=gen_cols)
top = importances.sort_values(ascending=False).head(10)
print()
for gene, imp in top.items():
    bar = '█' * int(imp * 100)
    print(f"    {gene.replace('MUT_',''):<12} {imp:.4f}  {bar}")

print("\n" + "="*70)
print("  DIAGNOSTIC 6: Best possible with hyperparameter search")
print("="*70)
print("  (quick grid on all-features to find realistic ceiling)")

from itertools import product
best_f1, best_params = 0, None
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.1],
    'n_estimators': [300, 600],
}
for md, lr, ne in product(param_grid['max_depth'],
                           param_grid['learning_rate'],
                           param_grid['n_estimators']):
    model = XGBClassifier(n_estimators=ne, learning_rate=lr, max_depth=md,
                          subsample=0.8, colsample_bytree=0.8,
                          objective='multi:softprob', num_class=3,
                          eval_metric='mlogloss', random_state=42,
                          n_jobs=-1, verbosity=0)
    scores = cross_val_score(model, X_tv_all, y_tv, cv=skf,
                             scoring='f1_macro', n_jobs=-1)
    if scores.mean() > best_f1:
        best_f1 = scores.mean()
        best_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne}

print(f"\n  Best CV F1: {best_f1:.4f}")
print(f"  Best params: {best_params}")

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
print(f"""
  Key questions answered:
  1. Is CV performance higher than single-split? → check Diagnostic 1 vs 2
  2. Which class is hardest? → Diagnostic 3
  3. Does genomic-only beat clinical-only? → Diagnostic 4
  4. What's the realistic ceiling with tuning? → Diagnostic 6 = {best_f1:.4f}
""")
