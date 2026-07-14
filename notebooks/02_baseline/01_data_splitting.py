"""
================================================================================
Phase 3 — Baseline Models
Script 01: Data Splitting and Preprocessing Pipeline
================================================================================

WHAT THIS SCRIPT DOES:
    1. Splits dataset into train / val / test (stratified)
    2. Builds preprocessing pipeline (scaling + encoding)
    3. Saves split indices to disk — used by ALL subsequent models
    4. Verifies split quality (class distribution preserved)

WHY SPLIT ONCE AND SAVE:
    Every model in Phase 3 and Phase 4 must be evaluated on the
    exact same patients. If we re-split for each model, the comparison
    is unfair — one model might get an easier test set by chance.

SPLIT STRATEGY:
    test  : 15% → held out, never touched until final evaluation
    val   : 17% of remaining → hyperparameter tuning
    train : 68% of remaining → model fitting

    StratifiedKFold ensures the ratio of three treatment classes
    is preserved in each split — preventing accidental imbalance.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
SPLITS_DIR  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/splits/')
MODELS_DIR  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/experiments/')
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42   # fixed seed → reproducible splits

# ================================================================================
# SECTION 1: Define Feature Sets
# ================================================================================
# CLINICAL_FEATURES: available at inference time (no genomic sequencing needed)
# GENOMIC_FEATURES:  only available at training time
# TARGET:            what we want to predict

CLINICAL_FEATURES = [
    # Numeric
    'CURRENT_AGE_DEID',
    'TMB_LOG',
    'MSI_LOG',
    'TUMOR_PURITY',
    # Ordinal / Encoded
    'STAGE_ENC',
    'SMOKING_ENC',
    'PRIOR_MED_ENC',
    'HAS_PROGRESSION',
    # Subtype (binary)
    'SUBTYPE_Adenocarcinoma',
    'SUBTYPE_Squamous',
    'SUBTYPE_Neuroendocrine',
    'SUBTYPE_Other_NSCLC',
    # Metastasis (binary)
    'BONE',
    'CNS_BRAIN',
    'LIVER',
    'LUNG',
    'LYMPH_NODES',
    # Demographics
    'GENDER_ENC',
]

GENOMIC_FEATURES = [c for c in [
    'MUT_TP53', 'MUT_KRAS', 'MUT_EGFR', 'MUT_KEAP1', 'MUT_STK11',
    'MUT_RBM10', 'MUT_PTPRD', 'MUT_SMARCA4', 'MUT_NF1', 'MUT_CDKN2A',
    'MUT_ALK', 'MUT_AMER1', 'MUT_APC', 'MUT_ARID1A', 'MUT_ARID2',
    'MUT_ATM', 'MUT_ATRX', 'MUT_BRAF', 'MUT_CREBBP', 'MUT_ERBB2',
    'MUT_ERBB4', 'MUT_FAT1', 'MUT_GRIN2A', 'MUT_HGF', 'MUT_KDR',
    'MUT_KMT2C', 'MUT_KMT2D', 'MUT_MED12', 'MUT_MET', 'MUT_MGA',
    'MUT_NF1', 'MUT_NOTCH1', 'MUT_NOTCH4', 'MUT_NTRK3', 'MUT_PDGFRA',
    'MUT_PIK3C2G', 'MUT_PIK3CA', 'MUT_PIK3CG', 'MUT_POLE', 'MUT_PREX2',
    'MUT_PTEN', 'MUT_PTPRT', 'MUT_RB1', 'MUT_RBM10', 'MUT_SETD2',
    'MUT_SMAD4', 'MUT_TBX3', 'MUT_TERT', 'MUT_ZFHX3', 'MUT_EPHA3',
] if c not in ['MUT_NF1', 'MUT_RBM10']]  # avoid duplicates

TARGET     = 'TREATMENT_FINAL'
TARGET_ENC = 'TREATMENT_ENC'

# Label mapping — for readable output
LABEL_MAP = {0: 'Immunotherapy', 1: 'Chemotherapy', 2: 'Targeted'}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# ================================================================================
# SECTION 2: Load and Validate
# ================================================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert bool columns
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(float)

print(f"✓ Loaded: {df.shape[0]:,} patients, {df.shape[1]} columns")

# Validate all required features exist
missing_clinical = [c for c in CLINICAL_FEATURES if c not in df.columns]
missing_genomic  = [c for c in GENOMIC_FEATURES  if c not in df.columns]

if missing_clinical:
    print(f"⚠ Missing clinical features: {missing_clinical}")
    CLINICAL_FEATURES = [c for c in CLINICAL_FEATURES if c in df.columns]
if missing_genomic:
    print(f"⚠ Missing genomic features: {missing_genomic}")
    GENOMIC_FEATURES = [c for c in GENOMIC_FEATURES if c in df.columns]

print(f"\n  Clinical features : {len(CLINICAL_FEATURES)}")
print(f"  Genomic features  : {len(GENOMIC_FEATURES)}")
print(f"  Total features    : {len(CLINICAL_FEATURES) + len(GENOMIC_FEATURES)}")

# Encode target if not already done
if TARGET_ENC not in df.columns:
    df[TARGET_ENC] = df[TARGET].map(LABEL_MAP_INV)

# ================================================================================
# SECTION 3: Train / Val / Test Split
# ================================================================================
print(f"\n{'='*50}")
print("  SPLITTING DATA")
print(f"{'='*50}")

# Step 1: Hold out test set (15%)
# stratify= ensures class proportions are preserved in both halves
train_val, test = train_test_split(
    df,
    test_size=0.15,
    stratify=df[TARGET_ENC],
    random_state=RANDOM_STATE
)

# Step 2: Split remaining into train and val (val = ~17% of train_val = ~15% total)
train, val = train_test_split(
    train_val,
    test_size=0.20,
    stratify=train_val[TARGET_ENC],
    random_state=RANDOM_STATE
)

print(f"\n  Total   : {len(df):,} patients")
print(f"  Train   : {len(train):,} ({len(train)/len(df)*100:.1f}%)")
print(f"  Val     : {len(val):,} ({len(val)/len(df)*100:.1f}%)")
print(f"  Test    : {len(test):,} ({len(test)/len(df)*100:.1f}%)")

# Verify class distribution is preserved across splits
print(f"\n  Class distribution (%) across splits:")
print(f"  {'Class':<20} {'Full':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
print(f"  {'─'*55}")

for cls in LABEL_MAP.values():
    full_pct  = (df[TARGET] == cls).mean() * 100
    train_pct = (train[TARGET] == cls).mean() * 100
    val_pct   = (val[TARGET] == cls).mean() * 100
    test_pct  = (test[TARGET] == cls).mean() * 100
    print(f"  {cls:<20} {full_pct:>7.1f}% {train_pct:>7.1f}% "
          f"{val_pct:>7.1f}% {test_pct:>7.1f}%")

# Save split indices
np.save(str(SPLITS_DIR / 'train_indices.npy'), train.index.values)
np.save(str(SPLITS_DIR / 'val_indices.npy'),   val.index.values)
np.save(str(SPLITS_DIR / 'test_indices.npy'),  test.index.values)
print(f"\n✓ Split indices saved to {SPLITS_DIR}/")

# ================================================================================
# SECTION 4: Build Preprocessing Pipeline
# ================================================================================
# WHY STANDARDSCALER FOR NUMERIC FEATURES?
#   Many ML models (especially Logistic Regression) are sensitive to feature
#   scale. If age ranges 18-90 and TMB_LOG ranges 0-5, the model might
#   give more weight to age simply because its values are larger.
#   StandardScaler transforms each feature to mean=0, std=1.
#   Formula: z = (x - mean) / std
#
# WHY NOT SCALE BINARY FEATURES?
#   Binary features (0/1) are already on a meaningful scale.
#   Scaling them would change 0→some negative number and 1→some positive number,
#   which could confuse tree-based models.
#
# IMPORTANT: fit ONLY on train, then transform val and test.
#   If we fit on the full dataset, information from test leaks into training.

print(f"\n{'='*50}")
print("  BUILDING PREPROCESSING PIPELINE")
print(f"{'='*50}")

# Numeric features that need scaling
NUMERIC_TO_SCALE = [
    'CURRENT_AGE_DEID', 'TMB_LOG', 'MSI_LOG', 'TUMOR_PURITY',
]
NUMERIC_TO_SCALE = [c for c in NUMERIC_TO_SCALE if c in CLINICAL_FEATURES]

# Features that stay as-is (binary, ordinal, genomic)
PASSTHROUGH = [c for c in CLINICAL_FEATURES if c not in NUMERIC_TO_SCALE]

print(f"\n  Features to scale (StandardScaler) : {len(NUMERIC_TO_SCALE)}")
print(f"    {NUMERIC_TO_SCALE}")
print(f"\n  Features passed through (no scaling): {len(PASSTHROUGH)}")
print(f"    {PASSTHROUGH[:5]}...")

# Build ColumnTransformer pipeline
# remainder='passthrough' keeps all other columns unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), NUMERIC_TO_SCALE),
    ],
    remainder='passthrough'
)

# Fit on train only, then transform all splits
X_train_clinical = train[CLINICAL_FEATURES]
X_val_clinical   = val[CLINICAL_FEATURES]
X_test_clinical  = test[CLINICAL_FEATURES]

preprocessor.fit(X_train_clinical)

X_train_clinical_scaled = preprocessor.transform(X_train_clinical)
X_val_clinical_scaled   = preprocessor.transform(X_val_clinical)
X_test_clinical_scaled  = preprocessor.transform(X_test_clinical)

# Get feature names after transformation (scaler changes order)
feature_names_out = (NUMERIC_TO_SCALE +
                     [c for c in CLINICAL_FEATURES if c not in NUMERIC_TO_SCALE])

X_train_clinical_scaled = pd.DataFrame(
    X_train_clinical_scaled, columns=feature_names_out, index=train.index)
X_val_clinical_scaled   = pd.DataFrame(
    X_val_clinical_scaled,   columns=feature_names_out, index=val.index)
X_test_clinical_scaled  = pd.DataFrame(
    X_test_clinical_scaled,  columns=feature_names_out, index=test.index)

print(f"\n  Scaling verification (train set after scaling):")
for col in NUMERIC_TO_SCALE:
    if col in X_train_clinical_scaled.columns:
        mean = X_train_clinical_scaled[col].mean()
        std  = X_train_clinical_scaled[col].std()
        print(f"    {col:<25} mean={mean:>6.3f}  std={std:>6.3f}"
              f"  {'✓' if abs(mean) < 0.01 and abs(std-1) < 0.01 else '⚠'}")

# Build all-features versions (clinical + genomic)
ALL_FEATURES = CLINICAL_FEATURES + GENOMIC_FEATURES

preprocessor_all = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), NUMERIC_TO_SCALE),
    ],
    remainder='passthrough'
)
preprocessor_all.fit(train[ALL_FEATURES])

feature_names_all = (NUMERIC_TO_SCALE +
                     [c for c in ALL_FEATURES if c not in NUMERIC_TO_SCALE])

X_train_all = pd.DataFrame(
    preprocessor_all.transform(train[ALL_FEATURES]),
    columns=feature_names_all, index=train.index)
X_val_all   = pd.DataFrame(
    preprocessor_all.transform(val[ALL_FEATURES]),
    columns=feature_names_all, index=val.index)
X_test_all  = pd.DataFrame(
    preprocessor_all.transform(test[ALL_FEATURES]),
    columns=feature_names_all, index=test.index)

# Target arrays
y_train = train[TARGET_ENC].values
y_val   = val[TARGET_ENC].values
y_test  = test[TARGET_ENC].values

# ================================================================================
# SECTION 5: Save Everything
# ================================================================================
print(f"\n{'='*50}")
print("  SAVING PREPROCESSED DATA")
print(f"{'='*50}")

# Save preprocessors
joblib.dump(preprocessor,     SPLITS_DIR / 'preprocessor_clinical.pkl')
joblib.dump(preprocessor_all, SPLITS_DIR / 'preprocessor_all.pkl')

# Save preprocessed arrays
X_train_clinical_scaled.to_csv(SPLITS_DIR / 'X_train_clinical.csv', index=False)
X_val_clinical_scaled.to_csv(  SPLITS_DIR / 'X_val_clinical.csv',   index=False)
X_test_clinical_scaled.to_csv( SPLITS_DIR / 'X_test_clinical.csv',  index=False)

X_train_all.to_csv(SPLITS_DIR / 'X_train_all.csv', index=False)
X_val_all.to_csv(  SPLITS_DIR / 'X_val_all.csv',   index=False)
X_test_all.to_csv( SPLITS_DIR / 'X_test_all.csv',  index=False)

np.save(SPLITS_DIR / 'y_train.npy', y_train)
np.save(SPLITS_DIR / 'y_val.npy',   y_val)
np.save(SPLITS_DIR / 'y_test.npy',  y_test)

# Save metadata
meta = {
    'random_state'        : RANDOM_STATE,
    'n_train'             : len(train),
    'n_val'               : len(val),
    'n_test'              : len(test),
    'clinical_features'   : CLINICAL_FEATURES,
    'genomic_features'    : GENOMIC_FEATURES,
    'all_features'        : ALL_FEATURES,
    'target'              : TARGET,
    'target_enc'          : TARGET_ENC,
    'label_map'           : LABEL_MAP,
    'numeric_scaled'      : NUMERIC_TO_SCALE,
}
with open(SPLITS_DIR / 'split_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n  Files saved to data/splits/:")
for f in sorted(SPLITS_DIR.iterdir()):
    size = f.stat().st_size / 1024
    print(f"    {f.name:<40} {size:>8.1f} KB")

print(f"\n{'='*50}")
print(f"✓ Data splitting complete")
print(f"{'='*50}")
print(f"\n  Train : {len(train):,} patients  →  X_train_clinical, X_train_all")
print(f"  Val   : {len(val):,} patients  →  X_val_clinical,   X_val_all")
print(f"  Test  : {len(test):,} patients  →  X_test_clinical,  X_test_all")
print(f"\n  Next: run 02_clinical_only_baseline.py")
