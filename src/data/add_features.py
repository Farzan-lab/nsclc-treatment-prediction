"""
================================================================================
Preprocessing — Encode New Features + Impute Missing Values
================================================================================

WHAT THIS SCRIPT DOES:
    Reads nsclc_final.csv, performs these operations, and saves the result:

    1. ENCODE  → PRIOR_MED_TO_MSK → PRIOR_MED_ENC     (3 categories → 0 / 0.5 / 1)
    2. IMPUTE  → MSI_SCORE         → fill missing values with median
    3. IMPUTE  → HAS_PROGRESSION   → fill missing values with mode
    4. CREATE  → MSI_LOG           → log1p(MSI_NORM) for modeling
    5. ENCODE  → PDL1_POSITIVE     → PDL1_ENC + PDL1_TESTED (0/0.5/1 + missing flag)
    6. ENCODE  → HISTORY_OF_PDL1   → HISTORY_PDL1_ENC  (0/0.5/1)

RUN ONCE from the project root before starting any EDA notebook:
    python src/data/add_features.py
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
df = pd.read_csv(DATA_PATH)

original_cols = len(df.columns)
print(f"Loaded: {df.shape[0]:,} patients, {df.shape[1]} columns")

# ================================================================================
# STEP 1: ENCODE — PRIOR_MED_TO_MSK → PRIOR_MED_ENC
# ================================================================================
# Did this patient receive cancer treatment BEFORE coming to MSK?
#
# ORIGINAL VALUES (verified from dataset):
#   'No prior medications'     → no prior cancer treatment
#   'Prior medications to MSK' → received treatment elsewhere before MSK
#   'Unknown'                  → information not available
#
# ENCODING:
#   'No prior medications'     → 0.0  (clean slate)
#   'Prior medications to MSK' → 1.0  (has prior treatment)
#   'Unknown'                  → 0.5  (uncertainty preserved as neutral midpoint)
#
# WHY 0.5 for Unknown:
#   Mode imputation would silently assume "no prior treatment."
#   Using 0.5 tells the model: "we genuinely do not know."

print(f"\n{'='*50}")
print("  STEP 1: PRIOR_MED_TO_MSK → PRIOR_MED_ENC")
print(f"{'='*50}")

print(f"\nOriginal distribution:")
print(df['PRIOR_MED_TO_MSK'].value_counts(dropna=False).to_string())

prior_med_map = {
    'No prior medications':     0.0,
    'Prior medications to MSK': 1.0,   # actual string in dataset
    'Prior medications':        1.0,   # fallback variant
    'Unknown':                  0.5,
}

df['PRIOR_MED_ENC'] = df['PRIOR_MED_TO_MSK'].map(prior_med_map)

n_unmapped = df['PRIOR_MED_ENC'].isna().sum()
if n_unmapped > 0:
    df['PRIOR_MED_ENC'] = df['PRIOR_MED_ENC'].fillna(0.5)
    print(f"\n⚠ {n_unmapped} unexpected values → filled with 0.5")
else:
    print(f"\n✓ All values mapped successfully")

print(f"\n  0.0 = No prior medications : {(df['PRIOR_MED_ENC']==0.0).sum():,} patients")
print(f"  0.5 = Unknown              : {(df['PRIOR_MED_ENC']==0.5).sum():,} patients")
print(f"  1.0 = Prior medications    : {(df['PRIOR_MED_ENC']==1.0).sum():,} patients")
print(f"✓ PRIOR_MED_ENC created")


# ================================================================================
# STEP 2: IMPUTE — MSI_SCORE (missing values → median)
# ================================================================================
# MSI (Microsatellite Instability) measures DNA repair defects.
# High MSI = broken DNA repair → often responds well to immunotherapy.
#
# WHY MEDIAN:
#   MSI has extreme outliers — median is more robust than mean.
#   We do NOT use 0 because MSI=0 has a specific clinical meaning (stable).

print(f"\n{'='*50}")
print("  STEP 2: IMPUTE — MSI_SCORE")
print(f"{'='*50}")

n_missing = df['MSI_SCORE'].isna().sum()
print(f"\nMissing values: {n_missing} ({n_missing/len(df)*100:.1f}%)")

msi_median = df['MSI_SCORE'].median()
df['MSI_SCORE'] = df['MSI_SCORE'].fillna(msi_median)

print(f"Imputed with median = {msi_median:.4f}")
print(f"Missing after: {df['MSI_SCORE'].isna().sum()}")
print(f"✓ MSI_SCORE imputed")


# ================================================================================
# STEP 3: IMPUTE — HAS_PROGRESSION (missing values → mode)
# ================================================================================
# Binary flag: 1 = disease progression recorded, 0 = no progression.
# Only 0.5% missing — mode imputation introduces negligible bias.

print(f"\n{'='*50}")
print("  STEP 3: IMPUTE — HAS_PROGRESSION")
print(f"{'='*50}")

n_missing = df['HAS_PROGRESSION'].isna().sum()
print(f"\nMissing values: {n_missing} ({n_missing/len(df)*100:.1f}%)")

mode_value = df['HAS_PROGRESSION'].mode()[0]
df['HAS_PROGRESSION'] = df['HAS_PROGRESSION'].fillna(mode_value)

print(f"Imputed with mode = {mode_value}")
print(f"Missing after: {df['HAS_PROGRESSION'].isna().sum()}")
print(f"✓ HAS_PROGRESSION imputed")


# ================================================================================
# STEP 4: CREATE — MSI_LOG = log1p(MSI_NORM)
# ================================================================================
# From distribution analysis, MSI_NORM is heavily skewed (skew=7.52).
# A few MSI-High patients have very large values that can dominate the model.
#
# Solution: log1p transform compresses the scale without losing signal.
#   log1p(x) = log(x + 1)
#
# WHY log1p instead of log:
#   MSI_NORM has many 0.0 values (stable MSS patients).
#   log(0) = -infinity → breaks the model.
#   log1p(0) = log(1) = 0 → safe.
#
# This does NOT remove MSI-High patients — it proportionally compresses
# the scale so high values don't dominate over other features.

print(f"\n{'='*50}")
print("  STEP 4: CREATE MSI_LOG")
print(f"{'='*50}")

df['MSI_LOG'] = np.log1p(df['MSI_NORM'])

print(f"\n  MSI_NORM → skewness: {df['MSI_NORM'].skew():.3f}")
print(f"  MSI_LOG  → skewness: {df['MSI_LOG'].skew():.3f}")
print(f"  MSI_LOG  → range   : [{df['MSI_LOG'].min():.3f}, {df['MSI_LOG'].max():.3f}]")
print(f"  MSI_LOG  → missing : {df['MSI_LOG'].isna().sum()}")
print(f"✓ MSI_LOG created")


# ================================================================================
# STEP 5: ENCODE — PDL1_POSITIVE → PDL1_ENC + PDL1_TESTED
# ================================================================================
# PD-L1 IHC status is the primary clinical biomarker used to decide
# immunotherapy eligibility in NSCLC, and it is measured during diagnostic
# workup — i.e. it is known BEFORE the treatment decision, so using it is
# not leakage. 57% missing is why it was dropped previously, but the
# diagnostic crosstab shows real signal on exactly the axis the model
# struggles with (Immunotherapy vs Chemotherapy):
#   PDL1 positive → Immunotherapy 56.2% of the time
#   PDL1 negative → Immunotherapy 43.2% of the time
#
# Rather than drop it, encode missingness explicitly (same 0/0.5/1 pattern
# as PRIOR_MED_ENC) plus a separate "was it tested" indicator, so the model
# can use the value when known and use "untested" itself as a signal.

print(f"\n{'='*50}")
print("  STEP 5: PDL1_POSITIVE → PDL1_ENC + PDL1_TESTED")
print(f"{'='*50}")

n_missing = df['PDL1_POSITIVE'].isna().sum()
print(f"\nMissing values: {n_missing} ({n_missing/len(df)*100:.1f}%)")

pdl1_map = {'Yes': 1.0, 'No': 0.0}
df['PDL1_ENC'] = df['PDL1_POSITIVE'].map(pdl1_map).fillna(0.5)
df['PDL1_TESTED'] = df['PDL1_POSITIVE'].notna().astype(float)

print(f"  1.0 = Positive : {(df['PDL1_ENC']==1.0).sum():,} patients")
print(f"  0.5 = Unknown  : {(df['PDL1_ENC']==0.5).sum():,} patients")
print(f"  0.0 = Negative : {(df['PDL1_ENC']==0.0).sum():,} patients")
print(f"✓ PDL1_ENC, PDL1_TESTED created")

# ================================================================================
# STEP 5b: ENCODE — HISTORY_OF_PDL1 → HISTORY_PDL1_ENC
# ================================================================================
# Whether a PD-L1 test was ever ordered for the patient — a weaker but
# more complete (79% non-missing) proxy for the same clinical intent.

print(f"\n{'='*50}")
print("  STEP 5b: HISTORY_OF_PDL1 → HISTORY_PDL1_ENC")
print(f"{'='*50}")

n_missing = df['HISTORY_OF_PDL1'].isna().sum()
print(f"\nMissing values: {n_missing} ({n_missing/len(df)*100:.1f}%)")

df['HISTORY_PDL1_ENC'] = df['HISTORY_OF_PDL1'].map(pdl1_map).fillna(0.5)
print(f"✓ HISTORY_PDL1_ENC created")

# ================================================================================
# STEP 5c: ENCODE — remaining metastasis site flags (Yes/No/Unknown → 1/0/0.5)
# ================================================================================
# BONE, CNS_BRAIN, LIVER, LUNG, LYMPH_NODES already arrive as 0/1 floats.
# These five sites use the same Yes/No/Unknown text as PRIOR_MED_TO_MSK and
# were left out of the model so far — no reason to keep dropping them.

print(f"\n{'='*50}")
print("  STEP 5c: Encode remaining metastasis site flags")
print(f"{'='*50}")

met_site_map = {'Yes': 1.0, 'No': 0.0, 'Unknown': 0.5}
for col in ['ADRENAL_GLANDS', 'INTRA_ABDOMINAL', 'OTHER', 'PLEURA', 'REPRODUCTIVE_ORGANS']:
    if df[col].dtype == object:
        df[col] = df[col].map(met_site_map).fillna(0.5)
    print(f"  {col:<22} → {sorted(df[col].dropna().unique())}")
print(f"✓ Metastasis site flags encoded")


# ================================================================================
# STEP 6: Verification
# ================================================================================
print(f"\n{'='*50}")
print("  STEP 6: VERIFICATION")
print(f"{'='*50}")

check_cols = ['PRIOR_MED_ENC', 'MSI_SCORE', 'HAS_PROGRESSION', 'MSI_LOG',
              'PDL1_ENC', 'PDL1_TESTED', 'HISTORY_PDL1_ENC',
              'ADRENAL_GLANDS', 'INTRA_ABDOMINAL', 'OTHER', 'PLEURA', 'REPRODUCTIVE_ORGANS']
for col in check_cols:
    n_miss   = df[col].isna().sum()
    status   = "✓" if n_miss == 0 else "✗"
    unique   = sorted(df[col].dropna().unique())
    u_str    = str(unique[:4]) + (' ...' if len(unique) > 4 else '')
    print(f"\n  {status} {col}")
    print(f"    missing: {n_miss}")
    print(f"    dtype  : {df[col].dtype}")
    print(f"    unique : {u_str}")


# ================================================================================
# STEP 7: Save
# ================================================================================
df.to_csv(DATA_PATH, index=False)

added = len(df.columns) - original_cols
print(f"\n{'='*50}")
print(f"✓ Saved")
print(f"{'='*50}")
print(f"  Path    : {DATA_PATH}")
print(f"  Patients: {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}  (+{added} new)")
print(f"\n  New columns: PRIOR_MED_ENC, MSI_LOG, PDL1_ENC, PDL1_TESTED, HISTORY_PDL1_ENC")
print(f"  Imputed   : MSI_SCORE, HAS_PROGRESSION")
print(f"\n  Ready for data validation ✓")
