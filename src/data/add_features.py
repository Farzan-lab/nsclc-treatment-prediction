"""
================================================================================
Preprocessing — Encode New Features + Impute Missing Values
================================================================================

WHAT THIS SCRIPT DOES:
    Reads nsclc_final.csv, performs three operations, and saves the result:

    1. ENCODE  → PRIOR_MED_TO_MSK → PRIOR_MED_ENC  (3 categories → 0 / 0.5 / 1)
    2. IMPUTE  → MSI_SCORE         → fill 177 missing values with median
    3. IMPUTE  → HAS_PROGRESSION   → fill 29 missing values with mode

COLUMNS REMOVED FROM PLAN:
    PDL1_POSITIVE → DROPPED (57% missing — too many to be reliable)

RUN THIS ONCE from the project root before starting any EDA notebook:
    cd nsclc-treatment-prediction/
    python add_features.py
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
df = pd.read_csv(DATA_PATH)

original_cols = df.shape[1]
print(f"Loaded: {df.shape[0]:,} patients, {df.shape[1]} columns")

# ================================================================================
# STEP 1: ENCODE — PRIOR_MED_TO_MSK → PRIOR_MED_ENC
# ================================================================================
# CLINICAL MEANING:
#   Did this patient receive cancer treatment BEFORE coming to MSK?
#   This is important because:
#     - Prior chemotherapy may cause drug resistance
#     - Prior immunotherapy can cause tolerance (reduced response)
#     - Treatment-naive patients have more options available
#
# ORIGINAL VALUES:
#   'No prior medications'  → patient arrived without prior cancer treatment
#   'Prior medications'     → patient received treatment elsewhere before MSK
#   'Unknown'               → information not available in the record
#
# ENCODING DECISION:
#   'No prior medications' → 0.0   clean slate, no prior treatment
#   'Prior medications'    → 1.0   has received prior treatment
#   'Unknown'              → 0.5   uncertainty preserved as a neutral midpoint
#
# WHY 0.5 FOR UNKNOWN instead of mode imputation?
#   Mode imputation silently assumes "no prior treatment" for all unknowns.
#   This would introduce a false clinical assumption for many patients.
#   Using 0.5 tells the model: "we genuinely do not know" — which is
#   different from "we know there was no prior treatment."
#   The model can learn that unknown prior treatment is its own signal.

print(f"\n{'='*50}")
print("  STEP 1: PRIOR_MED_TO_MSK → PRIOR_MED_ENC")
print(f"{'='*50}")

# Show the original distribution before encoding
print(f"\nOriginal distribution:")
print(df['PRIOR_MED_TO_MSK'].value_counts(dropna=False).to_string())

# Define the mapping from string categories to numeric values
prior_med_map = {
    'No prior medications': 0.0,
    'Prior medications':    1.0,
    'Unknown':              0.5,
}

# .map() replaces each string value with its numeric equivalent
# Any value NOT in the dictionary becomes NaN
df['PRIOR_MED_ENC'] = df['PRIOR_MED_TO_MSK'].map(prior_med_map)

# Check for any values that didn't match our mapping
n_unmapped = df['PRIOR_MED_ENC'].isna().sum()
if n_unmapped > 0:
    # This would happen if there are unexpected string values in the column
    # We treat them as Unknown → 0.5
    df['PRIOR_MED_ENC'] = df['PRIOR_MED_ENC'].fillna(0.5)
    print(f"\n⚠ {n_unmapped} unexpected values found → filled with 0.5 (Unknown)")

# Verify the result
print(f"\nEncoded distribution (PRIOR_MED_ENC):")
print(df['PRIOR_MED_ENC'].value_counts().to_string())
print(f"\n  Missing values: {df['PRIOR_MED_ENC'].isna().sum()}")
print(f"✓ PRIOR_MED_ENC created successfully")


# ================================================================================
# STEP 2: IMPUTE — MSI_SCORE (177 missing = 2.9%)
# ================================================================================
# WHAT IS MSI_SCORE?
#   Microsatellite Instability score measures DNA repair defects in the tumor.
#   High MSI = the tumor has accumulated many small mutations due to broken
#   DNA repair machinery. MSI-High tumors often respond well to immunotherapy.
#
# WHY MEDIAN IMPUTATION?
#   177 missing values = 2.9% of patients — low enough to impute safely.
#
#   We use MEDIAN instead of MEAN because:
#     - MSI scores can have outliers (a few very high values)
#     - The median is not affected by outliers
#     - The median represents the "typical" patient better than the mean
#
#   We do NOT use 0 because MSI_SCORE = 0 has a specific clinical meaning
#   (perfect microsatellite stability), which would be a false assumption.

print(f"\n{'='*50}")
print("  STEP 2: IMPUTE — MSI_SCORE")
print(f"{'='*50}")

n_missing_msi = df['MSI_SCORE'].isna().sum()
print(f"\nMissing values: {n_missing_msi} ({n_missing_msi/len(df)*100:.1f}%)")

# Calculate median on non-missing values only
msi_median = df['MSI_SCORE'].median()
print(f"Median (non-missing): {msi_median:.4f}")

# fillna() replaces all NaN values with the specified value
df['MSI_SCORE'] = df['MSI_SCORE'].fillna(msi_median)

# Verify — should now be 0
print(f"Missing after imputation: {df['MSI_SCORE'].isna().sum()}")
print(f"✓ MSI_SCORE imputed with median = {msi_median:.4f}")


# ================================================================================
# STEP 3: IMPUTE — HAS_PROGRESSION (29 missing = 0.5%)
# ================================================================================
# WHAT IS HAS_PROGRESSION?
#   Binary flag: did this patient have disease progression recorded?
#   1 = progression was documented (tumor grew or spread despite treatment)
#   0 = no progression recorded
#
# WHY MODE IMPUTATION?
#   Only 29 patients (0.5%) are missing — extremely small number.
#
#   We use MODE (most frequent value) because this is a binary column.
#   Mode imputation for binary columns means: assign the majority class.
#   For 0.5% missing, this introduces negligible bias.
#
#   We do NOT use 0.5 here (unlike PRIOR_MED_ENC) because:
#     - HAS_PROGRESSION is strictly binary in the original data
#     - 0.5 would be an out-of-distribution value for this feature
#     - With only 29 missing, the imputation choice barely matters

print(f"\n{'='*50}")
print("  STEP 3: IMPUTE — HAS_PROGRESSION")
print(f"{'='*50}")

n_missing_prog = df['HAS_PROGRESSION'].isna().sum()
print(f"\nMissing values: {n_missing_prog} ({n_missing_prog/len(df)*100:.1f}%)")

# Show distribution before imputation
print(f"\nDistribution before imputation:")
print(df['HAS_PROGRESSION'].value_counts(dropna=False).to_string())

# .mode() returns a Series — we take [0] to get the single most frequent value
# For a binary column this will be either 0.0 or 1.0
mode_value = df['HAS_PROGRESSION'].mode()[0]
print(f"\nMode (most frequent value): {mode_value}")

df['HAS_PROGRESSION'] = df['HAS_PROGRESSION'].fillna(mode_value)

print(f"Missing after imputation: {df['HAS_PROGRESSION'].isna().sum()}")
print(f"✓ HAS_PROGRESSION imputed with mode = {mode_value}")


# ================================================================================
# STEP 4: PDL1_POSITIVE — DOCUMENT THE DECISION TO DROP
# ================================================================================
# We are NOT creating PDL1_ENC.
#
# REASON: PDL1_POSITIVE has 3,502 missing values = 57% of all patients.
#
# The rule of thumb for missing data:
#   < 5%   → safe to impute
#   5–20%  → impute with caution, monitor results
#   20–40% → investigate pattern, probably drop
#   > 40%  → drop the column
#
# With 57% missing, any imputation would be inventing data for the majority
# of patients. The model would learn from fabricated values, not real ones.
# Additionally, in real-world inference, PDL1 will often be unavailable —
# making this feature unreliable even if we could impute it here.

print(f"\n{'='*50}")
print("  PDL1_POSITIVE — DROPPED (not encoded)")
print(f"{'='*50}")
print(f"\n  Missing: 3,502 / 6,110 = 57.3%")
print(f"  Decision: DROP — above the 40% threshold for reliable imputation")
print(f"  PDL1_ENC will NOT be added to the dataset")


# ================================================================================
# STEP 5: Final Verification
# ================================================================================
print(f"\n{'='*50}")
print("  FINAL VERIFICATION")
print(f"{'='*50}")

# Check all columns we modified or created
cols_to_check = ['PRIOR_MED_ENC', 'MSI_SCORE', 'HAS_PROGRESSION']

for col in cols_to_check:
    n_missing = df[col].isna().sum()
    unique    = sorted(df[col].dropna().unique())
    # Truncate unique list if too long
    unique_display = unique if len(unique) <= 6 else unique[:3] + ['...'] + unique[-1:]
    status = "✓" if n_missing == 0 else "✗"
    print(f"\n  {status} {col}")
    print(f"    missing : {n_missing}")
    print(f"    unique  : {unique_display}")
    print(f"    dtype   : {df[col].dtype}")


# ================================================================================
# STEP 6: Save
# ================================================================================
df.to_csv(DATA_PATH, index=False)

new_cols = df.shape[1]
added    = new_cols - original_cols

print(f"\n{'='*50}")
print(f"✓ Dataset saved successfully")
print(f"{'='*50}")
print(f"  Path         : {DATA_PATH}")
print(f"  Patients     : {df.shape[0]:,}")
print(f"  Columns      : {new_cols}  (+{added} new: PRIOR_MED_ENC)")
print(f"  MSI_SCORE    : 0 missing  (was 177)")
print(f"  HAS_PROGRESS : 0 missing  (was 29)")
print(f"  PDL1_POSITIVE: not encoded (57% missing — dropped)")
print(f"\n  Ready for EDA ✓")
