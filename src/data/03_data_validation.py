"""
================================================================================
Preprocessing — Data Validation and Cleaning
================================================================================

WHAT THIS SCRIPT DOES:
    Checks features for logically impossible values and fixes them.

DECISIONS:
    AGE < 0          → remove entire row (impossible value)
    TMB, MSI         → no action (extreme values may be real edge cases)
    TUMOR_PURITY     → clip to [0, 100]
    OS_MONTHS        → clip to [0, 300]
    Binary features  → validate only, no fix unless critical
    MSI_LOG          → add new column: log1p(MSI_NORM)

RUN ORDER:
    1. src/data/add_features.py       ← encoding + imputation
    2. src/data/03_data_validation.py ← this file
    3. notebooks/01_eda/...           ← EDA notebooks
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
df = pd.read_csv(DATA_PATH)

original_rows = len(df)
print(f"Loaded: {df.shape[0]:,} patients, {df.shape[1]} columns")

issues = []

# ================================================================================
# SECTION 1: AGE — Remove rows with impossible values
# ================================================================================
# Age below 0 is physically impossible — a data entry error.
# Age = 0 is also impossible for a cancer patient.
# Decision: REMOVE the entire row, not just fix the value.
#
# Why remove instead of fix?
#   We cannot guess the correct age for these patients.
#   Imputing with median would introduce a false data point.
#   Since we don't know the true age, the entire record is unreliable.

print(f"\n{'='*60}")
print("  SECTION 1: AGE VALIDATION")
print(f"{'='*60}")

print(f"\n  CURRENT_AGE_DEID:")
print(f"    Range before: [{df['CURRENT_AGE_DEID'].min():.1f}, "
      f"{df['CURRENT_AGE_DEID'].max():.1f}]")

# Count rows with invalid age
n_negative_age = (df['CURRENT_AGE_DEID'] <= 0).sum()
n_null_age     = df['CURRENT_AGE_DEID'].isna().sum()
n_invalid_age  = n_negative_age + n_null_age

print(f"    Age <= 0    : {n_negative_age} patients")
print(f"    Age is NaN  : {n_null_age} patients")
print(f"    Total to remove: {n_invalid_age} patients")

if n_invalid_age > 0:
    # Show the invalid ages before removing
    invalid_ages = df[df['CURRENT_AGE_DEID'] <= 0]['CURRENT_AGE_DEID'].tolist()
    if invalid_ages:
        print(f"    Invalid age values: {invalid_ages}")

    # Remove rows where age is <= 0 or NaN
    # ~ means NOT — so ~(condition) keeps rows where condition is False
    df = df[~(df['CURRENT_AGE_DEID'] <= 0)].copy()
    df = df[df['CURRENT_AGE_DEID'].notna()].copy()

    rows_removed = original_rows - len(df)
    print(f"\n    ✓ Removed {rows_removed} rows with invalid age")
    print(f"    Patients remaining: {len(df):,}")
    issues.append({
        'feature': 'CURRENT_AGE_DEID',
        'action': 'removed rows',
        'n_affected': rows_removed
    })
else:
    print(f"    ✓ No invalid age values found")

print(f"\n    Range after: [{df['CURRENT_AGE_DEID'].min():.1f}, "
      f"{df['CURRENT_AGE_DEID'].max():.1f}]")

# ================================================================================
# SECTION 2: TMB — No Action
# ================================================================================
# TMB_NONSYNONYMOUS has extreme values (up to 167 in our dataset).
# Decision: DO NOT clip or remove.
#
# Reason: Extreme TMB values are clinically real.
# Hypermutated tumors (TMB > 100) are documented in NSCLC and often
# respond exceptionally well to immunotherapy.
# Removing or capping these values would lose important signal.
# We already have TMB_LOG which handles the scale issue for modeling.

print(f"\n{'='*60}")
print("  SECTION 2: TMB — NO ACTION")
print(f"{'='*60}")
print(f"\n  TMB_NONSYNONYMOUS:")
print(f"    Range  : [{df['TMB_NONSYNONYMOUS'].min():.2f}, "
      f"{df['TMB_NONSYNONYMOUS'].max():.2f}]")
print(f"    Decision: KEEP as-is — extreme values are clinically real")
print(f"    Modeling: use TMB_LOG (log-transformed) instead of raw TMB")

# ================================================================================
# SECTION 3: MSI — No Action
# ================================================================================
# MSI_SCORE has extreme values and -1 sentinel values.
# Decision: DO NOT clip or remove.
#
# Reason: High MSI scores (MSI-High tumors) are rare but real.
# MSI-High is actually a strong predictor of immunotherapy response.
# Removing high MSI patients would lose the most clinically interesting cases.
# The -1 sentinel value is already handled by MSI_NORM (clipped to 0).
# We will add MSI_LOG = log1p(MSI_NORM) for modeling.

print(f"\n{'='*60}")
print("  SECTION 3: MSI — NO ACTION")
print(f"{'='*60}")
print(f"\n  MSI_SCORE:")
print(f"    Range  : [{df['MSI_SCORE'].min():.2f}, {df['MSI_SCORE'].max():.2f}]")
n_sentinel = (df['MSI_SCORE'] < 0).sum()
print(f"    Sentinel values (-1): {n_sentinel} patients")
print(f"    Decision: KEEP as-is — high MSI = clinically important signal")
print(f"    Modeling: use MSI_LOG = log1p(MSI_NORM)")

# ================================================================================
# SECTION 4: TUMOR_PURITY — Clip to [0, 100]
# ================================================================================
# Tumor purity is a percentage — physically cannot be below 0 or above 100.
# Unlike TMB and MSI, there is no clinical scenario where purity > 100%.
# A value outside [0, 100] is purely a data entry or measurement error.

print(f"\n{'='*60}")
print("  SECTION 4: TUMOR_PURITY VALIDATION")
print(f"{'='*60}")

col = 'TUMOR_PURITY'
n_below = (df[col] < 0).sum()
n_above = (df[col] > 100).sum()

print(f"\n  {col}:")
print(f"    Range before: [{df[col].min():.1f}, {df[col].max():.1f}]")
print(f"    Below 0     : {n_below} patients")
print(f"    Above 100   : {n_above} patients")

if n_below + n_above > 0:
    df[col] = df[col].clip(lower=0, upper=100)
    print(f"    ✓ Clipped to [0, 100]")
    issues.append({'feature': col, 'action': 'clipped [0,100]',
                   'n_affected': n_below + n_above})
else:
    print(f"    ✓ All values in valid range [0, 100]")

print(f"    Range after : [{df[col].min():.1f}, {df[col].max():.1f}]")

# ================================================================================
# SECTION 5: OS_MONTHS — Clip negative values
# ================================================================================
# Time cannot be negative — OS_MONTHS below 0 is impossible.
# We clip to 0 (not remove) because the patient record may still be valid
# apart from this one field.

print(f"\n{'='*60}")
print("  SECTION 5: OS_MONTHS VALIDATION")
print(f"{'='*60}")

col = 'OS_MONTHS'
n_negative = (df[col] < 0).sum()

print(f"\n  {col}:")
print(f"    Range before: [{df[col].min():.3f}, {df[col].max():.3f}]")
print(f"    Below 0     : {n_negative} patients")

if n_negative > 0:
    df[col] = df[col].clip(lower=0)
    print(f"    ✓ Clipped negative values to 0")
    issues.append({'feature': col, 'action': 'clipped to 0',
                   'n_affected': n_negative})
else:
    print(f"    ✓ No negative values")

# Check OS_MONTHS = 0 with OS_EVENT = 1 (died at exactly time 0)
died_at_zero = ((df['OS_MONTHS'] == 0) & (df['OS_EVENT'] == 1)).sum()
print(f"\n  OS_MONTHS=0 AND OS_EVENT=1: {died_at_zero} patients")
if died_at_zero > 0:
    # Set to 0.1 months (~3 days) — clinically plausible minimum
    mask = (df['OS_MONTHS'] == 0) & (df['OS_EVENT'] == 1)
    df.loc[mask, 'OS_MONTHS'] = 0.1
    print(f"    ✓ Set to 0.1 months (clinically plausible minimum)")
    issues.append({'feature': 'OS_MONTHS', 'action': 'set 0→0.1 for events',
                   'n_affected': died_at_zero})
else:
    print(f"    ✓ No patients with OS_MONTHS=0 and OS_EVENT=1")

# ================================================================================
# SECTION 6: Binary Features — Validate Only
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 6: BINARY FEATURES VALIDATION")
print(f"{'='*60}")

BINARY_COLS = [
    'GENDER_ENC', 'STAGE_ENC', 'OS_EVENT', 'HAS_PROGRESSION',
    'BONE', 'CNS_BRAIN', 'LIVER', 'LUNG', 'LYMPH_NODES',
]

all_binary_ok = True
for col in BINARY_COLS:
    if col not in df.columns:
        continue
    unique_vals  = set(df[col].dropna().unique())
    valid_vals   = {0, 1, 0.0, 1.0}
    invalid_vals = unique_vals - valid_vals
    if invalid_vals:
        print(f"\n  ⚠ {col}: unexpected values {invalid_vals}")
        all_binary_ok = False
    else:
        print(f"  ✓ {col}: {sorted(unique_vals)}")

if all_binary_ok:
    print(f"\n  ✓ All binary features valid")

# ================================================================================
# SECTION 7: Logical Consistency Checks
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 7: LOGICAL CONSISTENCY CHECKS")
print(f"{'='*60}")

# Check: each patient should have exactly one subtype
subtype_cols = ['SUBTYPE_Adenocarcinoma', 'SUBTYPE_Squamous',
                'SUBTYPE_Neuroendocrine', 'SUBTYPE_Other_NSCLC']
n_subtypes = df[subtype_cols].astype(int).sum(axis=1)

no_subtype    = (n_subtypes == 0).sum()
multi_subtype = (n_subtypes > 1).sum()

print(f"\n  Subtype consistency (each patient = exactly 1 subtype):")
print(f"    No subtype assigned    : {no_subtype} patients")
print(f"    Multiple subtypes      : {multi_subtype} patients")
if no_subtype == 0 and multi_subtype == 0:
    print(f"    ✓ Every patient has exactly 1 subtype")
else:
    print(f"    ⚠ Inconsistency — investigate")

# ================================================================================
# SECTION 8: Add MSI_LOG
# ================================================================================
# MSI_NORM is still heavily skewed (skew=7.52) after normalization.
# log1p transform compresses the long right tail.
# log1p(x) = log(x + 1) — safe for zero values (log1p(0) = 0)

print(f"\n{'='*60}")
print("  SECTION 8: ADD MSI_LOG")
print(f"{'='*60}")

df['MSI_LOG'] = np.log1p(df['MSI_NORM'])

print(f"\n  MSI_NORM skewness : {df['MSI_NORM'].skew():.3f}")
print(f"  MSI_LOG  skewness : {df['MSI_LOG'].skew():.3f}")
print(f"  MSI_LOG  range    : [{df['MSI_LOG'].min():.3f}, {df['MSI_LOG'].max():.3f}]")
print(f"  ✓ MSI_LOG created")

# ================================================================================
# SECTION 9: Summary and Save
# ================================================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

rows_removed = original_rows - len(df)
print(f"\n  Rows before : {original_rows:,}")
print(f"  Rows removed: {rows_removed}")
print(f"  Rows after  : {len(df):,}")
print(f"\n  Actions taken:")
for issue in issues:
    print(f"    • {issue['feature']:<25} {issue['action']}"
          f" — {issue['n_affected']} affected")
if not issues:
    print(f"    ✓ No issues found")

# Final missing value check
missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"\n  Missing values after validation: "
      f"{'none ✓' if len(missing) == 0 else ''}")
for col, n in missing.items():
    print(f"    ⚠ {col}: {n}")

# Save
df.to_csv(DATA_PATH, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved")
print(f"{'='*60}")
print(f"  Path    : {DATA_PATH}")
print(f"  Patients: {len(df):,}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Ready for EDA ✓")
