"""
================================================================================
Phase 2 — Exploratory Data Analysis (EDA)
Notebook 01: Descriptive Statistics
================================================================================

GOAL:
    Before building any model, we need to understand the structure of our data.
    This notebook answers three fundamental questions:
        1. What does our data look like numerically?
        2. How is each feature distributed?
        3. Where should we pay extra attention before modeling?
================================================================================
"""

# ================================================================================
# SECTION 1: Imports and Setup
# ================================================================================

# pandas: the main library for working with tabular data (like Excel, but in Python)
import pandas as pd

# numpy: the core library for numerical operations
import numpy as np

# matplotlib + seaborn: plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# pathlib.Path: cross-platform file path handling
from pathlib import Path

# suppress non-critical warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')

# ── Display Settings ──────────────────────────────────────────────────────────

# Always show ALL columns, never hide any with "..."
pd.set_option('display.max_columns', None)

# Round all floats to 3 decimal places for cleaner output
# '{:.3f}'.format means: decimal point, 3 places, float
pd.set_option('display.float_format', '{:.3f}'.format)

# Clean white background with light gray grid lines for all plots
plt.style.use('seaborn-v0_8-whitegrid')

print("✓ Libraries loaded")


# ================================================================================
# SECTION 2: Load Data
# ================================================================================

# '../../' navigates up two folders from notebooks/01_eda/ to project root
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
FIG_PATH  = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/figures/eda/')

# Create the figures/eda/ folder if it doesn't exist yet
# parents=True  → also create any missing parent folders
# exist_ok=True → don't throw an error if the folder already exists
FIG_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print(f"\n{'='*50}")
print(f"  Dataset loaded")
print(f"{'='*50}")
print(f"  Patients : {len(df):,}")        # {:,} adds thousands separator → 6,110
print(f"  Columns  : {df.shape[1]}")      # df.shape = (rows, columns) → we take [1]
print(f"  Memory   : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
# memory_usage(deep=True) = actual RAM used; divide by 1024² to convert bytes → MB


# ================================================================================
# SECTION 3: Define Feature Groups
# ================================================================================
# We split features into two groups because they need different analysis methods:
#   NUMERIC     → continuous numbers → calculate mean, std, skewness
#   CATEGORICAL → labels or binary flags → count occurrences

# ── Numeric Features ──────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'CURRENT_AGE_DEID',  # Patient age in years
    'TMB_NONSYNONYMOUS', # Tumor Mutational Burden: number of mutations per megabase
                         # Higher TMB often predicts better immunotherapy response
    'TMB_LOG',           # TMB after log-transform — less skewed, easier for models
    'MSI_SCORE',         # Microsatellite Instability score
                         # High MSI = DNA repair defect = may respond well to immunotherapy
    'MSI_NORM',          # MSI after normalization
    'TUMOR_PURITY',      # Fraction of actual tumor cells in the sample (0.0 to 1.0)
                         # Low purity = sample contains lots of normal (non-tumor) cells
    'OS_MONTHS',         # Overall Survival: how many months the patient survived
]

# ── Categorical Features ──────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    'GENDER_ENC',             # Gender encoded: 0=Female, 1=Male
    'TREATMENT_FINAL',        # Treatment received: Immunotherapy / Chemotherapy / Targeted
    'TREATMENT_ENC',          # Same but as integer: 0=Immunotherapy, 1=Chemo, 2=Targeted
    'STAGE_ENC',              # Cancer stage encoded: 0=Stage 1-3, 1=Stage 4
    'SMOKING_ENC',            # Smoking history encoded: 0=Never, 1=Former/Current
    'PRIOR_MED_ENC',          # Prior treatment before MSK: 0=None, 0.5=Unknown, 1=Yes
                              # Patients with prior treatment may have drug-resistant tumors
    'HAS_PROGRESSION',        # 1=disease progression was recorded, 0=no progression
                              # 86% of patients had progression — important context for treatment
    'SUBTYPE_Adenocarcinoma', # 1 if Adenocarcinoma subtype, 0 otherwise
                              # Most common NSCLC subtype — often EGFR/KRAS driven
    'SUBTYPE_Squamous',       # 1 if Squamous Cell Carcinoma, 0 otherwise
                              # Strongly linked to smoking history
    'SUBTYPE_Neuroendocrine', # 1 if Neuroendocrine subtype, 0 otherwise
                              # Aggressive subtype, often treated differently
    'SUBTYPE_Other_NSCLC',    # 1 if other NSCLC subtype, 0 otherwise
    'BONE',                   # 1 if bone metastasis present, 0 otherwise
    'CNS_BRAIN',              # 1 if brain/CNS metastasis present, 0 otherwise
                              # CNS metastasis often limits treatment options
    'LIVER',                  # 1 if liver metastasis present, 0 otherwise
    'LUNG',                   # 1 if lung metastasis present, 0 otherwise
    'LYMPH_NODES',            # 1 if lymph node involvement present, 0 otherwise
    'OS_EVENT',               # 1=patient died during follow-up, 0=censored (still alive
                              # or lost to follow-up at end of study)
]

# Safety filter: only keep names that actually exist in our DataFrame
# List comprehension: [item for item in list if condition]
# This prevents KeyError if any column name doesn't match exactly
NUMERIC_COLS     = [c for c in NUMERIC_COLS     if c in df.columns]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

print(f"\n  Numeric features    : {len(NUMERIC_COLS)}")
print(f"  Categorical features: {len(CATEGORICAL_COLS)}")


# ================================================================================
# SECTION 4: Descriptive Statistics — Numeric Features
# ================================================================================
print(f"\n{'='*60}")
print("  NUMERIC FEATURES — DESCRIPTIVE STATISTICS")
print(f"{'='*60}")

rows = []  # collect one dict per feature, then convert to DataFrame at the end

for col in NUMERIC_COLS:

    # df[col] selects the column as a pandas Series
    # .dropna() removes NaN values before calculating — avoids incorrect stats
    s = df[col].dropna()

    # ── Quartiles ──
    # quantile(0.25) = Q1: 25% of patients fall below this value
    # quantile(0.75) = Q3: 75% of patients fall below this value
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    # IQR = Q3 - Q1: the range of the middle 50% of patients
    # More robust than std because it ignores extreme outliers
    # Example: Q1=40, Q3=70 → IQR=30 → middle 50% spans 30 years
    iqr = q3 - q1

    # Skewness: measures how asymmetric the distribution is
    #   skew ≈  0  → symmetric (bell curve)
    #   skew >  0  → right-skewed: few patients have very HIGH values (long right tail)
    #   skew <  0  → left-skewed: few patients have very LOW values (long left tail)
    skew = s.skew()

    # Interpret skewness using standard statistical thresholds:
    #   |skew| < 0.5  → symmetric, no transformation needed
    #   0.5 – 1.0     → mild skew, transformation may help slightly
    #   > 1.0         → strong skew ⚠ → log-transform strongly recommended
    if abs(skew) < 0.5:
        skew_label = "symmetric"
    elif skew > 1.0:
        skew_label = "right-skewed ⚠"
    elif skew > 0.5:
        skew_label = "mild right-skew"
    elif skew < -1.0:
        skew_label = "left-skewed ⚠"
    else:
        skew_label = "mild left-skew"

    rows.append({
        'Feature'  : col,
        'Mean'     : round(s.mean(), 2),   # arithmetic average
        'Median'   : round(s.median(), 2), # middle value — robust to outliers
        'Std'      : round(s.std(), 2),    # average distance from the mean
        'Min'      : round(s.min(), 2),    # smallest observed value
        'Max'      : round(s.max(), 2),    # largest observed value
        'IQR'      : round(iqr, 2),        # spread of the middle 50%
        'Skewness' : round(skew, 3),       # asymmetry score
        'Shape'    : skew_label,           # human-readable interpretation
    })

# Convert list of dicts → DataFrame, then print without row index
stats_df = pd.DataFrame(rows)
print(stats_df.to_string(index=False))


# ================================================================================
# SECTION 5: Mean vs Median Gap — Detecting Skew and Outliers
# ================================================================================
# KEY CONCEPT:
#   In a perfectly symmetric distribution → mean == median
#   When they diverge significantly, it means:
#     1. The distribution is skewed (long tail pulling mean away from center), OR
#     2. Outliers exist (extreme values dragging the mean up or down)
#
# Example:
#   TMB values: 1, 2, 3, 4, 5      → mean=3.0, median=3.0 → gap=0%   (symmetric)
#   TMB values: 1, 2, 3, 4, 500    → mean=102, median=3.0 → gap=3300% (outlier!)
#
# Rule of thumb: gap > 20% → this feature needs investigation

print(f"\n{'─'*60}")
print("  MEAN vs MEDIAN GAP  (gap > 20% needs attention)")
print(f"{'─'*60}")

for _, row in stats_df.iterrows():
    # Absolute difference between mean and median
    gap = abs(row['Mean'] - row['Median'])

    # Express as percentage of median
    # Adding 1e-9 (= 0.000000001) prevents ZeroDivisionError when median = 0
    gap_pct = gap / (abs(row['Median']) + 1e-9) * 100

    # Flag features where mean and median diverge by more than 20%
    flag = "  ← CHECK" if gap_pct > 20 else ""

    # f-string formatting:
    # :<25  → left-align text in a 25-character wide field
    # :>8.2f → right-align number, 8 chars wide, 2 decimal places
    print(f"  {row['Feature']:<25}  "
          f"mean={row['Mean']:>8.2f}  "
          f"median={row['Median']:>8.2f}  "
          f"gap={gap_pct:>5.1f}%{flag}")


# ================================================================================
# SECTION 6: Descriptive Statistics — Categorical Features
# ================================================================================
print(f"\n{'='*60}")
print("  CATEGORICAL FEATURES — DISTRIBUTION")
print(f"{'='*60}")

for col in CATEGORICAL_COLS:
    if col not in df.columns:
        continue

    # value_counts() → count occurrences of each unique value, sorted most→least
    vc  = df[col].value_counts()

    # normalize=True → returns proportions (0.0–1.0); multiply by 100 → percentages
    pct = df[col].value_counts(normalize=True) * 100

    print(f"\n  {col}:")
    print(f"  {'Value':<20} {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'─'*50}")

    for val in vc.index:
        # Simple text bar chart: each █ block = 2%
        # int() truncates to integer → 43.6% becomes 21 blocks
        bar = '█' * int(pct[val] / 2)
        print(f"  {str(val):<20} {vc[val]:>6}  {pct[val]:>5.1f}%  {bar}")


# ================================================================================
# SECTION 7: Class Balance Check — Target Variable
# ================================================================================
# WHY THIS MATTERS:
#   If 80% of patients received Immunotherapy and only 10% received Targeted,
#   a naive model will learn to almost always predict Immunotherapy and will
#   perform poorly on the minority classes.
#
#   Imbalance ratio = count(largest class) / count(smallest class)
#     ratio ≈ 1.0  → perfectly balanced
#     ratio > 1.5  → mild imbalance → monitor per-class F1 score
#     ratio > 3.0  → severe imbalance → use class_weight='balanced'

print(f"\n{'='*60}")
print("  TARGET VARIABLE — CLASS BALANCE")
print(f"{'='*60}")

target_counts = df['TREATMENT_FINAL'].value_counts()
target_pct    = df['TREATMENT_FINAL'].value_counts(normalize=True) * 100

print(f"\n  Total patients: {len(df):,}")
print()
for cls, cnt in target_counts.items():
    p   = target_pct[cls]
    bar = '█' * int(p / 2)
    print(f"  {str(cls):<20} {cnt:>5,}  ({p:.1f}%)  {bar}")

# Imbalance ratio: how much bigger is the largest class vs the smallest?
ratio = target_counts.max() / target_counts.min()
print(f"\n  Imbalance ratio: {ratio:.2f}x")
if ratio > 3:
    print("  ⚠ Significant imbalance — use class_weight='balanced' in all models")
elif ratio > 1.5:
    print("  ⚡ Mild imbalance — monitor per-class F1, not just overall accuracy")
else:
    print("  ✓ Classes are reasonably balanced")


# ================================================================================
# SECTION 8: Survival Statistics
# ================================================================================
# KEY CONCEPTS:
#   OS_EVENT = 1 → death was observed during the study (we know exactly when)
#   OS_EVENT = 0 → patient was still alive when the study ended, OR was lost
#                  to follow-up. This is called "censored."
#
#   Event rate = proportion of patients where death was observed
#   A higher event rate = more statistical power for survival analysis

print(f"\n{'='*60}")
print("  SURVIVAL STATISTICS")
print(f"{'='*60}")

# .sum() on a binary column counts all 1s = total observed deaths
n_events   = int(df['OS_EVENT'].sum())

# Count rows where OS_EVENT is 0 = censored patients
n_censored = int((df['OS_EVENT'] == 0).sum())

# .mean() on a binary column = proportion of 1s = event rate
event_rate = df['OS_EVENT'].mean() * 100

print(f"\n  Events   (OS=1): {n_events:,}  ({event_rate:.1f}%)")
print(f"  Censored (OS=0): {n_censored:,}  ({100-event_rate:.1f}%)")
print(f"\n  OS_MONTHS:")
print(f"    Mean   : {df['OS_MONTHS'].mean():.1f} months")
print(f"    Median : {df['OS_MONTHS'].median():.1f} months")
print(f"    Range  : {df['OS_MONTHS'].min():.1f} – {df['OS_MONTHS'].max():.1f} months")


# ================================================================================
# SECTION 9: Genomic Features Overview
# ================================================================================
# In our dataset, mutation columns are prefixed with "MUT_"
# Example: MUT_TP53 = 1 if TP53 is mutated in this patient, 0 otherwise
#
# We check:
#   1. How many genomic columns are available?
#   2. What percentage of patients have at least one mutation?
#   3. Which genes are most frequently mutated? (likely most informative)

# Select all columns that start with "MUT_"
genomic_cols = [c for c in df.columns if c.startswith('MUT_')]

if genomic_cols:
    # .sum(axis=1) sums across columns for each row
    # axis=1 means "sum horizontally (across columns)" for each patient
    # Result: each patient gets their total mutation count
    # > 0 means at least one mutation was detected
    any_mutation = (df[genomic_cols].sum(axis=1) > 0).sum()

    print(f"\n{'='*60}")
    print("  GENOMIC FEATURES OVERVIEW")
    print(f"{'='*60}")
    print(f"\n  Genomic columns found : {len(genomic_cols)}")
    print(f"  Patients with ≥1 mutation: {any_mutation:,} "
          f"({any_mutation/len(df)*100:.1f}%)")

    # .mean() on binary column = mutation frequency per gene
    # .sort_values(ascending=False) → most mutated gene first
    mutation_freq = df[genomic_cols].mean().sort_values(ascending=False)

    print(f"\n  Top 10 most frequently mutated genes:")
    print(f"  {'Gene':<15} {'Freq':>6}  Bar")
    print(f"  {'─'*40}")
    for gene, freq in mutation_freq.head(10).items():
        # Remove "MUT_" prefix for cleaner display
        gene_name = gene.replace('MUT_', '')
        # Scale bar: freq=0.5 (50%) → 15 blocks
        bar = '█' * int(freq * 30)
        print(f"  {gene_name:<15} {freq*100:>5.1f}%  {bar}")


# ================================================================================
# SECTION 10: Summary and Decisions for Next Steps
# ================================================================================
print(f"\n{'='*60}")
print("  SUMMARY & DECISIONS FOR NEXT STEPS")
print(f"{'='*60}")
print("""
  After reviewing the output above, answer these questions
  before moving to notebook 02 (Distribution Analysis):

  [ ] Which numeric features show skewness > 1.0?  (marked with ⚠)
      → Those need log-transform before modeling
      → Note: TMB_LOG already exists — check if it's symmetric

  [ ] Which features have a Mean vs Median gap > 20%?  (marked ← CHECK)
      → Investigate for outliers in notebook 02

  [ ] Is the dataset balanced across treatment classes?
      → If imbalance ratio > 1.5, note which class is smallest
      → We will use class_weight='balanced' in Phase 3 models

  [ ] What is the event rate in OS_EVENT?
      → This tells us how much power we have for Kaplan-Meier in notebook 05

  [ ] Which genes are mutated most frequently?
      → These will likely receive the highest attention weights
        in our cross-modal architecture (Phase 4)
""")
