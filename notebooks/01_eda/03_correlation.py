"""
================================================================================
Phase 2 — EDA
Notebook 03: Correlation Analysis (Optimized)
================================================================================
"""

# ================================================================================
# SECTION 1: Imports and Setup
# ================================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — faster, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.3f}'.format)

TREATMENT_COLORS = {
    'Immunotherapy': '#2196F3',
    'Chemotherapy':  '#FF9800',
    'Targeted':      '#4CAF50',
}

FIG_PATH = Path('../../figures/eda/')
FIG_PATH.mkdir(parents=True, exist_ok=True)

print("✓ Libraries loaded")

# ================================================================================
# SECTION 2: Load Data
# ================================================================================
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded: {df.shape[0]:,} patients, {df.shape[1]} columns")

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'CURRENT_AGE_DEID', 'TMB_LOG', 'MSI_LOG', 'TUMOR_PURITY',
]

BINARY_CLINICAL = [
    'GENDER_ENC', 'STAGE_ENC', 'SMOKING_ENC', 'HAS_PROGRESSION',
    'SUBTYPE_Adenocarcinoma', 'SUBTYPE_Squamous',
    'SUBTYPE_Neuroendocrine', 'SUBTYPE_Other_NSCLC',
    'BONE', 'CNS_BRAIN', 'LIVER', 'LUNG', 'LYMPH_NODES',
]

TOP_GENOMIC = [
    'MUT_TP53', 'MUT_KRAS', 'MUT_EGFR', 'MUT_KEAP1', 'MUT_STK11',
    'MUT_RBM10', 'MUT_PTPRD', 'MUT_SMARCA4', 'MUT_NF1', 'MUT_CDKN2A',
]

TARGET = 'TREATMENT_ENC'

# Filter to existing columns
NUMERIC_COLS    = [c for c in NUMERIC_COLS    if c in df.columns]
BINARY_CLINICAL = [c for c in BINARY_CLINICAL if c in df.columns]
TOP_GENOMIC     = [c for c in TOP_GENOMIC     if c in df.columns]
ALL_CLINICAL    = NUMERIC_COLS + BINARY_CLINICAL

# Convert bool columns to float once — avoids repeated conversion later
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(float)

print(f"\n  Numeric  : {len(NUMERIC_COLS)}")
print(f"  Binary   : {len(BINARY_CLINICAL)}")
print(f"  Genomic  : {len(TOP_GENOMIC)}")

# ================================================================================
# SECTION 3: Correlation with Target — All Clinical Features
# ================================================================================
# OPTIMIZATION: compute correlation once as a matrix, don't loop with scipy
# df[cols].corr() uses optimized numpy under the hood — much faster than
# calling stats.pearsonr() for each pair individually

print(f"\n{'='*60}")
print("  SECTION 1: CORRELATION WITH TARGET")
print(f"{'='*60}")

cols_with_target = ALL_CLINICAL + [TARGET]
corr_matrix = df[cols_with_target].corr(method='pearson')

# Extract only the correlation with target, sorted by absolute value
target_corr = (corr_matrix[TARGET]
               .drop(TARGET)
               .sort_values(key=abs, ascending=False))

print(f"\n  Clinical features ranked by |correlation| with TREATMENT_ENC:")
print(f"  (+ = associated with Chemo/Targeted, - = associated with Immunotherapy)")
print(f"\n  {'Feature':<30} {'r':>7}  {'|r|':>6}  Signal")
print(f"  {'─'*60}")

for feat, val in target_corr.items():
    abs_val  = abs(val)
    signal   = "strong" if abs_val > 0.3 else "moderate" if abs_val > 0.1 else "weak"
    arrow    = "↑" if val > 0 else "↓"
    print(f"  {feat:<30} {val:>7.3f}  {abs_val:>6.3f}  {arrow} {signal}")

# ================================================================================
# SECTION 4: Numeric Feature Correlation Heatmap (small, fast)
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 2: NUMERIC FEATURES CORRELATION HEATMAP")
print(f"{'='*60}")

numeric_corr = df[NUMERIC_COLS + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(numeric_corr, dtype=bool))
sns.heatmap(
    numeric_corr, mask=mask,
    annot=True, fmt='.2f',
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5,
    annot_kws={'size': 11},
    ax=ax, cbar_kws={'shrink': 0.8}
)
ax.set_title('Numeric Clinical Features + Target', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(FIG_PATH / 'correlation_numeric.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()   # close immediately to free memory — don't call plt.show()
print(f"✓ Saved: correlation_numeric.png")

# ================================================================================
# SECTION 5: Full Clinical Heatmap — WITHOUT annot (much faster)
# ================================================================================
# annot=True on an 18x18 matrix renders 324 text labels — very slow.
# We turn off annotations and use color alone to show the pattern.
# The printed table in Section 1 already gives the exact numbers.

print(f"\n{'='*60}")
print("  SECTION 3: FULL CLINICAL HEATMAP (no annotations for speed)")
print(f"{'='*60}")

full_corr = df[ALL_CLINICAL + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(full_corr, dtype=bool))
sns.heatmap(
    full_corr, mask=mask,
    annot=False,             # ← key optimization: no text labels
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.2,
    ax=ax, cbar_kws={'shrink': 0.6}
)
ax.set_title('Correlation — All Clinical Features + Target',
             fontsize=12, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_PATH / 'correlation_heatmap_clinical.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: correlation_heatmap_clinical.png")

# ================================================================================
# SECTION 6: EGFR vs KRAS Mutual Exclusivity
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 4: EGFR vs KRAS MUTUAL EXCLUSIVITY TEST")
print(f"{'='*60}")

if 'MUT_EGFR' in df.columns and 'MUT_KRAS' in df.columns:
    both      = ((df['MUT_EGFR'] == 1) & (df['MUT_KRAS'] == 1)).sum()
    egfr_only = ((df['MUT_EGFR'] == 1) & (df['MUT_KRAS'] == 0)).sum()
    kras_only = ((df['MUT_EGFR'] == 0) & (df['MUT_KRAS'] == 1)).sum()
    neither   = ((df['MUT_EGFR'] == 0) & (df['MUT_KRAS'] == 0)).sum()
    corr_val  = df['MUT_EGFR'].corr(df['MUT_KRAS'])

    print(f"\n  Co-occurrence table:")
    print(f"  {'':20} KRAS=0        KRAS=1")
    print(f"  EGFR=0        {neither:>8,}      {kras_only:>8,}")
    print(f"  EGFR=1        {egfr_only:>8,}      {both:>8,}   ← both mutated")
    print(f"\n  Pearson r = {corr_val:.3f}")
    print(f"  Both mutated: {both} patients ({both/len(df)*100:.1f}%)")

    if both < 100:
        print(f"\n  ✓ CONFIRMED: Mutually exclusive")
        print(f"    Only {both/len(df)*100:.1f}% of patients have both mutations")
    else:
        print(f"\n  ✗ NOT confirmed: {both} patients have both")

# ================================================================================
# SECTION 7: Genomic Features Correlation
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 5: GENOMIC FEATURES CORRELATION")
print(f"{'='*60}")

genomic_corr = df[TOP_GENOMIC + [TARGET]].corr()

# Print genomic correlation with target
print(f"\n  Top genomic features ranked by |correlation| with TREATMENT_ENC:")
print(f"\n  {'Gene':<12} {'r':>7}  Direction")
print(f"  {'─'*35}")

genomic_target_corr = (genomic_corr[TARGET]
                       .drop(TARGET)
                       .sort_values(key=abs, ascending=False))

for feat, val in genomic_target_corr.items():
    gene      = feat.replace('MUT_', '')
    direction = "→ Chemo/Targeted" if val > 0 else "→ Immunotherapy"
    print(f"  {gene:<12} {val:>7.3f}  {direction}")

# Genomic heatmap — also no annotations for speed
clean_labels = [c.replace('MUT_', '') for c in genomic_corr.columns]

fig, ax = plt.subplots(figsize=(10, 9))
mask = np.triu(np.ones_like(genomic_corr, dtype=bool))
sns.heatmap(
    genomic_corr, mask=mask,
    annot=True, fmt='.2f',    # ← annot=True OK here: only 11x11 = 121 cells
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5,
    annot_kws={'size': 9},
    xticklabels=clean_labels,
    yticklabels=clean_labels,
    ax=ax, cbar_kws={'shrink': 0.7}
)
ax.set_title('Top 10 Genomic Features + Target', fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_PATH / 'correlation_heatmap_genomic.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: correlation_heatmap_genomic.png")

# ================================================================================
# SECTION 8: Redundancy Check
# ================================================================================
print(f"\n{'='*60}")
print("  SECTION 6: REDUNDANCY CHECK  (|r| > 0.7)")
print(f"{'='*60}")

corr_abs = df[ALL_CLINICAL].corr().abs()
high_corr_pairs = []

for i in range(len(corr_abs.columns)):
    for j in range(i + 1, len(corr_abs.columns)):
        val = corr_abs.iloc[i, j]
        if val > 0.7:
            high_corr_pairs.append({
                'feature_1':   corr_abs.columns[i],
                'feature_2':   corr_abs.columns[j],
                'correlation': round(val, 3)
            })

if high_corr_pairs:
    print(f"\n  ⚠ Redundant feature pairs (|r| > 0.7):")
    for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True):
        print(f"    {pair['feature_1']:<30} ↔  "
              f"{pair['feature_2']:<30}  r={pair['correlation']:.3f}")
else:
    print(f"\n  ✓ No redundant pairs — all |r| < 0.7")

# ================================================================================
# SECTION 9: Summary
# ================================================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

top5 = target_corr.abs().head(5)
print(f"\n  Top 5 clinical features for treatment prediction:")
for feat, val in top5.items():
    print(f"    {feat:<30} |r|={val:.3f}")

print(f"\n  Figures saved:")
print(f"    figures/eda/correlation_numeric.png")
print(f"    figures/eda/correlation_heatmap_clinical.png")
print(f"    figures/eda/correlation_heatmap_genomic.png")

print(f"\n✓ Notebook 03 complete")
