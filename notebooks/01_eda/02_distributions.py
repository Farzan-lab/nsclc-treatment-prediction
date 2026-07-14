"""
================================================================================
Phase 2 — Exploratory Data Analysis (EDA)
Notebook 02: Distribution Analysis
================================================================================

GOAL:
    Visualize the actual shape of every numeric feature.
    Answer three questions for each feature:
        1. What does the distribution look like?
        2. Does it need a log-transform before modeling?
        3. Does its distribution differ across the three treatment groups?

KEY CONCEPT — Why distributions matter:
    Most ML models assume features are roughly normally distributed.
    A heavily skewed feature (like TMB with skew=4.47) can dominate the model
    simply because of its scale, not because it is truly more important.
    Transforming skewed features makes the model focus on signal, not scale.
================================================================================
"""

# ================================================================================
# SECTION 1: Imports and Setup
# ================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.3f}'.format)

# Color palette for three treatment groups
# Using colorblind-safe colors
TREATMENT_COLORS = {
    'Immunotherapy': '#2196F3',   # blue
    'Chemotherapy':  '#FF9800',   # orange
    'Targeted':      '#4CAF50',   # green
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

# Separate dataset by treatment group for per-group plotting
# This creates three smaller DataFrames, one per treatment
groups = {
    name: df[df['TREATMENT_FINAL'] == name]
    for name in ['Immunotherapy', 'Chemotherapy', 'Targeted']
}
for name, grp in groups.items():
    print(f"  {name:<20} {len(grp):,} patients")

# ================================================================================
# SECTION 3: Helper Functions
# ================================================================================

def plot_feature_distribution(col, df, groups, ax_hist, ax_kde,
                               log_scale=False, title_suffix=''):
    """
    Plot histogram (overall) and KDE (per treatment group) for one feature.

    Parameters:
        col          : column name to plot
        df           : full DataFrame
        groups       : dict of {treatment_name: sub-DataFrame}
        ax_hist      : matplotlib Axes for the histogram
        ax_kde       : matplotlib Axes for the KDE plot
        log_scale    : if True, apply log1p transform before plotting
        title_suffix : extra text added to plot title
    """
    # Apply log1p transform if requested
    # log1p(x) = log(x + 1) — the +1 handles zero values safely
    # Regular log(0) = -infinity, but log1p(0) = 0
    if log_scale:
        # We create a temporary transformed series just for plotting
        # The original data is NOT modified
        values = np.log1p(df[col].clip(lower=0))
        group_values = {
            name: np.log1p(grp[col].clip(lower=0))
            for name, grp in groups.items()
        }
        xlabel = f'log1p({col})'
    else:
        values = df[col].dropna()
        group_values = {name: grp[col].dropna() for name, grp in groups.items()}
        xlabel = col

    # ── LEFT PLOT: Overall Histogram ──────────────────────────────────────────
    # bins=50 divides the range into 50 equal-width bars
    # edgecolor adds a thin border around each bar for clarity
    ax_hist.hist(values, bins=50, color='#607D8B', edgecolor='white',
                 linewidth=0.3, alpha=0.85)

    # Add vertical lines for mean and median
    # This visually shows the gap we calculated in notebook 01
    mean_val   = values.mean()
    median_val = values.median()
    ax_hist.axvline(mean_val,   color='#E53935', linestyle='--',
                    linewidth=1.5, label=f'Mean={mean_val:.2f}')
    ax_hist.axvline(median_val, color='#1E88E5', linestyle='-',
                    linewidth=1.5, label=f'Median={median_val:.2f}')

    ax_hist.set_xlabel(xlabel, fontsize=9)
    ax_hist.set_ylabel('Number of patients', fontsize=9)
    ax_hist.set_title(f'{col}{title_suffix}\n(Overall Distribution)',
                      fontsize=10, fontweight='bold')
    ax_hist.legend(fontsize=8)

    # ── RIGHT PLOT: KDE per Treatment Group ───────────────────────────────────
    # KDE (Kernel Density Estimation) smooths the histogram into a continuous curve
    # Plotting all three treatment groups on the same axes lets us see:
    #   - Do the distributions overlap? (feature may not separate groups well)
    #   - Are the distributions shifted? (feature may help separate groups)
    for name, vals in group_values.items():
        if len(vals) < 10:
            continue    # skip if too few values to estimate density

        # BUG FIX: Previously we called both ax_kde.plot([], []) AND vals.plot.kde()
        # which added TWO entries per group to the legend:
        #   one for the empty plot() line  → showed treatment name correctly
        #   one for the KDE curve          → showed the column name (unwanted)
        #
        # FIX: Use scipy gaussian_kde directly to draw the curve manually.
        # This gives full control — exactly ONE legend entry per treatment group.
        from scipy.stats import gaussian_kde

        kde_estimator = gaussian_kde(vals)
        x_range   = np.linspace(vals.min(), vals.max(), 300)
        y_density = kde_estimator(x_range)

        # label= adds exactly one legend entry per group
        ax_kde.plot(x_range, y_density,
                    color=TREATMENT_COLORS[name], linewidth=2,
                    label=f'{name} (n={len(vals):,})')

        # shade area under the curve for visual clarity
        ax_kde.fill_between(x_range, y_density,
                            alpha=0.1, color=TREATMENT_COLORS[name])

    ax_kde.set_xlabel(xlabel, fontsize=9)
    ax_kde.set_ylabel('Density', fontsize=9)
    ax_kde.set_title(f'{col}\n(KDE by Treatment Group)', fontsize=10,
                     fontweight='bold')
    ax_kde.legend(fontsize=8)


def summarize_transform_decision(col, original_skew, transformed_skew=None):
    """Print a clear decision about whether to transform this feature."""
    print(f"\n  {col}:")
    print(f"    Original skewness : {original_skew:.3f}")
    if transformed_skew is not None:
        print(f"    After transform   : {transformed_skew:.3f}")
        improvement = abs(original_skew) - abs(transformed_skew)
        print(f"    Improvement       : {improvement:.3f}")
        if abs(transformed_skew) < 0.5:
            print(f"    Decision          : ✓ USE TRANSFORMED VERSION")
        elif abs(transformed_skew) < abs(original_skew) * 0.5:
            print(f"    Decision          : ⚡ TRANSFORM HELPS — use transformed")
        else:
            print(f"    Decision          : ✗ TRANSFORM DOES NOT HELP ENOUGH")
    else:
        if abs(original_skew) < 0.5:
            print(f"    Decision          : ✓ NO TRANSFORM NEEDED")
        elif abs(original_skew) < 1.0:
            print(f"    Decision          : ⚡ MILD SKEW — transform optional")
        else:
            print(f"    Decision          : ⚠ NEEDS TRANSFORM")


# ================================================================================
# SECTION 4: Plot All Numeric Features
# ================================================================================
# We plot each feature as a pair: histogram (left) + KDE by treatment (right)
# Features that already have a log-transformed version get an extra comparison row

NUMERIC_COLS = [
    'CURRENT_AGE_DEID',
    'TMB_NONSYNONYMOUS',
    'MSI_SCORE',
    'TUMOR_PURITY',
    'OS_MONTHS',
]
# Note: TMB_LOG, MSI_NORM handled separately in Section 5

print(f"\n{'='*60}")
print("  DISTRIBUTION PLOTS — All Numeric Features")
print(f"{'='*60}")

# Create a figure with one row per feature, two columns (hist + kde)
# figsize: width=14 inches, height=4 inches per feature
fig, axes = plt.subplots(
    nrows=len(NUMERIC_COLS),
    ncols=2,
    figsize=(14, 4 * len(NUMERIC_COLS))
)

for i, col in enumerate(NUMERIC_COLS):
    skew = df[col].skew()
    print(f"\n  Plotting: {col}  (skew={skew:.3f})")

    plot_feature_distribution(
        col=col,
        df=df,
        groups=groups,
        ax_hist=axes[i, 0],
        ax_kde=axes[i, 1],
    )

plt.suptitle('Distribution of All Numeric Features', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIG_PATH / 'distributions_numeric.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\n✓ Saved: figures/eda/distributions_numeric.png")


# ================================================================================
# SECTION 5: Transform Comparison — TMB and MSI
# ================================================================================
# For features that already have pre-computed transforms (TMB_LOG, MSI_NORM),
# we compare the original vs transformed distribution side by side.
# This validates whether the transform actually improved the shape.

print(f"\n{'='*60}")
print("  TRANSFORM COMPARISON — TMB and MSI")
print(f"{'='*60}")

# Figure with 2 rows (TMB, MSI) × 3 columns (original, transformed, KDE comparison)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

# ── Row 1: TMB ────────────────────────────────────────────────────────────────
tmb_raw  = df['TMB_NONSYNONYMOUS'].dropna()
tmb_log  = df['TMB_LOG'].dropna()

# Original TMB histogram
axes[0, 0].hist(tmb_raw, bins=50, color='#EF9A9A', edgecolor='white', linewidth=0.3)
axes[0, 0].axvline(tmb_raw.mean(),   color='red',  linestyle='--', linewidth=1.5,
                    label=f'Mean={tmb_raw.mean():.1f}')
axes[0, 0].axvline(tmb_raw.median(), color='blue', linestyle='-',  linewidth=1.5,
                    label=f'Median={tmb_raw.median():.1f}')
axes[0, 0].set_title(f'TMB Raw\nskew={tmb_raw.skew():.2f}  ⚠ right-skewed',
                      fontweight='bold')
axes[0, 0].set_xlabel('TMB (mutations/Mb)')
axes[0, 0].legend(fontsize=8)

# Log-transformed TMB histogram
axes[0, 1].hist(tmb_log, bins=50, color='#A5D6A7', edgecolor='white', linewidth=0.3)
axes[0, 1].axvline(tmb_log.mean(),   color='red',  linestyle='--', linewidth=1.5,
                    label=f'Mean={tmb_log.mean():.2f}')
axes[0, 1].axvline(tmb_log.median(), color='blue', linestyle='-',  linewidth=1.5,
                    label=f'Median={tmb_log.median():.2f}')
axes[0, 1].set_title(f'TMB_LOG\nskew={tmb_log.skew():.2f}  ✓ symmetric',
                      fontweight='bold')
axes[0, 1].set_xlabel('log(TMB + 1)')
axes[0, 1].legend(fontsize=8)

# KDE comparison: TMB_LOG by treatment group
for name, grp in groups.items():
    grp['TMB_LOG'].dropna().plot.kde(
        ax=axes[0, 2], color=TREATMENT_COLORS[name],
        linewidth=2, label=name
    )
axes[0, 2].set_title('TMB_LOG — KDE by Treatment', fontweight='bold')
axes[0, 2].set_xlabel('log(TMB + 1)')
axes[0, 2].legend(fontsize=8)

# ── Row 2: MSI ────────────────────────────────────────────────────────────────
msi_raw  = df['MSI_SCORE'].dropna()
msi_norm = df['MSI_NORM'].dropna()

# Original MSI histogram
# NOTE: MSI_SCORE has values of -1.0 — these are sentinel values
# meaning "test not performed" or "inconclusive result"
n_negative = (msi_raw < 0).sum()
axes[1, 0].hist(msi_raw, bins=50, color='#EF9A9A', edgecolor='white', linewidth=0.3)
axes[1, 0].axvline(msi_raw.mean(),   color='red',  linestyle='--', linewidth=1.5,
                    label=f'Mean={msi_raw.mean():.2f}')
axes[1, 0].axvline(msi_raw.median(), color='blue', linestyle='-',  linewidth=1.5,
                    label=f'Median={msi_raw.median():.2f}')
axes[1, 0].set_title(f'MSI_SCORE Raw\nskew={msi_raw.skew():.2f}  '
                      f'(n_negative={n_negative})',
                      fontweight='bold')
axes[1, 0].set_xlabel('MSI Score')
axes[1, 0].legend(fontsize=8)

# MSI_NORM histogram
axes[1, 1].hist(msi_norm, bins=50, color='#A5D6A7', edgecolor='white', linewidth=0.3)
axes[1, 1].axvline(msi_norm.mean(),   color='red',  linestyle='--', linewidth=1.5,
                    label=f'Mean={msi_norm.mean():.2f}')
axes[1, 1].axvline(msi_norm.median(), color='blue', linestyle='-',  linewidth=1.5,
                    label=f'Median={msi_norm.median():.2f}')
axes[1, 1].set_title(f'MSI_NORM\nskew={msi_norm.skew():.2f}',
                      fontweight='bold')
axes[1, 1].set_xlabel('MSI Score (normalized)')
axes[1, 1].legend(fontsize=8)

# KDE comparison: MSI_NORM by treatment group
for name, grp in groups.items():
    grp['MSI_NORM'].dropna().plot.kde(
        ax=axes[1, 2], color=TREATMENT_COLORS[name],
        linewidth=2, label=name
    )
axes[1, 2].set_title('MSI_NORM — KDE by Treatment', fontweight='bold')
axes[1, 2].set_xlabel('MSI Score (normalized)')
axes[1, 2].legend(fontsize=8)

plt.suptitle('Transform Comparison: TMB and MSI', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH / 'transform_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"✓ Saved: figures/eda/transform_comparison.png")


# ================================================================================
# SECTION 6: MSI Deep Dive — Negative Values
# ================================================================================
# In notebook 01 we saw MSI_SCORE has min = -1.0
# Negative MSI scores do NOT have a biological meaning.
# They are sentinel values — a special code used to flag:
#   "MSI test was performed but the result was inconclusive or should not be reported"
# We need to understand how many patients have this and what to do about them.

print(f"\n{'='*60}")
print("  MSI DEEP DIVE — Negative Values Investigation")
print(f"{'='*60}")

msi_negative = df[df['MSI_SCORE'] < 0]
msi_zero     = df[df['MSI_SCORE'] == 0]
msi_positive = df[df['MSI_SCORE'] > 0]

print(f"\n  MSI_SCORE value ranges:")
print(f"    Negative (< 0) : {len(msi_negative):,} patients  "
      f"({len(msi_negative)/len(df)*100:.1f}%)")
print(f"    Zero (= 0)     : {len(msi_zero):,} patients  "
      f"({len(msi_zero)/len(df)*100:.1f}%)")
print(f"    Positive (> 0) : {len(msi_positive):,} patients  "
      f"({len(msi_positive)/len(df)*100:.1f}%)")

# Check MSI_NORM for the same patients
print(f"\n  MSI_NORM for patients with MSI_SCORE < 0:")
print(df.loc[df['MSI_SCORE'] < 0, 'MSI_NORM'].value_counts().head())

# Treatment distribution of patients with negative MSI
print(f"\n  Treatment distribution of patients with MSI_SCORE < 0:")
print(msi_negative['TREATMENT_FINAL'].value_counts())

# Decision
print(f"\n  ── DECISION ──────────────────────────────────────")
print(f"  MSI_NORM clips negative values to 0 (min=0.0 confirmed).")
print(f"  This is the correct approach: -1.0 → 0.0 (treated as MSS)")
print(f"  → USE MSI_NORM in the model, NOT MSI_SCORE")
print(f"  → MSI_SCORE kept only for reference/validation")


# ================================================================================
# SECTION 7: Transform Decision Summary
# ================================================================================
print(f"\n{'='*60}")
print("  TRANSFORM DECISION SUMMARY")
print(f"{'='*60}")

print("""
  Feature              Original Skew   Decision
  ────────────────────────────────────────────────────────
  CURRENT_AGE_DEID     -0.606          ✓ Use as-is
  TMB_NONSYNONYMOUS     4.479          ✗ Do NOT use — use TMB_LOG instead
  TMB_LOG              -0.098          ✓ Use as-is (symmetric)
  MSI_SCORE             7.464          ✗ Do NOT use — use MSI_NORM instead
  MSI_NORM              7.520          ⚠ Still skewed but negative values handled
                                         → apply log1p in preprocessing
  TUMOR_PURITY          0.775          ✓ Use as-is (mild skew, acceptable)
  OS_MONTHS             1.078          → survival analysis only, not a model feature
""")

# ================================================================================
# SECTION 8: FINAL Clinical Feature Set
# ================================================================================
print(f"{'='*60}")
print("  FINAL CLINICAL FEATURE SET FOR MODELING")
print(f"{'='*60}")
print("""
  After distribution analysis, the final 18 clinical features are:

  Numeric (continuous):
    CURRENT_AGE_DEID    ← use as-is
    TMB_LOG             ← use instead of TMB_NONSYNONYMOUS
    MSI_NORM            ← use instead of MSI_SCORE (log1p in preprocessing)
    TUMOR_PURITY        ← use as-is

  Ordinal / Encoded:
    STAGE_ENC
    SMOKING_ENC
    PRIOR_MED_ENC
    HAS_PROGRESSION

  Binary — Subtype:
    SUBTYPE_Adenocarcinoma
    SUBTYPE_Squamous
    SUBTYPE_Neuroendocrine
    SUBTYPE_Other_NSCLC

  Binary — Metastasis:
    BONE
    CNS_BRAIN
    LIVER
    LUNG
    LYMPH_NODES

  Demographics:
    GENDER_ENC

  Total: 18 clinical features
""")

print("✓ Notebook 02 complete — ready for notebook 03 (Correlation Analysis)")
