"""
================================================================================
Phase 2 — EDA
Notebook 04: Treatment Group Comparison
================================================================================

GOAL:
    Test whether each feature differs significantly across the three
    treatment groups (Immunotherapy / Chemotherapy / Targeted).

    This answers: "Is the difference we see real, or just random noise?"

METHODS:
    Kruskal-Wallis → non-parametric test for numeric features
                     (does not assume normal distribution)
    Chi-squared    → test for binary/categorical features
                     (are the proportions different across groups?)

KEY CONCEPTS:
    p-value      → probability the difference is due to chance
                   p < 0.05 = statistically significant
    Effect size  → HOW BIG is the difference (not just whether it exists)
                   Eta-squared (η²) for numeric, Cramér's V for categorical
================================================================================
"""

# ================================================================================
# SECTION 1: Imports and Setup
# ================================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

TREATMENT_COLORS = {
    'Immunotherapy': '#2196F3',
    'Chemotherapy':  '#FF9800',
    'Targeted':      '#4CAF50',
}
TREATMENT_ORDER = ['Immunotherapy', 'Chemotherapy', 'Targeted']

FIG_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/figures/eda')
FIG_PATH.mkdir(parents=True, exist_ok=True)

print("✓ Libraries loaded")

# ================================================================================
# SECTION 2: Load Data
# ================================================================================
DATA_PATH = Path('C:/Users/farza/Job/Github/nsclc-treatment-prediction/data/Processed Dataset/nsclc_final.csv')
df = pd.read_csv(DATA_PATH)

# Convert bool columns to float
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(float)

# Split into three groups — used repeatedly throughout the notebook
groups = {
    name: df[df['TREATMENT_FINAL'] == name]
    for name in TREATMENT_ORDER
}

print(f"✓ Loaded: {df.shape[0]:,} patients")
for name, grp in groups.items():
    print(f"  {name:<20} {len(grp):,} patients")

# ── Feature definitions ────────────────────────────────────────────────────────
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

NUMERIC_COLS    = [c for c in NUMERIC_COLS    if c in df.columns]
BINARY_CLINICAL = [c for c in BINARY_CLINICAL if c in df.columns]
TOP_GENOMIC     = [c for c in TOP_GENOMIC     if c in df.columns]

# ================================================================================
# SECTION 3: Helper Functions
# ================================================================================

def kruskal_wallis_test(col, groups):
    """
    Kruskal-Wallis test: are the distributions of 'col' different
    across the three treatment groups?

    HOW IT WORKS:
        1. Rank all values from 1 to N across all groups combined
        2. Check if the average rank differs between groups
        3. If one group consistently has higher/lower ranks, the test detects it

    WHY NOT ANOVA?
        ANOVA assumes normal distribution. Our features (TMB, MSI) are
        heavily skewed. Kruskal-Wallis makes no distributional assumption.

    Returns:
        stat    : test statistic (higher = more different)
        p_value : probability the difference is due to chance
        eta_sq  : effect size (how large is the difference)
    """
    group_arrays = [grp[col].dropna().values for grp in groups.values()]

    # stats.kruskal() returns (H-statistic, p-value)
    # H-statistic follows a chi-squared distribution under the null hypothesis
    stat, p_value = stats.kruskal(*group_arrays)

    # Eta-squared: proportion of variance explained by group membership
    # Formula: (H - k + 1) / (n - k)  where k=number of groups, n=total samples
    n = sum(len(a) for a in group_arrays)
    k = len(group_arrays)
    eta_sq = (stat - k + 1) / (n - k)
    eta_sq = max(0, eta_sq)   # clip to 0 if negative (numerical artifact)

    return round(stat, 3), round(p_value, 6), round(eta_sq, 4)


def chi_squared_test(col, df):
    """
    Chi-squared test: are the proportions of 'col' different
    across treatment groups?

    HOW IT WORKS:
        1. Build a contingency table: rows=treatment, cols=feature values
        2. Compare observed counts to what we'd expect if there were NO difference
        3. Large deviation from expected → small p-value → significant difference

    Used for binary and categorical features (not numeric ones).

    Returns:
        chi2    : test statistic
        p_value : probability the difference is due to chance
        cramers_v : effect size (0=no association, 1=perfect association)
    """
    contingency = pd.crosstab(df['TREATMENT_FINAL'], df[col])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    # Cramér's V: normalized effect size for chi-squared tests
    n   = contingency.sum().sum()
    phi = chi2 / n
    r, k = contingency.shape
    v   = np.sqrt(phi / min(r - 1, k - 1))

    return round(chi2, 3), round(p_value, 6), round(v, 4)


def interpret_effect(eta_sq_or_v):
    """Interpret effect size using standard thresholds."""
    if eta_sq_or_v < 0.01:
        return "negligible"
    elif eta_sq_or_v < 0.06:
        return "small"
    elif eta_sq_or_v < 0.14:
        return "medium"
    else:
        return "large"


# ================================================================================
# SECTION 4: Numeric Features — Kruskal-Wallis Test
# ================================================================================
print(f"\n{'='*70}")
print("  SECTION 1: NUMERIC FEATURES — KRUSKAL-WALLIS TEST")
print(f"{'='*70}")
print(f"\n  {'Feature':<25} {'H-stat':>8}  {'p-value':>10}  {'η²':>7}  "
      f"{'Effect':>10}  Significant?")
print(f"  {'─'*75}")

numeric_results = []
for col in NUMERIC_COLS:
    stat, pval, eta = kruskal_wallis_test(col, groups)
    effect  = interpret_effect(eta)
    sig     = "✓ Yes" if pval < 0.05 else "✗ No"
    numeric_results.append({
        'feature': col, 'stat': stat,
        'p_value': pval, 'eta_sq': eta, 'effect': effect
    })
    print(f"  {col:<25} {stat:>8.3f}  {pval:>10.6f}  {eta:>7.4f}  "
          f"{effect:>10}  {sig}")

# Print group means for context
print(f"\n  Group means per feature:")
print(f"  {'Feature':<25} {'Immunotherapy':>15} {'Chemotherapy':>14} {'Targeted':>10}")
print(f"  {'─'*70}")
for col in NUMERIC_COLS:
    means = [groups[name][col].mean() for name in TREATMENT_ORDER]
    print(f"  {col:<25} {means[0]:>15.3f} {means[1]:>14.3f} {means[2]:>10.3f}")

# ================================================================================
# SECTION 5: Boxplot — Numeric Features by Treatment Group
# ================================================================================
print(f"\n  Generating boxplots...")

fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(5 * len(NUMERIC_COLS), 5))

for i, col in enumerate(NUMERIC_COLS):
    ax = axes[i]

    # Prepare data for boxplot: list of arrays, one per group
    data_to_plot = [groups[name][col].dropna().values for name in TREATMENT_ORDER]

    bp = ax.boxplot(
        data_to_plot,
        tick_labels=TREATMENT_ORDER,
        patch_artist=True,    # fill boxes with color
        medianprops={'color': 'black', 'linewidth': 2},
        flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3}
    )

    # Color each box by treatment group
    for patch, name in zip(bp['boxes'], TREATMENT_ORDER):
        patch.set_facecolor(TREATMENT_COLORS[name])
        patch.set_alpha(0.7)

    # Get the test result for this feature
    result = next(r for r in numeric_results if r['feature'] == col)
    sig_star = "***" if result['p_value'] < 0.001 else \
               "**"  if result['p_value'] < 0.01  else \
               "*"   if result['p_value'] < 0.05  else "ns"

    ax.set_title(f"{col}\np={result['p_value']:.4f} {sig_star}  "
                 f"η²={result['eta_sq']:.3f} ({result['effect']})",
                 fontsize=9, fontweight='bold')
    ax.set_ylabel('Value', fontsize=8)
    ax.tick_params(axis='x', labelrotation=15, labelsize=8)

plt.suptitle('Numeric Features by Treatment Group (Kruskal-Wallis)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_PATH / 'treatment_comparison_numeric.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: treatment_comparison_numeric.png")

# ================================================================================
# SECTION 6: Binary Clinical Features — Chi-Squared Test
# ================================================================================
print(f"\n{'='*70}")
print("  SECTION 2: BINARY CLINICAL FEATURES — CHI-SQUARED TEST")
print(f"{'='*70}")
print(f"\n  {'Feature':<30} {'χ²':>8}  {'p-value':>10}  {'V':>7}  "
      f"{'Effect':>10}  Significant?")
print(f"  {'─'*75}")

binary_results = []
for col in BINARY_CLINICAL:
    chi2, pval, v = chi_squared_test(col, df)
    effect = interpret_effect(v)
    sig    = "✓ Yes" if pval < 0.05 else "✗ No"
    binary_results.append({
        'feature': col, 'chi2': chi2,
        'p_value': pval, 'cramers_v': v, 'effect': effect
    })
    print(f"  {col:<30} {chi2:>8.3f}  {pval:>10.6f}  {v:>7.4f}  "
          f"{effect:>10}  {sig}")

# Print proportions per group for significant features
sig_binary = [r for r in binary_results if r['p_value'] < 0.05]
print(f"\n  Proportions for significant binary features (% with value=1):")
print(f"  {'Feature':<30} {'Immuno':>8} {'Chemo':>8} {'Targeted':>10}")
print(f"  {'─'*60}")
for result in sorted(sig_binary, key=lambda x: x['cramers_v'], reverse=True):
    col   = result['feature']
    props = [groups[name][col].mean() * 100 for name in TREATMENT_ORDER]
    print(f"  {col:<30} {props[0]:>7.1f}% {props[1]:>7.1f}% {props[2]:>9.1f}%")

# ================================================================================
# SECTION 7: Genomic Features — Chi-Squared Test
# ================================================================================
print(f"\n{'='*70}")
print("  SECTION 3: GENOMIC FEATURES — CHI-SQUARED TEST")
print(f"{'='*70}")
print(f"\n  {'Gene':<15} {'χ²':>8}  {'p-value':>10}  {'V':>7}  "
      f"{'Effect':>10}  Significant?")
print(f"  {'─'*65}")

genomic_results = []
for col in TOP_GENOMIC:
    chi2, pval, v = chi_squared_test(col, df)
    effect = interpret_effect(v)
    sig    = "✓ Yes" if pval < 0.05 else "✗ No"
    gene   = col.replace('MUT_', '')
    genomic_results.append({
        'feature': col, 'gene': gene, 'chi2': chi2,
        'p_value': pval, 'cramers_v': v, 'effect': effect
    })
    print(f"  {gene:<15} {chi2:>8.3f}  {pval:>10.6f}  {v:>7.4f}  "
          f"{effect:>10}  {sig}")

# Mutation rates per treatment group
print(f"\n  Mutation rates per treatment group (%):")
print(f"  {'Gene':<15} {'Immuno':>8} {'Chemo':>8} {'Targeted':>10}")
print(f"  {'─'*45}")
for result in sorted(genomic_results,
                     key=lambda x: x['cramers_v'], reverse=True):
    col   = result['feature']
    rates = [groups[name][col].mean() * 100 for name in TREATMENT_ORDER]
    print(f"  {result['gene']:<15} {rates[0]:>7.1f}% {rates[1]:>7.1f}% "
          f"{rates[2]:>9.1f}%")

# ================================================================================
# SECTION 8: Bar Chart — Mutation Rates by Treatment
# ================================================================================
print(f"\n  Generating mutation rate bar chart...")

# Build a DataFrame of mutation rates per group
mutation_rates = pd.DataFrame({
    name: [groups[name][col].mean() * 100 for col in TOP_GENOMIC]
    for name in TREATMENT_ORDER
}, index=[c.replace('MUT_', '') for c in TOP_GENOMIC])

fig, ax = plt.subplots(figsize=(12, 6))

x     = np.arange(len(mutation_rates))
width = 0.25    # width of each bar

for i, (name, color) in enumerate(TREATMENT_COLORS.items()):
    ax.bar(x + i * width, mutation_rates[name],
           width=width, label=name, color=color, alpha=0.8)

ax.set_xlabel('Gene', fontsize=10)
ax.set_ylabel('Mutation Rate (%)', fontsize=10)
ax.set_title('Mutation Rates by Treatment Group — Top 10 Genes',
             fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(mutation_rates.index, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig(FIG_PATH / 'treatment_comparison_genomic.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: treatment_comparison_genomic.png")

# ================================================================================
# SECTION 9: Summary Table — All Features Ranked by Effect Size
# ================================================================================
print(f"\n{'='*70}")
print("  SUMMARY — ALL FEATURES RANKED BY EFFECT SIZE")
print(f"{'='*70}")

# Combine all results
all_results = []
for r in numeric_results:
    all_results.append({
        'feature': r['feature'], 'type': 'numeric',
        'p_value': r['p_value'], 'effect_size': r['eta_sq'],
        'effect': r['effect']
    })
for r in binary_results:
    all_results.append({
        'feature': r['feature'], 'type': 'binary',
        'p_value': r['p_value'], 'effect_size': r['cramers_v'],
        'effect': r['effect']
    })
for r in genomic_results:
    all_results.append({
        'feature': r['gene'], 'type': 'genomic',
        'p_value': r['p_value'], 'effect_size': r['cramers_v'],
        'effect': r['effect']
    })

# Sort by effect size
all_results.sort(key=lambda x: x['effect_size'], reverse=True)

print(f"\n  {'Rank':<5} {'Feature':<30} {'Type':<10} "
      f"{'Effect Size':>12}  {'Effect':>10}  {'p-value':>10}")
print(f"  {'─'*80}")
for i, r in enumerate(all_results[:15], 1):
    sig = "✓" if r['p_value'] < 0.05 else " "
    print(f"  {i:<5} {r['feature']:<30} {r['type']:<10} "
          f"{r['effect_size']:>12.4f}  {r['effect']:>10}  "
          f"{r['p_value']:>10.6f} {sig}")

# Features with no significant difference
not_sig = [r for r in all_results if r['p_value'] >= 0.05]
if not_sig:
    print(f"\n  Features with NO significant difference (p ≥ 0.05):")
    for r in not_sig:
        print(f"    • {r['feature']:<30} p={r['p_value']:.4f}")
    print(f"\n  → These features may not help the model distinguish treatment groups")

print(f"\n✓ Notebook 04 complete — ready for notebook 05 (Survival Analysis)")
